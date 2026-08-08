//! **Titanium v15 engine session** — two-thread design: I/O on main thread,
//! search daemon thread.
//!
//! Titanium v15 is the production engine: grafts Titanium O1 movegen,
//! adaptive TT, win-certificate solver, and incremental HalfPW accumulator
//! onto the gen13 search core.  This session adds continuous-search support
//! on top of the standard game-server protocol.
//!
//! ## Protocol
//!
//! ### Standard commands (compatible with self_match.js and run_overnight.bat)
//!   reset / position [MOVES] / makemove MOVE / go TIME_SEC / quit
//!
//! ### Titanium v15 infinite-search extensions (disabled — not routed in main.rs)
//!   go infinite [PONDER_MOVE]   — start pondering; applies PONDER_MOVE first if given
//!   stop                        — stop pondering; replies "bestmove MOVE"
//!   ponderhit TIME_MS           — ponder move was correct; think for TIME_MS; replies "bestmove MOVE"
//!   movemiss MOVE TIME_MS       — opponent played MOVE (unexpected);
//!                                 migrate root and think for TIME_MS; replies "bestmove MOVE"
//!
//! For `go TIME_SEC` the daemon does a single blocking think and replies
//! "bestmove MOVE" — the standard path used by self_match.js.

use std::io::{self, BufRead, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;

use crate::titanium::{
    algebraic_to_move_id, move_id_to_algebraic, GameState, ThinkResult, TitaniumSearch,
    TITANIUM_NO_MOVE,
};

// ── Inter-thread messages ─────────────────────────────────────────────────────

enum Cmd {
    /// Replace the engine position (no I/O reply needed — I/O thread handles "ready").
    SetGame(GameState),
    /// Timed search: think for `time_ms` and reply BestMove.
    GoTimed(u64),
    /// Start pondering on the current position (pre-apply `ponder_mv` if given).
    GoInfinite(i16),
    /// Stop pondering; reply with the last best move from the ponder search.
    StopAndGet,
    /// Ponder was correct: think for `time_ms` from the current (ponder) position.
    PonderHit(u64),
    /// Ponder was wrong: reset to `new_game` then think for `time_ms`.
    MoveMiss {
        new_game: GameState,
        time_ms: u64,
    },
    Quit,
}

/// Work done while pondering, attributed to the think that consumed it.
///
/// Carried on `Reply::BestMove` rather than sent as its own message: as a
/// separate reply it raced the bestmove it belonged to and usually landed on
/// the *next* move's telemetry instead.
#[derive(Clone, Copy)]
struct PonderTelemetry {
    nodes: u64,
    chunks: u32,
    ms: u64,
    /// Opponent reply we speculatively stood on (`TITANIUM_NO_MOVE` = none).
    predicted: i16,
    /// Whether the opponent actually played `predicted`.
    hit: bool,
}

impl Default for PonderTelemetry {
    fn default() -> Self {
        // Hand-written: a derived Default would leave `predicted` at move id 0,
        // which is a real move ("a9") rather than "nothing was predicted".
        Self {
            nodes: 0,
            chunks: 0,
            ms: 0,
            predicted: TITANIUM_NO_MOVE,
            hit: false,
        }
    }
}

enum Reply {
    BestMove(i16, Option<Box<ThinkResult>>, PonderTelemetry),
    Error(String),
}

/// Effectively infinite ponder budget. The search is ended by the abort flag
/// when a command arrives, not by this deadline -- it exists only so `think`
/// has a number. Pondering is a genuine continuous search, as in Stockfish,
/// rather than a stream of restarts.
const PONDER_INFINITE_MS: u64 = 3_600_000;

/// A ponder search returning more than this far short of its budget has solved
/// the position out rather than run out of time; there is nothing left to
/// compute and re-running it would spin.
const PONDER_EXHAUSTED_MARGIN_MS: u64 = 1_000;

/// Poll interval once the position is exhausted. Sleeping rather than
/// re-searching keeps an exhausted ponder off the CPU, which matters because
/// the core it would burn is one the opponent's search is entitled to.
const PONDER_IDLE_POLL_MS: u64 = 20;

/// Budget for choosing which reply to ponder on.
///
/// The guess is committed to for the whole ponder window, so a cheap wrong
/// guess costs far more than the probe that would have avoided it. Measured
/// hit rate in real games: 72%.
const PONDER_PROBE_MS: u64 = 400;

// ── Search daemon ─────────────────────────────────────────────────────────────

fn build_search(_engine_flag: &str, g: GameState) -> Box<TitaniumSearch> {
    // One engine; the label no longer selects a configuration.
    TitaniumSearch::production(g, None)
}

fn search_daemon(
    engine_flag: String,
    rx: Receiver<Cmd>,
    tx: Sender<Reply>,
    abort: Arc<AtomicBool>,
) {
    let mut search = build_search(&engine_flag, GameState::new());
    search.set_external_stop(Arc::clone(&abort));
    let mut last_score: i32 = 0;
    let label = engine_flag.as_str();
    // Mirror of the position the caller last set, so pondering can build the
    // speculative child position and recognise a correct prediction.
    let mut cur_g = GameState::new();

    loop {
        let cmd = match rx.recv() {
            Ok(c) => c,
            Err(_) => return,
        };
        // The I/O thread raises `abort` before EVERY command to cut a ponder
        // search short. Clear it here, once, so the search this command asks
        // for is allowed to run: previously only GoTimed cleared it, so
        // PonderHit/MoveMiss searches aborted after ~63 nodes and returned no
        // move at all.
        abort.store(false, Ordering::Relaxed);
        match cmd {
            Cmd::SetGame(g) => {
                cur_g = g.clone();
                search.set_position(g);
            }
            Cmd::GoTimed(time_ms) => {
                let r = search.think(time_ms, 128, false, true, label);
                last_score = r.score;
                let mv = r.mv;
                let _ = tx.send(Reply::BestMove(mv, Some(Box::new(r)), PonderTelemetry::default()));
            }
            Cmd::GoInfinite(hint_mv) => {
                // Ponder mode: tt_gen and history are frozen for the whole
                // session so the TT entries pondering builds stay fresh and
                // history accumulates.
                search.set_pondering(true);
                let mut tel = PonderTelemetry::default();
                let mut last_mv = TITANIUM_NO_MOVE;
                // Hash of the speculative position we are standing on, if any.
                let mut spec_hash: Option<(u32, u32)> = None;
                let mut hint = hint_mv;
                let mut respeculate = true;
                // The search has nothing left to compute at this position, so
                // re-running it would spin. Set when a think returns far short
                // of the budget it was given.
                let mut exhausted = false;

                loop {
                    // NEVER clear the stop flag speculatively. The I/O thread
                    // raises it BEFORE putting the command on the channel, so
                    // there is a window where the flag is set and the queue is
                    // still empty. Clearing it there and re-entering an infinite
                    // search parks the daemon for an hour on a command that
                    // lands a microsecond later -- which is exactly how this
                    // deadlocked, twice. The flag is cleared only once a command
                    // is actually in hand, below.
                    let pending = abort.load(Ordering::Relaxed);

                    if respeculate && !pending {
                        respeculate = false;
                        exhausted = false;
                        spec_hash = None;
                        tel.predicted = TITANIUM_NO_MOVE;
                        // Stand on the node we will actually have to search
                        // next: the position *after* the opponent's likely
                        // reply. Searching the opponent-to-move node instead
                        // spreads effort over every reply, so a correct guess
                        // buys almost nothing.
                        let mut predicted = std::mem::replace(&mut hint, TITANIUM_NO_MOVE);
                        if predicted == TITANIUM_NO_MOVE {
                            let probe = search.think(PONDER_PROBE_MS, 128, false, false, label);
                            tel.nodes = tel.nodes.saturating_add(probe.nodes);
                            tel.ms = tel.ms.saturating_add(probe.ms);
                            tel.chunks += 1;
                            predicted = probe.mv;
                        }
                        if predicted != TITANIUM_NO_MOVE {
                            let mut spec = cur_g.clone();
                            spec.make_move(predicted);
                            // Never ponder past a terminal node.
                            if spec.winner() < 0 {
                                search.apply_move(predicted);
                                spec_hash = Some((spec.hash_lo, spec.hash_hi));
                                tel.predicted = predicted;
                            }
                        }
                    }

                    // One continuous search rather than a stream of slices.
                    // Slicing only ever existed so the daemon could notice a
                    // command between calls; the abort flag does that within
                    // ~63 nodes, so the slices bought nothing and cost a search
                    // restart every time. Measured before this: 1188 think()
                    // calls totalling 456ms of actual search for one turn --
                    // almost all of it restart overhead and spin.
                    if !exhausted && !pending {
                        let r = search.think(PONDER_INFINITE_MS, 128, false, false, label);
                        tel.nodes = tel.nodes.saturating_add(r.nodes);
                        tel.ms = tel.ms.saturating_add(r.ms);
                        tel.chunks += 1;
                        if r.mv != TITANIUM_NO_MOVE && spec_hash.is_some() {
                            last_mv = r.mv;
                            last_score = r.score;
                        }
                        // Returning early means the tree is solved out, not that
                        // time ran out. Stop searching and just wait: hammering
                        // think() on an exhausted position burns a core, and
                        // that core is one the opponent is entitled to.
                        // An aborted search also returns early, so returning
                        // short of budget alone does not mean the tree is
                        // solved out -- check the flag to tell the two apart.
                        // Getting this wrong would park a perfectly searchable
                        // position in the idle poll.
                        exhausted = !abort.load(Ordering::Relaxed)
                            && r.ms + PONDER_EXHAUSTED_MARGIN_MS < PONDER_INFINITE_MS;
                    } else {
                        // Either the tree is solved out, or a command is in
                        // flight and we must not start a search we cannot see
                        // the end of. Wait rather than burn a core.
                        std::thread::sleep(std::time::Duration::from_millis(
                            PONDER_IDLE_POLL_MS,
                        ));
                    }

                    match rx.try_recv() {
                        Ok(Cmd::GoTimed(time_ms)) => {
                            search.set_pondering(false);
                            abort.store(false, Ordering::Relaxed);
                            // A `go` with no intervening position update would
                            // otherwise search the speculative child.
                            if spec_hash.is_some() {
                                search.set_position(cur_g.clone());
                            }
                            let r2 = search.think(time_ms, 128, false, true, label);
                            last_score = r2.score;
                            let mv = r2.mv;
                            let _ = tx.send(Reply::BestMove(mv, Some(Box::new(r2)), tel));
                            break;
                        }
                        Ok(Cmd::StopAndGet) => {
                            search.set_pondering(false);
                            let _ = tx.send(Reply::BestMove(last_mv, None, tel));
                            break;
                        }
                        Ok(Cmd::PonderHit(time_ms)) => {
                            abort.store(false, Ordering::Relaxed);
                            // Predicted correctly — zero TT/history loss.
                            // Exit ponder mode so the real think does one normal
                            // tt_gen advance + history halving and then runs.
                            search.set_pondering(false);
                            tel.hit = tel.predicted != TITANIUM_NO_MOVE;
                            let r2 = search.think(time_ms, 128, false, true, label);
                            last_score = r2.score;
                            let mv = r2.mv;
                            let _ = tx.send(Reply::BestMove(mv, Some(Box::new(r2)), tel));
                            break;
                        }
                        Ok(Cmd::MoveMiss { new_game, time_ms }) => {
                            abort.store(false, Ordering::Relaxed);
                            search.set_pondering(false);
                            cur_g = new_game.clone();
                            search.set_position(new_game);
                            search.decay_history_by_surprise(last_score);
                            let r2 = search.think(time_ms, 128, false, true, label);
                            last_score = r2.score;
                            let mv = r2.mv;
                            let _ = tx.send(Reply::BestMove(mv, Some(Box::new(r2)), tel));
                            break;
                        }
                        Ok(Cmd::Quit) | Err(mpsc::TryRecvError::Disconnected) => return,
                        Ok(Cmd::SetGame(g)) => {
                            // Position update mid-ponder. Credit the guess if the
                            // opponent walked into the node we were standing on;
                            // the nodes already spent stay on the tally either way
                            // because they are what the next think inherits.
                            if let Some(h) = spec_hash {
                                if (g.hash_lo, g.hash_hi) == h {
                                    tel.hit = true;
                                }
                            }
                            cur_g = g.clone();
                            search.set_position(g);
                            last_mv = TITANIUM_NO_MOVE;
                            respeculate = true;
                        }
                        Ok(_) => {}
                        Err(mpsc::TryRecvError::Empty) => {
                            // Queue drained: nothing is in flight, so resuming
                            // the search cannot swallow a pending command. This
                            // is the only safe place to clear the flag, and it
                            // is safe only because the sender queues before it
                            // raises the flag.
                            abort.store(false, Ordering::Relaxed);
                        }
                    }
                }
            }
            Cmd::StopAndGet => {
                // Not pondering — nothing to return.
                let _ = tx.send(Reply::BestMove(
                    TITANIUM_NO_MOVE,
                    None,
                    PonderTelemetry::default(),
                ));
            }
            Cmd::PonderHit(time_ms) => {
                let r = search.think(time_ms, 128, false, true, label);
                last_score = r.score;
                let mv = r.mv;
                let _ = tx.send(Reply::BestMove(mv, Some(Box::new(r)), PonderTelemetry::default()));
            }
            Cmd::MoveMiss { new_game, time_ms } => {
                cur_g = new_game.clone();
                search.set_position(new_game);
                let r = search.think(time_ms, 128, false, true, label);
                last_score = r.score;
                let mv = r.mv;
                let _ = tx.send(Reply::BestMove(mv, Some(Box::new(r)), PonderTelemetry::default()));
            }
            Cmd::Quit => return,
        }
    }
}

// ── I/O loop ──────────────────────────────────────────────────────────────────

fn emit_info_json(
    stdout: &mut io::Stdout,
    engine_flag: &str,
    r: &ThinkResult,
    g: &GameState,
    ponder: &PonderTelemetry,
) {
    let helper_nodes = r
        .helper_nodes
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let helper_depths = r
        .helper_completed_depths
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let elapsed_ms = r.ms;
    let nps = r.nodes.saturating_mul(1000) / elapsed_ms.max(1);
    let root_score_text = crate::titanium::search::score_label(r.score);
    let t = &r.timing;

    // Game-state telemetry for time-management analysis.
    let walls_p0 = g.wl[0];
    let walls_p1 = g.wl[1];
    let pawn_p0 = g.pawn[0];
    let pawn_p1 = g.pawn[1];
    let turn = g.turn;
    let hist_len = g.hist_len;
    let dist_p0 = g.dist_to_goal_for_pawn(0);
    let dist_p1 = g.dist_to_goal_for_pawn(1);
    // Jump-aware distances (accounts for opponent pawn jumps).
    let mut g_for_ja = g.clone();
    let ja = crate::titanium::race::jump_aware_goal_distances(&mut g_for_ja);
    let ja_d0 = if ja.d0 == u8::MAX { -1i32 } else { ja.d0 as i32 };
    let ja_d1 = if ja.d1 == u8::MAX { -1i32 } else { ja.d1 as i32 };
    // Lower bound on remaining plies: both sides need at least dist_to_goal moves.
    let min_remaining_plies = (dist_p0.min(dist_p1) as u32).saturating_add(
        (dist_p0.max(dist_p1) as u32).saturating_sub(dist_p0.min(dist_p1) as u32),
    );
    // Wall differential: positive = side to move has more walls.
    let wall_diff = walls_p0 as i32 - walls_p1 as i32;
    let wall_diff_side = if turn == 0 { wall_diff } else { -wall_diff };

    // Pondering telemetry (cumulative nodes/time spent pondering before this think).
    let ponder_nodes = ponder.nodes;
    let ponder_chunks = ponder.chunks;
    let ponder_ms = ponder.ms;
    let ponder_predicted = if ponder.predicted == TITANIUM_NO_MOVE {
        String::new()
    } else {
        move_id_to_algebraic(ponder.predicted)
    };
    let ponder_hit = if ponder.hit { "true" } else { "false" };

    let _ = writeln!(
        stdout,
        "info json {{\"engine\":\"{}\",\"stoppedBy\":\"{}\",\"searchDepth\":{},\"nodes\":{},\"mainThreadNodes\":{},\"helperNodes\":[{}],\"totalNodes\":{},\"mainCompletedDepth\":{},\"helperCompletedDepths\":[{}],\"rootScore\":{},\"rootScoreText\":\"{}\",\"whiteDist\":{},\"blackDist\":{},\"elapsedMs\":{},\"nps\":{},\"allocatedHardMs\":{},\"allocatedSoftMs\":{},\"searchableMs\":{},\"gateReserveMs\":{},\"hardOvershootMs\":{},\"softOvershootMs\":{},\"lastIterMs\":{},\"prevIterMs\":{},\"bestMoveChanges\":{},\"partialIterUsed\":{},\"softFractionBp\":{},\"ponderNodes\":{},\"ponderChunks\":{},\"ponderMs\":{},\"ponderPredicted\":\"{}\",\"ponderHit\":{},\"wallsP0\":{},\"wallsP1\":{},\"pawnP0\":{},\"pawnP1\":{},\"turn\":{},\"histLen\":{},\"distP0\":{},\"distP1\":{},\"jaDistP0\":{},\"jaDistP1\":{},\"minRemainingPlies\":{},\"wallDiffSide\":{}}}",
        engine_flag,
        r.stop_reason,
        r.depth,
        r.nodes,
        r.main_thread_nodes,
        helper_nodes,
        r.total_nodes,
        r.main_completed_depth,
        helper_depths,
        r.score,
        root_score_text,
        r.white_dist,
        r.black_dist,
        elapsed_ms,
        nps,
        t.allocated_hard_ms,
        t.allocated_soft_ms,
        t.searchable_ms,
        t.gate_reserve_ms,
        t.hard_overshoot_ms,
        t.soft_overshoot_ms,
        t.last_iter_ms,
        t.prev_iter_ms,
        t.best_move_changes,
        if t.partial_iter_used { "true" } else { "false" },
        t.soft_fraction_bp,
        ponder_nodes,
        ponder_chunks,
        ponder_ms,
        ponder_predicted,
        ponder_hit,
        walls_p0,
        walls_p1,
        pawn_p0,
        pawn_p1,
        turn,
        hist_len,
        dist_p0,
        dist_p1,
        ja_d0,
        ja_d1,
        min_remaining_plies,
        wall_diff_side,
    );
    let _ = stdout.flush();
}

pub fn run_v15_session_stdio(engine_flag: &str) {
    let (cmd_tx, cmd_rx) = mpsc::channel::<Cmd>();
    let (reply_tx, reply_rx) = mpsc::channel::<Reply>();
    // Raised by this thread before every command so a ponder search stops at
    // once instead of running out its slice. Without it the daemon only noticed
    // a `go` between slices, and the harness charges that delay to our clock --
    // it cost two games on time in completely won positions.
    let abort = Arc::new(AtomicBool::new(false));
    let abort_io = Arc::clone(&abort);

    let flag_owned = engine_flag.to_string();
    thread::spawn(move || search_daemon(flag_owned, cmd_rx, reply_tx, abort));

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    let mut applied: Vec<String> = Vec::new();
    // Track game state in I/O thread for position management.
    let mut current_g = GameState::new();
    // Move the engine was asked to ponder on (TITANIUM_NO_MOVE if none).
    let mut ponder_mv: i16 = TITANIUM_NO_MOVE;
    let auto_ponder = std::env::var("TITANIUM_PONDERING")
        .map(|value| value != "0")
        .unwrap_or(true);

    macro_rules! ok {
        ($msg:expr) => {{
            let _ = writeln!(stdout, "{}", $msg);
            let _ = stdout.flush();
        }};
    }
    macro_rules! err {
        ($msg:expr) => {{
            let _ = writeln!(stdout, "error {}", $msg);
            let _ = stdout.flush();
        }};
    }

    fn replay_moves(moves: &[String]) -> Result<GameState, String> {
        let mut g = GameState::new();
        for text in moves {
            if g.winner() >= 0 {
                return Err(format!("move {text} past terminal position"));
            }
            g.make_move(algebraic_to_move_id(text));
        }
        Ok(g)
    }

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                err!(e);
                break;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.splitn(4, ' ').collect();

        match parts[0] {
            "reset" => {
                current_g = GameState::new();
                applied.clear();
                ponder_mv = TITANIUM_NO_MOVE;
                let _ = { let r = cmd_tx.send(Cmd::SetGame(GameState::new())); abort_io.store(true, Ordering::Relaxed); r };
                ok!("ready");
            }
            "position" => {
                let moves: Vec<String> = if parts.len() > 1 {
                    parts[1..]
                        .join(" ")
                        .split_whitespace()
                        .map(String::from)
                        .collect()
                } else {
                    Vec::new()
                };
                let extends = !applied.is_empty()
                    && moves.len() >= applied.len()
                    && moves.iter().zip(applied.iter()).all(|(a, b)| a == b);
                if extends {
                    let mut err = None;
                    for text in &moves[applied.len()..] {
                        if current_g.winner() >= 0 {
                            err = Some(format!("move {text} past terminal position"));
                            break;
                        }
                        current_g.make_move(algebraic_to_move_id(text));
                    }
                    if let Some(msg) = err {
                        err!(msg);
                        continue;
                    }
                    // Incremental update: send only the new game state.
                    let _ = { let r = cmd_tx.send(Cmd::SetGame(current_g.clone())); abort_io.store(true, Ordering::Relaxed); r };
                } else {
                    match replay_moves(&moves) {
                        Ok(g) => {
                            current_g = g.clone();
                            let _ = { let r = cmd_tx.send(Cmd::SetGame(g)); abort_io.store(true, Ordering::Relaxed); r };
                        }
                        Err(msg) => {
                            err!(msg);
                            continue;
                        }
                    }
                }
                applied = moves;
                ponder_mv = TITANIUM_NO_MOVE;
                let _ = writeln!(stdout, "ready {}", applied.len());
                let _ = stdout.flush();
            }
            "makemove" => {
                let Some(mv_str) = parts.get(1) else {
                    err!("makemove requires a move");
                    continue;
                };
                if current_g.winner() >= 0 {
                    err!("terminal position");
                    continue;
                }
                let mv = algebraic_to_move_id(mv_str);
                current_g.make_move(mv);
                applied.push((*mv_str).to_string());
                ponder_mv = TITANIUM_NO_MOVE;
                let _ = { let r = cmd_tx.send(Cmd::SetGame(current_g.clone())); abort_io.store(true, Ordering::Relaxed); r };
                ok!("ready");
            }
            "go" => {
                if current_g.winner() >= 0 {
                    err!("terminal position");
                    continue;
                }
                let arg1 = parts.get(1).copied().unwrap_or("4.0");
                if arg1 == "infinite" {
                    // go infinite [PONDER_MOVE]
                    let pm_str = parts.get(2).copied().unwrap_or("");
                    ponder_mv = if pm_str.is_empty() {
                        TITANIUM_NO_MOVE
                    } else {
                        algebraic_to_move_id(pm_str)
                    };
                    let _ = { let r = cmd_tx.send(Cmd::GoInfinite(ponder_mv)); abort_io.store(true, Ordering::Relaxed); r };
                    // No reply expected — daemon starts pondering.
                } else {
                    let time_ms = if arg1 == "rem" {
                        let rem_sec: f64 = parts
                            .get(2)
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(4.0);
                        let remaining_ms = (rem_sec * 1000.0).max(0.0) as u64;
                        let ja = crate::titanium::race::jump_aware_goal_distances(&mut current_g);
                        let d0 = (ja.d0 != u8::MAX).then_some(u32::from(ja.d0));
                        let d1 = (ja.d1 != u8::MAX).then_some(u32::from(ja.d1));
                        crate::titanium::time_alloc::allocate_move_budget_with_dists(
                            remaining_ms,
                            current_g.hist_len,
                            current_g.turn,
                            current_g.pawn,
                            d0,
                            d1,
                        )
                        .move_ms
                        .max(1)
                    } else {
                        let time_sec: f64 = arg1.parse().unwrap_or(4.0);
                        (time_sec * 1000.0).max(1.0) as u64
                    };
                    let _ = { let r = cmd_tx.send(Cmd::GoTimed(time_ms)); abort_io.store(true, Ordering::Relaxed); r };
                    match reply_rx.recv() {
                        Ok(Reply::BestMove(mv, info, ponder)) => {
                            if let Some(r) = info.as_deref() {
                                emit_info_json(&mut stdout, engine_flag, r, &current_g, &ponder);
                            }
                            if mv == TITANIUM_NO_MOVE {
                                ok!("bestmove (none)");
                            } else {
                                let mv_text = move_id_to_algebraic(mv);
                                ok!(format!("bestmove {mv_text}"));
                                if auto_ponder && current_g.winner() < 0 {
                                    current_g.make_move(mv);
                                    applied.push(mv_text);
                                    let _ = { let r = cmd_tx.send(Cmd::SetGame(current_g.clone())); abort_io.store(true, Ordering::Relaxed); r };
                                    let _ = { let r = cmd_tx.send(Cmd::GoInfinite(TITANIUM_NO_MOVE)); abort_io.store(true, Ordering::Relaxed); r };
                                }
                            }
                        }
                        Ok(Reply::Error(msg)) => err!(msg),
                        Err(_) => break,
                    }
                }
            }
            "stop" => {
                let _ = { let r = cmd_tx.send(Cmd::StopAndGet); abort_io.store(true, Ordering::Relaxed); r };
                match reply_rx.recv() {
                    Ok(Reply::BestMove(mv, info, ponder)) => {
                        if let Some(r) = info.as_deref() {
                            emit_info_json(&mut stdout, engine_flag, r, &current_g, &ponder);
                        }
                        if mv == TITANIUM_NO_MOVE {
                            ok!("bestmove (none)");
                        } else {
                            ok!(format!("bestmove {}", move_id_to_algebraic(mv)));
                        }
                    }
                    Ok(Reply::Error(msg)) => err!(msg),
                    Err(_) => break,
                }
            }
            "ponderhit" => {
                // ponderhit TIME_MS  — ponder move was correct
                let time_ms: u64 = parts
                    .get(1)
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(|s| (s * 1000.0).max(1.0) as u64)
                    .unwrap_or(5000);
                // Update I/O position: the ponder move was played.
                if ponder_mv != TITANIUM_NO_MOVE {
                    if current_g.winner() < 0 {
                        current_g.make_move(ponder_mv);
                        applied.push(move_id_to_algebraic(ponder_mv));
                    }
                    ponder_mv = TITANIUM_NO_MOVE;
                }
                let _ = { let r = cmd_tx.send(Cmd::PonderHit(time_ms)); abort_io.store(true, Ordering::Relaxed); r };
                match reply_rx.recv() {
                    Ok(Reply::BestMove(mv, info, ponder)) => {
                        if let Some(r) = info.as_deref() {
                            emit_info_json(&mut stdout, engine_flag, r, &current_g, &ponder);
                        }
                        if mv == TITANIUM_NO_MOVE {
                            ok!("bestmove (none)");
                        } else {
                            ok!(format!("bestmove {}", move_id_to_algebraic(mv)));
                        }
                    }
                    Ok(Reply::Error(msg)) => err!(msg),
                    Err(_) => break,
                }
            }
            "movemiss" => {
                // movemiss MOVE TIME_MS  — opponent played MOVE, not the ponder move
                let Some(mv_str) = parts.get(1) else {
                    err!("movemiss requires MOVE TIME_MS");
                    continue;
                };
                let time_ms: u64 = parts
                    .get(2)
                    .and_then(|s| s.parse::<f64>().ok())
                    .map(|s| (s * 1000.0).max(1.0) as u64)
                    .unwrap_or(5000);
                // Rewind to pre-ponder position, then apply actual move.
                // We do this by replaying applied (which doesn't include ponder_mv).
                let actual_mv = algebraic_to_move_id(mv_str);
                if current_g.winner() < 0 {
                    current_g.make_move(actual_mv);
                    applied.push((*mv_str).to_string());
                }
                ponder_mv = TITANIUM_NO_MOVE;
                let new_game = current_g.clone();
                let _ = { let r = cmd_tx.send(Cmd::MoveMiss { new_game, time_ms }); abort_io.store(true, Ordering::Relaxed); r };
                match reply_rx.recv() {
                    Ok(Reply::BestMove(mv, info, ponder)) => {
                        if let Some(r) = info.as_deref() {
                            emit_info_json(&mut stdout, engine_flag, r, &current_g, &ponder);
                        }
                        if mv == TITANIUM_NO_MOVE {
                            ok!("bestmove (none)");
                        } else {
                            ok!(format!("bestmove {}", move_id_to_algebraic(mv)));
                        }
                    }
                    Ok(Reply::Error(msg)) => err!(msg),
                    Err(_) => break,
                }
            }
            "quit" => {
                let _ = { let r = cmd_tx.send(Cmd::Quit); abort_io.store(true, Ordering::Relaxed); r };
                break;
            }
            _ => err!("unknown command"),
        }
    }
}
