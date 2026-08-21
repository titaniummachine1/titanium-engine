//! Titanium Engine CLI — perft / divide / bench / genmove entry points.

use std::env;
use std::time::{Duration, Instant};

use titanium::{
    both_players_reach_goals, cat_snapshot_json, format_move, generate_legal_moves, perft_divide,
    perft_fast_anchor_baseline, perft_no_tt_anchor_baseline, Board, Engine, PERFT5_TIMEOUT_SECS,
};

#[cfg(not(target_arch = "wasm32"))]
fn maybe_pin_core() {
    use core_affinity::CoreId;

    let core = if let Ok(s) = env::var("TITANIUM_PIN_CORE") {
        s.parse::<usize>().ok().map(|id| CoreId { id })
    } else if env::var("TITANIUM_PIN_LAST").is_ok() {
        core_affinity::get_core_ids().and_then(|ids| ids.last().copied())
    } else {
        None
    };
    if let Some(c) = core {
        if core_affinity::set_for_current(c) {
            eprintln!("pinned: logical core {}", c.id);
        } else {
            eprintln!("warning: could not pin to core {}", c.id);
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn maybe_pin_core() {}

fn main() {
    maybe_pin_core();
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        print_usage();
        return;
    }

    // Cold-start pawn tables (~1-2s, once per process). Long-lived server modes
    // kick the build off in the background AT LAUNCH so it overlaps the GUI
    // handshake (`isready`/first move blocks on it only if it isn't done yet);
    // one-shot timed commands build synchronously up front so the build is never
    // inside a measured region. Never rebuilds mid-session — that's the OnceLock.
    match args[1].as_str() {
        "uci" | "session" => {
            std::thread::spawn(|| titanium::movegen::prewarm());
        }
        _ => titanium::movegen::prewarm(),
    }

    match args[1].as_str() {
        "perft" => run_perft(&args),
        "perft-bench" => run_perft_bench(&args),
        "divide" => run_divide(&args),
        "bench" => run_bench(&args),
        "perft-race" => run_perft_race(&args),
        "perft-id" => run_perft_id(&args),
        "thread-bench" => run_thread_bench(&args),
        "moves" => run_moves(&args),
        "cat" => run_cat(&args),
        "eval" => run_eval(&args),
        "eval-batch" => run_eval_batch(),
        "eval-packed-batch" => run_eval_packed_batch(),
        "path-scan" => run_path_scan(),
        "cat-packed-batch" => run_cat_packed_batch(),
        "score-out" => run_score_out(&args),
        "tbgen" => run_tbgen(&args),
        "tbdump" => run_tbdump(&args),
        "tblayers" => run_tblayers(&args),
        "tbsolve" => run_tbsolve(&args),
        "tbmerge" => run_tbmerge(&args),
        "fields" => run_fields(&args),
        // One engine, reached without ceremony. There used to be a flag, and an
        // unrecognised value -- including none at all, or a version newer than
        // the hardcoded list knew -- fell through to the legacy search
        // *silently*. That is how a v19 engine answered only to
        // "titanium-v17", with anything else playing at roughly a sixth of the
        // node rate, six plies shallower, and no warning.
        "uci" | "session" => {
            // Only genuine ACE engines are rejected. `ace_engine_flag` also
            // returns titanium-* names, so testing it directly rejected the very
            // flag every existing caller passes.
            if let Some(flag) = ace_engine_flag(&args).filter(|f| f.starts_with("ace-")) {
                eprintln!(
                    "error: ACE engines ({flag}) live in the `ace` binary under engines/ace/"
                );
                std::process::exit(2);
            }
            titanium::run_titanium_session_stdio(parse_threads_arg(&args));
        }
        _ => print_usage(),
    }
}

fn print_usage() {
    println!("Titanium Engine 0.1.0");
    println!("  titanium perft [depth] [--threads N]  — node count (default depth 3, threads 1; d5 caps at 20s)");
    println!("  titanium perft-bench [depth]          — prewarm, readyok, then timed perft only");
    println!("  titanium divide <depth>                — perft with move breakdown");
    println!("  titanium bench <depth> <n> [--threads N]");
    println!("  titanium thread-bench [depth] [--threads N] — 1 vs N threads, same nodes");
    println!("  titanium perft-race <sec>              — max depth within time budget");
    println!("  titanium perft-id [depth]              — iterative deepening perft 0..depth");
    println!("  titanium moves [moves...]              — list legal moves after the given line");
    println!("  titanium genmove [moves...] [--engine mcts|minimax|greedy] [--cat]");
    println!("              [--time SEC] [--sims N] [--uct F] [--nodes N] [--log]");
    println!("              — default: Gorisanson-style MCTS in Rust");
    println!("  titanium uci                           — UCI-style stdio protocol (testing infra)");
    println!("  titanium cat [moves...]                — CAT v3 heatmap JSON for current position");
    println!(
        "  titanium lmr [moves...] [--time SEC] [--depth N] — root LMR plan JSON (default depth 8)"
    );
    println!(
        "  titanium session [--engine ace-v13-ti|titanium-v17] — long-lived REPL (TT persists between plies)"
    );
    println!(
        "  titanium genmove --engine ace-v13 [moves...] — gen13 / Titanium search (O1 movegen; ace-v13-pure = faithful 1:1)"
    );
    println!("  True ACE engines (ace, ace-v8, …) live in engines/ace — use the `ace` binary.");
    println!("  titanium eval [moves...] [--json]     — HalfPW net eval dump (trainer parity)");
    println!(
        "  titanium eval-packed-batch            — stdin: u32 row + 24-byte packed state records"
    );
    println!(
        "  titanium path-scan                    — stdin lines: game_id move1 move2 ...; Titanium legal+path check"
    );
    println!(
        "  titanium cat-packed-batch             — stdin: u32 row + 24-byte packed state records"
    );
    println!(
        "  titanium score-out --nodes N (--packed HEX | --moves MOVE...) — bounded AB score JSON"
    );
    println!(
        "  titanium fields [moves...] [--check]  — ASCII distance/corridor field grids + invariants"
    );
}

const DEFAULT_PERFT_DEPTH: u32 = 3;
const DEFAULT_THREAD_BENCH_DEPTH: u32 = 4;

struct CliArgs {
    positional: Vec<String>,
    threads: usize,
    no_tt: bool,
}

fn parse_cli(args: &[String]) -> CliArgs {
    let mut threads = 1usize;
    let mut no_tt = false;
    let mut positional = Vec::new();
    let mut i = 0usize;
    while i < args.len() {
        if args[i] == "--threads" {
            threads = args
                .get(i + 1)
                .and_then(|s| s.parse().ok())
                .unwrap_or(1)
                .max(1);
            i += 2;
            continue;
        }
        if args[i] == "--no-tt" {
            no_tt = true;
            i += 1;
            continue;
        }
        positional.push(args[i].clone());
        i += 1;
    }
    CliArgs {
        positional,
        threads,
        no_tt,
    }
}

fn default_parallel_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .max(2)
}

fn load_board(cli: &CliArgs, depth_index: usize) -> (Board, u32) {
    let depth: u32 = cli
        .positional
        .get(depth_index)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PERFT_DEPTH);
    let mut board = Board::new();
    for mv in cli.positional.iter().skip(depth_index + 1) {
        board.apply_algebraic(mv);
    }
    (board, depth)
}

fn make_engine(threads: usize) -> Engine {
    if threads <= 1 {
        Engine::new()
    } else {
        Engine::with_threads(threads)
    }
}

fn run_perft(args: &[String]) {
    let cli = parse_cli(args);
    let (board, depth) = load_board(&cli, 2);
    if depth == 5 {
        match run_perft_depth5_timed(&board, cli.threads, cli.no_tt, false) {
            Ok((nodes, elapsed)) => {
                println!("perft {} {}", depth, nodes);
                println!("threads {}", cli.threads);
                println!("time {:.3}s", elapsed.as_secs_f64());
            }
            Err(()) => fail_perft5_timeout(),
        }
        return;
    }

    let mut engine = make_engine(cli.threads);
    let start = Instant::now();
    let nodes = if cli.no_tt {
        let mut board_copy = board.clone();
        engine.perft_no_tt(&mut board_copy, depth)
    } else {
        engine.perft(&board, depth)
    };
    let elapsed = start.elapsed();
    println!("perft {} {}", depth, nodes);
    println!("threads {}", cli.threads);
    println!("time {:.3}s", elapsed.as_secs_f64());
}

fn perft5_timeout() -> Duration {
    #[cfg(feature = "bench-instrument")]
    return Duration::from_secs(120);
    #[cfg(not(feature = "bench-instrument"))]
    Duration::from_secs(PERFT5_TIMEOUT_SECS)
}

fn fail_perft5_timeout() -> ! {
    eprintln!(
        "perft(5) TIMEOUT after {}s — aborting (not worth continuing)",
        PERFT5_TIMEOUT_SECS
    );
    std::process::exit(1);
}

#[cfg(not(target_arch = "wasm32"))]
fn run_perft_depth5_timed(
    board: &Board,
    threads: usize,
    no_tt: bool,
    anchor_baseline: bool,
) -> Result<(u64, Duration), ()> {
    use std::sync::mpsc;

    let (done_tx, done_rx) = mpsc::channel();
    let board_copy = board.clone();
    let handle = std::thread::Builder::new()
        .name("perft-d5".into())
        .spawn(move || {
            let mut engine = make_engine(threads);
            titanium::bench_instr::begin_search();
            let start = Instant::now();
            let nodes = if anchor_baseline && no_tt {
                let mut scratch = board_copy.clone();
                perft_no_tt_anchor_baseline(&mut scratch, 5)
            } else if anchor_baseline {
                let mut scratch = board_copy.clone();
                perft_fast_anchor_baseline(&mut scratch, 5)
            } else if no_tt {
                let mut scratch = board_copy.clone();
                engine.perft_no_tt(&mut scratch, 5)
            } else {
                engine.perft(&board_copy, 5)
            };
            let elapsed = start.elapsed();
            titanium::bench_instr::end_search(nodes);
            if let Some(report) = titanium::bench_instr::take_json_report() {
                eprintln!("perft_profile {report}");
            }
            let _ = done_tx.send((nodes, elapsed));
        })
        .expect("spawn perft-d5");

    match done_rx.recv_timeout(perft5_timeout()) {
        Ok(result) => {
            handle.join().ok();
            Ok(result)
        }
        Err(_) => {
            std::mem::forget(handle);
            Err(())
        }
    }
}

#[cfg(target_arch = "wasm32")]
fn run_perft_depth5_timed(
    board: &Board,
    threads: usize,
    no_tt: bool,
    anchor_baseline: bool,
) -> Result<(u64, Duration), ()> {
    let mut engine = make_engine(threads);
    let start = Instant::now();
    let nodes = if anchor_baseline && no_tt {
        let mut board_copy = board.clone();
        perft_no_tt_anchor_baseline(&mut board_copy, 5)
    } else if anchor_baseline {
        let mut board_copy = board.clone();
        perft_fast_anchor_baseline(&mut board_copy, 5)
    } else if no_tt {
        let mut board_copy = board.clone();
        engine.perft_no_tt(&mut board_copy, 5)
    } else {
        engine.perft(board, 5)
    };
    let elapsed = start.elapsed();
    if elapsed > perft5_timeout() {
        return Err(());
    }
    Ok((nodes, elapsed))
}

fn perft_bench_uses_anchor() -> bool {
    if env::var("TITANIUM_BENCH").ok().as_deref() != Some("1") {
        return false;
    }
    matches!(
        env::var("TITANIUM_WALL_FLOOD_SKIP")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "anchor" | "old" | "0"
    )
}

/// Timed perft for benchmarks: full prewarm, emit `readyok`, then measure perft only.
fn run_perft_bench(args: &[String]) {
    use std::io::Write as _;

    let cli = parse_cli(args);
    let (board, depth) = load_board(&cli, 2);
    titanium::movegen::prewarm();
    let flood_anchor = perft_bench_uses_anchor();
    println!("readyok");
    let _ = std::io::stdout().flush();

    let flood_skip = if flood_anchor { "anchor" } else { "topo" };

    if depth == 5 {
        match run_perft_depth5_timed(&board, cli.threads, cli.no_tt, flood_anchor) {
            Ok((nodes, elapsed)) => {
                let nps = nodes as f64 / elapsed.as_secs_f64();
                println!(
                    "perft_bench depth={depth} nodes={nodes} threads={} wall_flood_skip={flood_skip} time_s={:.6} nps={nps:.0}",
                    cli.threads,
                    elapsed.as_secs_f64(),
                );
            }
            Err(()) => fail_perft5_timeout(),
        }
        return;
    }

    let start = Instant::now();
    let nodes = if flood_anchor && cli.no_tt {
        let mut board_copy = board.clone();
        perft_no_tt_anchor_baseline(&mut board_copy, depth)
    } else if flood_anchor {
        let mut board_copy = board.clone();
        perft_fast_anchor_baseline(&mut board_copy, depth)
    } else if cli.no_tt {
        let mut engine = make_engine(cli.threads);
        let mut board_copy = board.clone();
        engine.perft_no_tt(&mut board_copy, depth)
    } else {
        let mut engine = make_engine(cli.threads);
        engine.perft(&board, depth)
    };
    let elapsed = start.elapsed();
    let nps = nodes as f64 / elapsed.as_secs_f64();
    println!(
        "perft_bench depth={depth} nodes={nodes} threads={} wall_flood_skip={flood_skip} time_s={:.6} nps={nps:.0}",
        cli.threads,
        elapsed.as_secs_f64(),
    );
}

fn run_divide(args: &[String]) {
    let cli = parse_cli(args);
    let (board, depth) = load_board(&cli, 2);
    let (total, lines) = perft_divide(&board, depth);
    for (mv, nodes) in &lines {
        println!("{} {}", mv, nodes);
    }
    println!();
    println!("Nodes searched: {}", total);
}

fn run_bench(args: &[String]) {
    let cli = parse_cli(args);
    let depth: u32 = cli
        .positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PERFT_DEPTH);
    let iterations: u32 = cli
        .positional
        .get(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let board = Board::new();
    let mut engine = make_engine(cli.threads);

    engine.perft(&board, depth);

    let start = Instant::now();
    let mut nodes = 0u64;
    for _ in 0..iterations {
        nodes = engine.perft(&board, depth);
    }
    let elapsed = start.elapsed();
    let total_nodes = nodes * iterations as u64;
    let nps = total_nodes as f64 / elapsed.as_secs_f64();

    println!(
        "bench depth={} iters={} threads={} nodes={} time={:.3}s nps={:.0}",
        depth,
        iterations,
        cli.threads,
        total_nodes,
        elapsed.as_secs_f64(),
        nps
    );
}

fn run_perft_id(args: &[String]) {
    let cli = parse_cli(args);
    let max_depth: u32 = cli
        .positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PERFT_DEPTH);
    let mut board = Board::new();
    let mut engine = make_engine(cli.threads);
    let start = Instant::now();
    let lines = engine.perft_iterative(&mut board, max_depth);
    let elapsed = start.elapsed();

    for (depth, nodes) in &lines {
        println!("perft {} {}", depth, nodes);
    }
    println!("threads {}", cli.threads);
    println!("perft-id total {:.3}s", elapsed.as_secs_f64());
}

fn run_thread_bench(args: &[String]) {
    let cli = parse_cli(args);
    let depth: u32 = cli
        .positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_THREAD_BENCH_DEPTH);
    let parallel = if cli.threads > 1 {
        cli.threads
    } else {
        default_parallel_threads()
    };
    let board = Board::new();

    let result = Engine::bench_threads(&board, depth, parallel);

    println!("thread-bench depth={} nodes={}", result.depth, result.nodes);
    println!("threads=1  time {:.3}s", result.threads_one_secs);
    println!(
        "threads={} time {:.3}s",
        result.threads_n, result.threads_n_secs
    );
    println!("speedup {:.2}x", result.speedup());
}

fn run_perft_race(args: &[String]) {
    let cli = parse_cli(args);
    let budget: f64 = cli
        .positional
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(3.0);
    let board = Board::new();
    let mut engine = make_engine(cli.threads);
    let mut best_depth = 0u32;
    let mut best_nodes = 0u64;
    let mut best_ms = 0.0f64;

    for depth in 1..=8 {
        let start = Instant::now();
        let nodes = engine.perft(&board, depth);
        let ms = start.elapsed().as_secs_f64() * 1000.0;
        if ms > budget * 1000.0 {
            break;
        }
        best_depth = depth;
        best_nodes = nodes;
        best_ms = ms;
    }

    println!(
        "perft-race budget={:.1}s threads={} best_depth={} nodes={} time_ms={:.0}",
        budget, cli.threads, best_depth, best_nodes, best_ms
    );
}

fn run_moves(args: &[String]) {
    let mut board = Board::new();
    let mut plies = 0usize;
    for text in args.iter().skip(2) {
        if text.starts_with("--") {
            break;
        }
        // Fail closed: applying an illegal move would leave the caller listing
        // moves for a position it did not ask about, with no way to tell.
        // Silently ignoring the arguments entirely is what this used to do.
        let legal = generate_legal_moves(&board);
        if !legal.iter().any(|m| titanium::format_move(*m) == *text) {
            eprintln!("illegal move '{text}' at ply {plies}");
            std::process::exit(2);
        }
        board.apply_algebraic(text);
        plies += 1;
    }
    let moves = generate_legal_moves(&board);
    if plies == 0 {
        println!("{} legal moves at startpos", moves.len());
    } else {
        println!("{} legal moves after {} plies", moves.len(), plies);
    }
    for mv in moves {
        println!("{}", titanium::format_move(mv));
    }
}

fn run_cat(args: &[String]) {
    let mut board = Board::new();
    for mv in args.iter().skip(2) {
        if mv.starts_with("--") {
            break;
        }
        board.apply_algebraic(mv);
    }
    println!("{}", cat_snapshot_json(&mut board));
}

/// Dump the Titanium v15 (grafted) net evaluation for a position — used to verify the
/// Python NNUE trainer's forward pass matches the engine bit-for-bit. On mid-game
/// positions (both sides hold walls, not near mate) this is the pure net output.
fn run_eval(args: &[String]) {
    use titanium::{algebraic_to_move_id, GameState, TitaniumSearch};
    let mut g = GameState::new();
    for a in args.iter().skip(2) {
        if a.starts_with("--") {
            break;
        }
        g.make_move(algebraic_to_move_id(a));
    }
    // No-raceproof: the certificate floor fires on cert-eligible races (d_me<=d_opp+1)
    // and overrides the net score, which the Python HalfPW trainer does NOT model.
    // This command's purpose is the PURE NET eval (see doc above), so disable cert
    // to keep py↔engine parity exact for training.
    let mut s = TitaniumSearch::production(g, None);
    s.set_race_proof(false);
    if args.iter().any(|a| a == "--parity-trace") {
        println!("{}", s.eval_parity_trace_json());
    } else if args.iter().any(|a| a == "--json") {
        println!("{}", s.eval_dump_json());
    } else {
        println!("eval {}", s.eval_position());
    }
}

/// ASCII grids for NNUE distance / corridor fields — eyeball training geometry.
fn run_fields(args: &[String]) {
    use titanium::fields_viz::{compute_nnue_fields, render_fields_text, validate_fields};
    use titanium::{algebraic_to_move_id, GameState};
    let check = args.iter().any(|a| a == "--check");
    let mut g = GameState::new();
    for a in args.iter().skip(2) {
        if a.starts_with("--") {
            continue;
        }
        g.make_move(algebraic_to_move_id(a));
    }
    let fields = compute_nnue_fields(&g);
    if check {
        let errs = validate_fields(&g, &fields);
        if errs.is_empty() {
            eprintln!("fields: all invariants OK");
        } else {
            for e in &errs {
                eprintln!("fields ERROR: {e}");
            }
            std::process::exit(1);
        }
    }
    print!("{}", render_fields_text(&g, &fields));
}

/// Batch eval — reads one move-sequence per stdin line, emits one JSON per stdout line.
/// Dramatically faster than launching `titanium eval --json` per position (single startup).
/// Input:  `e2 e8 e3 e7 d3h f5v`  (space-separated algebraic moves, empty line = startpos)
/// Output: one compact JSON record per line (same format as `eval --json`)
fn run_eval_batch() {
    use std::io::{self, BufRead};
    use titanium::{algebraic_to_move_id, GameState, TitaniumSearch};
    let stdin = io::stdin();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let line = line.trim();
        if line.starts_with('#') {
            continue; // skip comment lines
        }
        let mut g = GameState::new();
        for tok in line.split_whitespace() {
            g.make_move(algebraic_to_move_id(tok));
        }
        let mut s = TitaniumSearch::production(g, None);
        println!("{}", s.eval_dump_json());
    }
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Batch eval from canonical packed states — stdin records: u32 LE row index + 24-byte packed state.
fn read_packed_batch(command: &str) -> Vec<(u32, [u8; titanium::PACKED_STATE_LEN])> {
    use std::io::Read;
    use titanium::PACKED_STATE_LEN;

    let mut payload = Vec::new();
    if let Err(err) = std::io::stdin().lock().read_to_end(&mut payload) {
        eprintln!("{command} read error: {err}");
        std::process::exit(1);
    }
    const ROW_BYTES: usize = 4;
    let record_len = ROW_BYTES + PACKED_STATE_LEN;
    if payload.len() % record_len != 0 {
        eprintln!("{command} read error: partial packed record");
        std::process::exit(1);
    }
    payload
        .chunks_exact(record_len)
        .map(|record| {
            let row = u32::from_le_bytes(record[..ROW_BYTES].try_into().unwrap());
            let packed = record[ROW_BYTES..].try_into().unwrap();
            (row, packed)
        })
        .collect()
}

/// Replay algebraic move lists; fail if any move is illegal for Titanium or
/// either pawn loses its goal path. Stdin: `game_id move1 move2 ...` per line.
/// Stdout: `OK id plies=N` or `FAIL id ply=K reason=... move=...`
fn run_path_scan() {
    use std::io::{self, BufRead, Write};

    let stdin = io::stdin();
    let mut stdout = io::stdout();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                eprintln!("path-scan read error: {e}");
                std::process::exit(1);
            }
        };
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let mut parts = line.split_whitespace();
        let Some(id) = parts.next() else { continue };
        let mut board = Board::new();
        let mut ply = 0usize;
        let mut done = false;
        for mv_str in parts {
            let legal = generate_legal_moves(&board);
            let Some(mv) = legal.into_iter().find(|m| format_move(*m) == mv_str) else {
                // Game already ended: trailing PGN noise after a goal — not a path violation.
                if board.is_terminal().is_some() {
                    let _ = writeln!(
                        stdout,
                        "OK {id} plies={ply} trailing_after_terminal={mv_str}"
                    );
                    done = true;
                    break;
                }
                let _ = writeln!(
                    stdout,
                    "FAIL {id} ply={ply} reason=illegal_move move={mv_str}"
                );
                done = true;
                break;
            };
            let _ = board.make_move(mv);
            ply += 1;
            // Legal walls already require paths; still assert for pawn-only corruption.
            if !both_players_reach_goals(&board) {
                let _ = writeln!(stdout, "FAIL {id} ply={ply} reason=no_path move={mv_str}");
                done = true;
                break;
            }
        }
        if !done {
            let _ = writeln!(stdout, "OK {id} plies={ply}");
        }
    }
}

fn run_eval_packed_batch() {
    use rayon::prelude::*;
    use titanium::{titanium_game_from_packed, TitaniumSearch};

    // Database training sends bounded chunks (normally 4,096 positions). The
    // positions are independent, so preserve input order while fanning their
    // feature extraction across Rayon’s local CPU worker pool.
    let rows = read_packed_batch("eval-packed-batch");
    let lines: Vec<String> = rows
        .par_iter()
        .map(|(row, packed)| match titanium_game_from_packed(packed) {
            Ok(g) => {
                let mut s = TitaniumSearch::production(g, None);
                s.eval_dump_json_packed(*row)
            }
            Err(err) => format!(
                "{{\"row\":{row},\"ok\":false,\"protocol\":\"eval-packed-v1\",\"error\":\"{}\"}}",
                json_escape(&err)
            ),
        })
        .collect();
    for line in lines {
        println!("{line}");
    }
}

/// CAT-only batch extraction from canonical packed states. This avoids the
/// full net trace, scalar evaluation, distance fields, and large eval JSON when
/// the database already stores every non-CAT feature.
fn run_cat_packed_batch() {
    use rayon::prelude::*;
    use titanium::{titanium_game_from_packed, TitaniumSearch};
    let rows = read_packed_batch("cat-packed-batch");
    let lines: Vec<String> = rows
        .par_iter()
        .map(|(row, packed)| match titanium_game_from_packed(packed) {
            Ok(g) => {
                let s = TitaniumSearch::production(g, None);
                s.cat_dump_json_packed(*row)
            }
            Err(err) => format!(
                "{{\"row\":{row},\"ok\":false,\"protocol\":\"cat-packed-v1\",\"error\":\"{}\"}}",
                json_escape(&err)
            ),
        })
        .collect();
    for line in lines {
        println!("{line}");
    }
}

/// Emit one bounded alpha-beta root score as machine-readable JSON.
///
/// Protocol `score-out-v1`:
/// `titanium score-out --nodes N --packed HEX`
/// or
/// `titanium score-out --nodes N --moves e2 e8 ...`
///
/// `packed` is the canonical 24-byte position format (48 lowercase/uppercase
/// hex characters), not the engine-internal ACE format. `nodes` is a hard
/// search-node budget; the reported `nodes` is the exact number consumed.
/// `bound` describes the last completed iterative-deepening root score:
/// `exact` means at least one complete AB iteration was committed, while
/// `unknown` means the budget stopped the search before the first iteration.
/// `proven` is true only for a mate score verified by the AB search. A finite
/// depth-limited score is not game-theoretically proven.
/// Generate (or verify) the zero-wall tablebase.
///
/// The table is small and solves in a few milliseconds, so the engine can
/// always build it at startup -- this command exists so it can instead be
/// produced once and shipped as a file, for builds that would rather not pay
/// even that cost (notably wasm, where it is paid on every page load).
///
///   titanium tbgen --out tb_zero.bin     write the table
///   titanium tbgen --verify tb_zero.bin  check a file against a fresh solve
///   titanium tbgen                       report the census and hash only
/// Phase-1 label generator: solve the broke-hands subgame for ONE wall
/// configuration and emit every state as exact ground truth.
///
/// `titanium tbmerge --out STORE IN.tbpk [IN.tbpk ...]`
///
/// Fold packs into one store holding each configuration ONCE, converting to the
/// two-byte format on the way. Solving is already a DAG that never repeats
/// work; writing one pack per run is what duplicates it on disk, since every run
/// dumps its whole descendant cone and the cones overlap.
fn run_tbmerge(args: &[String]) {
    use titanium::titanium::research::tb_zero::TbSolver;
    let mut out = None::<String>;
    let mut inputs = Vec::<String>::new();
    let mut i = 2usize;
    while i < args.len() {
        if args[i] == "--out" && i + 1 < args.len() {
            out = Some(args[i + 1].clone());
            i += 2;
        } else {
            inputs.push(args[i].clone());
            i += 1;
        }
    }
    let Some(out) = out else {
        eprintln!("error: tbmerge needs --out STORE");
        std::process::exit(2);
    };
    if inputs.is_empty() {
        eprintln!("error: tbmerge needs at least one input pack");
        std::process::exit(2);
    }
    let before: u64 = inputs
        .iter()
        .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
        .sum();
    let t0 = std::time::Instant::now();
    match TbSolver::merge_packs(&inputs, &out) {
        Ok((distinct, seen)) => {
            let after = std::fs::metadata(&out).map(|m| m.len()).unwrap_or(0);
            println!(
                "{seen} records in, {distinct} distinct ({:.2}x duplication, {:.1}% redundant)",
                seen as f64 / distinct.max(1) as f64,
                100.0 * (1.0 - distinct as f64 / seen.max(1) as f64)
            );
            println!(
                "{:.2} GB -> {:.2} GB ({:.1}x smaller) in {:.1}s",
                before as f64 / 1e9,
                after as f64 / 1e9,
                before as f64 / after.max(1) as f64,
                t0.elapsed().as_secs_f64()
            );
        }
        Err(e) => {
            eprintln!("merge failed: {e}");
            std::process::exit(1);
        }
    }
}

/// `titanium tbsolve --hands A,B [--rng S] [--certify]`
///
/// Solve one tier exactly and report what it cost.
///
/// The tier is the HAND PAIR, not the total. A+B walls in hand means 20-(A+B)
/// on the board, but (1,1), (2,0) and (0,2) are the same board and three
/// different games — who holds a wall decides who may place it, so each needs
/// its own table and they are not interchangeable.
fn run_tbsolve(args: &[String]) {
    use titanium::titanium::research::{tb_layers, tb_zero::TbSolver};
    let mut hands = [1i32, 1i32];
    let mut rng = 0x5eedu64;
    let mut certify = false;
    let mut index = 0usize;
    let mut save = None::<String>;
    let mut load = None::<String>;
    let mut i = 2usize;
    while i < args.len() {
        match args[i].as_str() {
            "--index" if i + 1 < args.len() => {
                index = args[i + 1].parse().unwrap_or(0);
                i += 2;
            }
            "--hands" if i + 1 < args.len() => {
                let parts: Vec<i32> = args[i + 1]
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                if parts.len() != 2 {
                    eprintln!("error: --hands wants A,B (e.g. 1,1)");
                    std::process::exit(2);
                }
                hands = [parts[0], parts[1]];
                i += 2;
            }
            "--rng" if i + 1 < args.len() => {
                rng = args[i + 1].parse().unwrap_or(0x5eed);
                i += 2;
            }
            "--certify" => {
                certify = true;
                i += 1;
            }
            "--save" if i + 1 < args.len() => {
                save = Some(args[i + 1].clone());
                i += 2;
            }
            "--load" if i + 1 < args.len() => {
                load = Some(args[i + 1].clone());
                i += 2;
            }
            _ => i += 1,
        }
    }

    let total = (hands[0] + hands[1]) as usize;
    let Some(&seed) = tb_layers::seed_boards(1, rng).first() else {
        eprintln!("error: could not generate a 20-wall seed board");
        std::process::exit(1);
    };
    let layers = tb_layers::expand(&[seed], total);
    // Sorted, not `HashSet::iter().next()`. Rust randomises the hash seed per
    // process, so taking the first element of a set silently solves a DIFFERENT
    // configuration on every run -- which made these measurements irreproducible
    // and looked like the solver was nondeterministic.
    let mut candidates: Vec<_> = layers[total].iter().copied().collect();
    candidates.sort_unstable();
    if candidates.is_empty() {
        eprintln!("error: layer {total} is empty");
        std::process::exit(1);
    }
    let config = candidates[index % candidates.len()];
    let g = tb_layers::state_from_config(config, hands);
    assert_eq!(
        tb_layers::wall_count(config) as i32 + hands[0] + hands[1],
        20,
        "wall conservation"
    );

    println!(
        "tier ({}, {}): {} walls on board, {} legal pawn states",
        hands[0],
        hands[1],
        tb_layers::wall_count(config),
        tb_layers::live_state_count(config)
    );

    let t0 = std::time::Instant::now();
    let mut s = TbSolver::new();
    // Load first, so the tiers below are read from disk instead of re-solved.
    // That is what makes the ladder climbable one step per run.
    if let Some(path) = &load {
        match s.load(path) {
            Ok(n) => println!("loaded {n} tables from {path}"),
            Err(e) => {
                eprintln!("error loading {path}: {e}");
                std::process::exit(1);
            }
        }
    }
    let table = s.solve(&g);
    let solve_s = t0.elapsed().as_secs_f64();
    let (w, l, d) = table.census();

    println!(
        "solved in {solve_s:.2}s | {} tables held, {:.1}% cache hits, {:.0} MB resident",
        s.cached(),
        s.hit_rate() * 100.0,
        s.bytes() as f64 / 1e6
    );
    // Separate real draws from the illegal pawn states, which also sit at the
    // `TbEntry` default of Draw. Conflating them makes a table look like it
    // found mutual zugzwang when it only skipped unreachable squares.
    let excluded = 81 * 80 * 2 - tb_layers::live_state_count(config);
    let real_draws = d.saturating_sub(excluded);
    println!(
        "this tier's table: win={w} loss={l} | draws: {real_draws} real, {excluded} excluded as illegal"
    );
    if real_draws > 0 {
        println!("  ^ mutual zugzwang: neither side can force a win");
    }

    let (maxd, over) = table.distance_extremes();
    println!("max distance {maxd} plies, {over} states exceed i8 (+/-127)");

    if let Some(path) = &save {
        match s.save(path) {
            Ok(n) => {
                let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
                println!("saved {n} tables to {path} ({:.0} MB)", bytes as f64 / 1e6);
            }
            Err(e) => {
                eprintln!("error saving {path}: {e}");
                std::process::exit(1);
            }
        }
    }

    if certify {
        let t1 = std::time::Instant::now();
        match s.certify(&g) {
            Ok(()) => println!(
                "certified locally consistent in {:.2}s",
                t1.elapsed().as_secs_f64()
            ),
            Err(e) => {
                eprintln!("CERTIFY FAILED: {e}");
                std::process::exit(1);
            }
        }
    }
}

/// `titanium tblayers [--seeds N] [--depth K] [--rng S] [--out FILE]`
///
/// Enumerate the wall CONFIGURATIONS each tablebase tier has to cover, by
/// peeling walls off full 20-wall boards. Layer k has `20 - k` walls placed and
/// therefore k walls in hand.
///
/// This generates positions only. Who is owed the removed wall is not part of a
/// configuration -- it does not change the board, and it matters only once these
/// are handed to a solver.
///
/// Output is one JSONL row per configuration: `{layer, walls, hw, vw}`.
fn run_tblayers(args: &[String]) {
    use titanium::titanium::research::tb_layers;
    let mut seeds = 1usize;
    let mut depth = 4usize;
    let mut rng = 0x5eedu64;
    let mut out_path = None::<String>;
    let mut i = 2usize;
    while i < args.len() {
        let val = args.get(i + 1).and_then(|s| s.parse::<u64>().ok());
        match args[i].as_str() {
            "--seeds" => {
                seeds = val.unwrap_or(1) as usize;
                i += 2;
            }
            "--depth" => {
                depth = val.unwrap_or(4) as usize;
                i += 2;
            }
            "--rng" => {
                rng = val.unwrap_or(0x5eed);
                i += 2;
            }
            "--out" if i + 1 < args.len() => {
                out_path = Some(args[i + 1].clone());
                i += 2;
            }
            _ => i += 1,
        }
    }

    let t0 = std::time::Instant::now();
    let boards = tb_layers::seed_boards(seeds, rng);
    if boards.len() < seeds {
        eprintln!(
            "warning: only {} of {seeds} seed boards generated (the rest saturated before 20 walls)",
            boards.len()
        );
    }
    let layers = tb_layers::expand(&boards, depth);

    let mut total = 0usize;
    let mut total_states = 0usize;
    for (k, layer) in layers.iter().enumerate() {
        total += layer.len();
        // Pawn configurations, not just wall ones: a configuration is only half
        // the position. The gap against the unpruned 12,960 is the flood fill
        // removing pawn squares that can never reach a goal.
        let (live, max) = tb_layers::layer_state_census(layer);
        total_states += live;
        let pruned = if max == 0 {
            0.0
        } else {
            (max - live) as f64 * 100.0 / max as f64
        };
        println!(
            "layer {k}: {:>8} configs, {} on board, {} in hand | {:>12} pawn states ({pruned:.2}% pruned)",
            layer.len(),
            20 - k,
            k,
            live
        );
    }
    println!(
        "total {total} configurations, {total_states} pawn states, from {} seeds in {:.2}s",
        boards.len(),
        t0.elapsed().as_secs_f64()
    );

    if let Some(path) = out_path {
        let mut buf = String::new();
        for (k, layer) in layers.iter().enumerate() {
            // Sorted so the file is reproducible: a HashSet iterates in an
            // arbitrary order, and a dataset that differs run to run cannot be
            // diffed when something later disagrees.
            let mut rows: Vec<_> = layer.iter().copied().collect();
            rows.sort_unstable();
            for (hw, vw) in rows {
                buf.push_str(&format!(
                    "{{\"layer\":{k},\"walls\":{},\"hw\":{hw},\"vw\":{vw}}}\n",
                    20 - k
                ));
            }
        }
        match std::fs::write(&path, buf) {
            Ok(()) => println!("wrote {path}"),
            Err(e) => eprintln!("error writing {path}: {e}"),
        }
    }
}

/// `titanium tbdump --moves e2 e8 e3h ... [--out FILE]`
///
/// The move list supplies the walls; hands are then forced empty, which is the
/// subgame the table solves. Output is one JSONL row per live state:
/// `{packed, result, distance, best_move}` with `result` in W/L/D, `distance`
/// in plies (-1 for draws) and `best_move` a pawn destination (-1 when none).
///
/// Every row is EXACT -- retrograde to a fixpoint, not a search result -- so
/// these are tablebase-quality targets, not estimates. One configuration
/// yields up to 81*80*2 = 12,960 of them.
fn run_tbdump(args: &[String]) {
    use titanium::titanium::research::tb_zero::{TbResult, ZeroWallTb};
    let mut moves = Vec::<String>::new();
    let mut out_path = None::<String>;
    let mut i = 2usize;
    while i < args.len() {
        match args[i].as_str() {
            "--out" if i + 1 < args.len() => {
                out_path = Some(args[i + 1].clone());
                i += 2;
            }
            "--moves" => {
                i += 1;
                while i < args.len() && !args[i].starts_with("--") {
                    moves.push(args[i].clone());
                    i += 1;
                }
            }
            other => {
                eprintln!("unknown tbdump option: {other}");
                std::process::exit(2);
            }
        }
    }

    let mut g = titanium::GameState::new();
    for mv in &moves {
        g.make_move(titanium::algebraic_to_move_id(mv));
    }
    // Force the subgame: whatever walls the line placed stay as scenery, and
    // neither side has one left. This is legal by construction only if the
    // caller supplied a real line; the walls themselves are what matter here.
    g.wl = [0, 0];

    let t0 = Instant::now();
    let tb = ZeroWallTb::build_for(&g);
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut rows = String::new();
    let mut emitted = 0usize;
    for p0 in 0..81usize {
        for p1 in 0..81usize {
            if p0 == p1 {
                continue;
            }
            for stm in 0..2usize {
                let e = tb.probe_raw(p0, p1, stm);
                let mut pos = g.clone();
                pos.pawn[0] = p0;
                pos.pawn[1] = p1;
                pos.turn = stm;
                let packed: String = titanium::pack_state(&pos)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect();
                let r = match e.result {
                    TbResult::Win => "W",
                    TbResult::Loss => "L",
                    TbResult::Draw => "D",
                };
                rows.push_str(&format!(
                    "{{\"packed\":\"{packed}\",\"result\":\"{r}\",\"distance\":{},\"best_move\":{}}}
",
                    e.distance, e.best_move
                ));
                emitted += 1;
            }
        }
    }

    match out_path {
        Some(path) => {
            if let Err(e) = std::fs::write(&path, &rows) {
                eprintln!("tbdump: cannot write {path}: {e}");
                std::process::exit(1);
            }
            eprintln!("tbdump: {emitted} states -> {path} (build {build_ms:.1}ms)");
        }
        None => print!("{rows}"),
    }
}

fn run_tbgen(args: &[String]) {
    use titanium::titanium::research::tb_zero::ZeroWallTb;
    let mut out: Option<String> = None;
    let mut verify: Option<String> = None;
    let mut i = 2usize;
    while i < args.len() {
        match args[i].as_str() {
            "--out" if i + 1 < args.len() => {
                out = Some(args[i + 1].clone());
                i += 2;
            }
            "--verify" if i + 1 < args.len() => {
                verify = Some(args[i + 1].clone());
                i += 2;
            }
            other => {
                eprintln!("unknown tbgen option: {other}");
                std::process::exit(2);
            }
        }
    }

    let t0 = Instant::now();
    let tb = ZeroWallTb::build();
    let build_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let (w, l, d) = tb.census();
    let hash = tb.content_hash();
    println!(
        "tbgen solved live_states={} win={w} loss={l} draw={d} hash={hash:016x} build_ms={build_ms:.1}",
        tb.live_states()
    );

    if let Some(path) = verify {
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("tbgen: cannot read {path}: {e}");
                std::process::exit(1);
            }
        };
        match ZeroWallTb::from_bytes(&bytes) {
            Ok(loaded) => {
                if loaded.content_hash() == hash {
                    println!("tbgen verify OK: {path} matches a fresh solve");
                } else {
                    eprintln!("tbgen verify FAILED: {path} disagrees with a fresh solve");
                    std::process::exit(1);
                }
            }
            Err(e) => {
                eprintln!("tbgen verify FAILED: {path}: {e}");
                std::process::exit(1);
            }
        }
    }

    if let Some(path) = out {
        let bytes = tb.to_bytes();
        if let Err(e) = std::fs::write(&path, &bytes) {
            eprintln!("tbgen: cannot write {path}: {e}");
            std::process::exit(1);
        }
        println!("tbgen wrote {path} ({} bytes)", bytes.len());
    }
}

fn run_score_out(args: &[String]) {
    let mut nodes = None::<u64>;
    let mut packed_hex = None::<String>;
    let mut moves = Vec::<String>::new();
    let mut input_kind = None::<&str>;
    let mut error = None::<String>;
    let mut i = 2usize;

    while i < args.len() {
        match args[i].as_str() {
            "--nodes" => match args.get(i + 1).and_then(|s| s.parse::<u64>().ok()) {
                Some(value) if value > 0 => {
                    nodes = Some(value);
                    i += 2;
                }
                _ => {
                    error = Some("--nodes requires a positive u64".to_string());
                    break;
                }
            },
            "--packed" => match args.get(i + 1) {
                Some(value) => {
                    packed_hex = Some(value.clone());
                    input_kind = Some("packed");
                    i += 2;
                }
                None => {
                    error = Some("--packed requires hexadecimal bytes".to_string());
                    break;
                }
            },
            "--moves" => {
                input_kind = Some("moves");
                i += 1;
                while i < args.len() && !args[i].starts_with("--") {
                    moves.push(args[i].clone());
                    i += 1;
                }
            }
            flag if flag.starts_with("--") => {
                error = Some(format!("unknown score-out option: {flag}"));
                break;
            }
            value => {
                error = Some(format!("unexpected score-out argument: {value}"));
                break;
            }
        }
    }

    let budget = match (error, nodes) {
        (Some(message), _) => {
            println!("{}", score_out_error_json(&message));
            return;
        }
        (None, Some(value)) => value,
        (None, None) => {
            println!("{}", score_out_error_json("--nodes is required"));
            return;
        }
    };

    if (packed_hex.is_some() && input_kind != Some("packed"))
        || (packed_hex.is_none() && input_kind != Some("moves"))
        || (packed_hex.is_some() && !moves.is_empty())
    {
        println!(
            "{}",
            score_out_error_json("provide exactly one of --packed HEX or --moves MOVE...")
        );
        return;
    }

    let packed_for_engine = packed_hex.clone();
    let board = if let Some(hex) = packed_hex {
        match parse_packed_board(&hex) {
            Ok(board) => board,
            Err(message) => {
                println!("{}", score_out_error_json(&message));
                return;
            }
        }
    } else {
        let mut board = Board::new();
        for mv in &moves {
            if !looks_like_algebraic_move(mv) {
                println!(
                    "{}",
                    score_out_error_json(&format!("invalid algebraic move: {mv}"))
                );
                return;
            }
            board.apply_algebraic(mv);
        }
        board
    };

    if board.is_terminal().is_some() {
        println!("{}", score_out_error_json("position is terminal"));
        return;
    }

    // A long deadline makes the node budget the controlling limit without
    // allowing a malformed/degenerate position to run indefinitely.
    // Runs the production engine. This used to drive the legacy search, so every
    // teacher label the training pipeline collected through `score-out` was
    // produced by an engine that reaches depth 7 / 162k nodes in 3s where this
    // one reaches depth 13 / 985k. Labels from before this are not comparable.
    //
    // The old contract was a node budget; TitaniumSearch is time-bounded, so the
    // budget is converted at the production search's measured rate rather than
    // being silently ignored.
    const SCORE_OUT_NODES_PER_MS: u64 = 400;
    // Built from the same input the board was, because a Board carries no move
    // history and a guessed position would be scored silently.
    let game = if let Some(hex) = packed_for_engine {
        let mut packed = [0u8; titanium::PACKED_STATE_LEN];
        for (i, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
            packed[i] =
                u8::from_str_radix(std::str::from_utf8(pair).unwrap_or("zz"), 16).unwrap_or(0);
        }
        match titanium::titanium_game_from_packed(&packed) {
            Ok(g) => g,
            Err(message) => {
                println!("{}", score_out_error_json(&message));
                return;
            }
        }
    } else {
        let mut g = titanium::GameState::new();
        for mv in &moves {
            g.make_move(titanium::algebraic_to_move_id(mv));
        }
        g
    };
    // Echo the canonical position. Without it score-out states a verdict about
    // a board the caller cannot identify, which makes it useless for generating
    // training labels -- the whole point of proving a position offline is to
    // attach the proof to the position.
    //
    // WARNING: `packed` uses DATASET convention, the `side_to_move` field above
    // uses ENGINE convention, and they are OPPOSITE. pack_state writes
    // dataset player0 = engine pawn[1] and side_to_move = 1 - g.turn. So a
    // position with engine turn 1 reports "side_to_move":1 here and byte 5 == 0
    // inside `packed`. Both are correct; joining them naively mislabels the
    // side to move on every row. Decode `packed` with titanium_game_from_packed,
    // which is the exact inverse, rather than reading byte 5 against this field.
    let packed_out: String = titanium::pack_state(&game)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let mut search = titanium::TitaniumSearch::production(game, None);
    let time_ms = (budget / SCORE_OUT_NODES_PER_MS).clamp(1, 86_400_000);
    let result = search.think(time_ms, 128, false, false, "score-out");
    if result.mv == titanium::TITANIUM_NO_MOVE {
        println!("{}", score_out_error_json("no legal moves"));
        return;
    }
    struct ScoreOutReport {
        root_score: i32,
        search_depth: i32,
        nodes: u64,
    }
    let report = ScoreOutReport {
        root_score: result.score,
        search_depth: result.depth,
        nodes: result.nodes,
    };
    let score = report.root_score;
    let bound = if report.search_depth > 0 {
        "exact"
    } else {
        "unknown"
    };
    let proven = score.abs() >= 19_500 && report.search_depth > 0;
    println!(
        "{{\"schema\":\"score-out-v1\",\"ok\":true,\"input\":\"{}\",\"side_to_move\":{},\"score\":{},\"bound\":\"{}\",\"proven\":{},\"nodes\":{},\"node_budget\":{},\"depth\":{},\"selected_move\":\"{}\",\"packed\":\"{packed_out}\"}}",
        input_kind.unwrap_or("unknown"),
        match board.side() {
            titanium::Player::One => 0,
            titanium::Player::Two => 1,
        },
        score,
        bound,
        proven,
        report.nodes,
        budget,
        report.search_depth,
        json_escape(&titanium::move_id_to_algebraic(result.mv)),
    );
}

#[cfg(test)]
mod score_out_tests {
    use super::*;

    #[test]
    fn canonical_packed_frame_converts_to_board_coordinates() {
        let mut packed = [0u8; titanium::PACKED_STATE_LEN];
        packed[0] = 1;
        packed[1] = 4; // canonical player 0: e1 / bottom goal
        packed[2] = 76; // canonical player 1: e9 / top goal
        packed[3] = 10;
        packed[4] = 10;
        packed[5] = 0; // player 0 to move
        let board = parse_packed_board(
            &packed
                .iter()
                .map(|b| format!("{b:02x}"))
                .collect::<String>(),
        )
        .expect("valid canonical frame");

        assert_eq!(board.pawns, [(0, 4), (8, 4)]);
        assert_eq!(board.walls_remaining, [10, 10]);
        assert_eq!(board.side_to_move, titanium::Player::One);
        assert_eq!(board.horizontal_walls, 0);
        assert_eq!(board.vertical_walls, 0);
    }

    #[test]
    fn score_out_error_is_single_json_frame() {
        let frame = score_out_error_json("bad \"packed\" input");
        assert!(frame.starts_with("{\"schema\":\"score-out-v1\""));
        assert!(frame.contains("\"ok\":false"));
        assert!(frame.contains("bad \\\"packed\\\" input"));
        assert!(frame.ends_with('}'));
    }
}

fn score_out_error_json(message: &str) -> String {
    format!(
        "{{\"schema\":\"score-out-v1\",\"ok\":false,\"error\":\"{}\"}}",
        json_escape(message)
    )
}

fn looks_like_algebraic_move(arg: &str) -> bool {
    let b = arg.as_bytes();
    b.len() >= 2 && b[0].is_ascii_lowercase() && b[1].is_ascii_digit()
}

fn parse_packed_board(hex: &str) -> Result<Board, String> {
    if hex.len() != titanium::PACKED_STATE_LEN * 2 {
        return Err(format!(
            "packed state hex must contain exactly {} bytes",
            titanium::PACKED_STATE_LEN
        ));
    }
    let mut packed = [0u8; titanium::PACKED_STATE_LEN];
    for (i, pair) in hex.as_bytes().chunks_exact(2).enumerate() {
        let text = std::str::from_utf8(pair).map_err(|_| "packed state is not ASCII hex")?;
        packed[i] =
            u8::from_str_radix(text, 16).map_err(|_| format!("invalid packed hex at byte {i}"))?;
    }
    let fields = titanium::decode_packed_state(&packed)?;
    let board = Board {
        pawns: [
            (fields.player0_cell / 9, fields.player0_cell % 9),
            (fields.player1_cell / 9, fields.player1_cell % 9),
        ],
        walls_remaining: [fields.player0_walls, fields.player1_walls],
        horizontal_walls: fields.horizontal_walls,
        vertical_walls: fields.vertical_walls,
        side_to_move: if fields.side_to_move == 0 {
            titanium::Player::One
        } else {
            titanium::Player::Two
        },
        move_number: 1,
        hash: 0,
    };
    let mut board = board;
    board.hash = titanium::core::zobrist::hash_board(&board);
    Ok(board)
}

/// Returns the engine flag if it routes through Titanium search (`ace-v13*` / `titanium-v*`).
/// True ACE engines (`ace`, `ace-v8`, …) live in the separate `ace` crate.
fn ace_engine_flag(args: &[String]) -> Option<&str> {
    args.windows(2).find_map(|w| {
        if w[0] != "--engine" {
            return None;
        }
        match w[1].as_str() {
            // ACE v13 reference engines (JS-equivalent baselines) — TitaniumSearch
            "ace-v13" | "ace-v13-ti" | "ace-v13-ti-pmc" | "ace-v13-pure" | "ace-v13-grafted"
            | "ace-v13-grafted-no-raceproof" | "ace-v13-ti-pure"
            // Titanium production engines
            | "titanium-v14" | "titanium-v15" | "titanium-v15-medium" | "titanium-v15-frozen"
            | "titanium-v16" | "titanium-v17" | "titanium-v15-no-raceproof" => Some(w[1].as_str()),
            _ => None,
        }
    })
}

fn parse_threads_arg(args: &[String]) -> usize {
    let mut threads = 1usize;
    let mut i = 0usize;
    while i < args.len() {
        if args[i] == "--threads" {
            let Some(raw) = args.get(i + 1) else {
                eprintln!("error --threads requires a value");
                std::process::exit(2);
            };
            match raw.parse::<usize>() {
                Ok(0) | Err(_) => {
                    eprintln!("error --threads must be a positive integer");
                    std::process::exit(2);
                }
                Ok(v) => {
                    threads = v.min(16);
                    i += 2;
                    continue;
                }
            }
        }
        i += 1;
    }
    threads
}
