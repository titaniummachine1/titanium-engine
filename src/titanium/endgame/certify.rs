//! ACE v13 static win certificates — 1:1 port of the inlined `certify_win.js`
//! (`ACEV13.html` RP_CERT, lines 473–888).
//!
//! Claim "side S wins" iff S's race lead survives the opponent O's best
//! reactive wall campaign in a restricted subgame: S may only step strictly
//! toward goal (plus one equal-dist re-commit move right after each O wall);
//! O may place any legal wall or make any legal pawn move. Value = AND over O,
//! OR over S, memoized on the incremental zobrist. PROVEN ⇒ S wins the real
//! game (the restriction only removes options from the maximizing side, so the
//! certificate is pessimistic/sound). See the JS header for the full soundness
//! argument and residual caveats.
//!
//! gen13: the JS inlines this so `RP_CERT` exists in node AND browser; the v11
//! Rust port (mirroring the browser's `RP_CERT === null`) omitted it. This is
//! the file that makes the v13 port faithful.

use crate::titanium::dist::fill_ace_dist_from_pawn;
use crate::titanium::game::GameState;
use crate::titanium::race::{estimated_plies_to_result, RaceBound, RACE_WIN_FLOOR};
use crate::titanium::time_alloc::LengthBound;
use crate::util::clock::Instant;

/// Budget / deadline abort. Mirrors the JS `BUDGET_EX` throw: it unwinds
/// through the recursion; every `make_move` is paired with an `unmake_move`
/// before the `?` propagates, so the board is restored as the error climbs.
#[derive(Debug, Clone, Copy)]
pub struct Budget;

type RecResult = Result<bool, Budget>;

/// Options for [`certify`]. Defaults mirror the JS `certify(game, opts)`.
#[derive(Debug, Clone)]
pub struct CertifyOpts {
    /// Total node budget across both candidate sides (JS default 200000).
    pub budget: u64,
    /// Absolute wall-clock deadline; `None` = no deadline.
    pub deadline: Option<Instant>,
    /// `false` = 'all' (sound, default); `true` = 'pruned' (measurement only).
    pub mode_pruned: bool,
    /// Corridor slack for the near-wall relevance test (JS default 2).
    pub slack: i32,
    /// Force the candidate side; `None` = favored race winner then the other.
    pub side: Option<usize>,
    /// Equal-dist S move allowed right after an O wall (JS default true).
    pub recommit: bool,
}

impl Default for CertifyOpts {
    fn default() -> Self {
        Self {
            budget: 200_000,
            deadline: None,
            mode_pruned: false,
            slack: 2,
            side: None,
            recommit: true,
        }
    }
}

/// Result of [`certify`] — score cut plus remaining-game length for TM.
///
/// `bound` is the αβ [`RaceBound`] (Lower/Upper floor when proven). `length` is
/// never confused with that cut: min may come from geometry / walking ETA; max
/// is set only for a known forced end (exact terminal or exact DTM).
#[derive(Debug, Clone)]
pub struct CertifyReport {
    /// `Some(side)` if that side's win is proven; `None` otherwise.
    pub proven: Option<usize>,
    /// Total certify nodes burned (used for the search's failure-memo work
    /// accounting — a deadline-starved run stamps only the work it actually did).
    pub nodes: u64,
    /// Score cut for search (`Lower`/`Upper` floor, or `Unknown`).
    pub bound: RaceBound,
    /// Remaining-game length for time management / horizon (not an αβ cut).
    pub length: LengthBound,
}

/// Pack proven side + node count into typed score/length fields.
fn pack_report(game: &GameState, proven: Option<usize>, nodes: u64) -> CertifyReport {
    let geom = LengthBound::optimistic_board(game.turn, game.pawn);
    if game.winner() >= 0 {
        let w = game.winner() as usize;
        let bound = if w == game.turn {
            RaceBound::Lower(RACE_WIN_FLOOR)
        } else {
            RaceBound::Upper(-RACE_WIN_FLOOR)
        };
        return CertifyReport {
            proven: Some(w),
            nodes,
            bound,
            length: LengthBound::exact(0),
        };
    }
    let Some(winner) = proven else {
        return CertifyReport {
            proven: None,
            nodes,
            bound: RaceBound::Unknown,
            length: geom,
        };
    };
    let bound = if winner == game.turn {
        RaceBound::Lower(RACE_WIN_FLOOR)
    } else {
        RaceBound::Upper(-RACE_WIN_FLOOR)
    };
    // Walking ETA raises min only — certify does not invent an exact DTM max.
    let mut length = geom;
    let mut d0 = [0u8; 81];
    let mut d1 = [0u8; 81];
    game.compute_dist(0, &mut d0);
    game.compute_dist(1, &mut d1);
    let wd = if winner == 0 {
        d0[game.pawn[0]]
    } else {
        d1[game.pawn[1]]
    };
    if wd != u8::MAX {
        let est = estimated_plies_to_result(game, winner, wd) as u32;
        length = length.merge(LengthBound::with_min(est));
    }
    CertifyReport {
        proven: Some(winner),
        nodes,
        bound,
        length,
    }
}

// ── helpers ─────────────────────────────────────────────────────────────────

/// Engine race convention: side to move wins iff `dMe <= dOpp`.
fn race_winner_stm(turn: usize, d0: u8, d1: u8) -> usize {
    let (d_me, d_opp) = if turn == 0 { (d0, d1) } else { (d1, d0) };
    if d_me <= d_opp {
        turn
    } else {
        1 - turn
    }
}

/// Exact no-wall race winner. This is the hard boundary between the expensive
/// wall-campaign certifier and the already-solved pawn race: once both hands are
/// empty, never recurse through certificate nodes.
fn no_wall_race_winner(g: &mut GameState) -> usize {
    use crate::titanium::cert_bridge::hands_empty_race_stm_wins;
    let stm_wins = hands_empty_race_stm_wins(g).unwrap_or(false);
    if stm_wins {
        g.turn
    } else {
        1 - g.turn
    }
}

/// Wall geometry: top-left cell of the 2×2 block a slot covers.
#[inline]
fn wall_cell_a(slot: usize) -> usize {
    (slot / 8) * 9 + (slot % 8)
}

/// Chebyshev-distance proximity test (`max(|dr|, |dc|) <= rad`).
fn cheb_near(cell_a: usize, cell_b: usize, rad: i32) -> bool {
    let dr = (cell_a as i32 / 9) - (cell_b as i32 / 9);
    let dc = (cell_a as i32 % 9) - (cell_b as i32 % 9);
    let dr = dr.abs();
    let dc = dc.abs();
    dr.max(dc) <= rad
}

// ── the solver ───────────────────────────────────────────────────────────────

/// Wall-interdiction subgame solver for a single candidate side `s`.
struct Solver<'a> {
    g: &'a mut GameState,
    s: usize,
    budget: u64,
    deadline: Option<Instant>,
    mode_pruned: bool,
    recommit: bool,
    slack: i32,
    nodes: u64,
    /// `(hashLo, hashHi*2 + allowEq)` → subgame value (S wins?).
    memo: std::collections::HashMap<(u32, u64), bool>,
    /// No-wall race terminal cache reached from the 1-2 wall campaign.
    /// Key includes frozen placed walls, pawns, and side-to-move via Ace hash.
    race_memo: std::collections::HashMap<(u32, u32), usize>,
}

impl<'a> Solver<'a> {
    fn no_wall_race_winner_cached(&mut self) -> usize {
        let key = (self.g.hash_lo, self.g.hash_hi);
        if let Some(&winner) = self.race_memo.get(&key) {
            return winner;
        }
        let winner = no_wall_race_winner(self.g);
        self.race_memo.insert(key, winner);
        winner
    }

    /// `jw` = the previous move was an O wall (enables S's recommit move).
    fn rec(&mut self, jw: bool) -> RecResult {
        self.nodes += 1;
        if self.nodes > self.budget {
            return Err(Budget);
        }
        if self.nodes & 1023 == 0 {
            if let Some(dl) = self.deadline {
                if Instant::now() > dl {
                    return Err(Budget);
                }
            }
        }
        let s = self.s;
        let o = 1 - s;

        let w = self.g.winner();
        if w >= 0 {
            return Ok(w as usize == s);
        }
        if self.g.wl[0] == 0 && self.g.wl[1] == 0 {
            return Ok(self.no_wall_race_winner_cached() == s);
        }

        // S's legal-move set depends on jw (recommit rule), so key it at S nodes.
        let allow_eq = self.recommit && jw && self.g.turn == s;
        let lo = self.g.hash_lo;
        let hi_key = (self.g.hash_hi as u64) * 2 + if allow_eq { 1 } else { 0 };
        if let Some(&hit) = self.memo.get(&(lo, hi_key)) {
            return Ok(hit);
        }

        let res;
        if self.g.turn == s {
            // OR node: S steps strictly down its dist field (jumps included).
            let mut d_s = [0u8; 81];
            self.g.compute_dist(s, &mut d_s);
            let cur = d_s[self.g.pawn[s]];
            let mut buf = [0i16; 16];
            let n = self.g.gen_pawn_moves(&mut buf, 0);
            let mut found = false;
            if cur != 255 {
                // strictly-decreasing targets (plus equal-dist right after an O
                // wall), best (lowest dist) first.
                let mut cand = [0i16; 16];
                let mut ck = [0u8; 16];
                let mut cn = 0usize;
                for i in 0..n {
                    let dv = d_s[buf[i] as usize];
                    if dv < cur || (allow_eq && dv == cur) {
                        cand[cn] = buf[i];
                        ck[cn] = dv;
                        cn += 1;
                    }
                }
                insertion_sort(&mut cand[..cn], &mut ck[..cn]);
                for i in 0..cn {
                    self.g.make_move(cand[i]);
                    let v = self.rec(false);
                    self.g.unmake_move();
                    if v? {
                        found = true;
                        break;
                    }
                }
            }
            res = found;
        } else {
            // AND node: O may race/jam with the pawn or drop any (relevant) wall.
            let mut ok = true;
            let mut d_o = [0u8; 81];
            self.g.compute_dist(o, &mut d_o);
            let mut buf = [0i16; 16];
            let n = self.g.gen_pawn_moves(&mut buf, 0);
            // pawn moves first, racing (lowest dO) first — cheapest refutations
            let mut pm = [0i16; 16];
            let mut pk = [0u8; 16];
            for i in 0..n {
                pm[i] = buf[i];
                pk[i] = d_o[buf[i] as usize];
            }
            insertion_sort(&mut pm[..n], &mut pk[..n]);
            let mut i = 0;
            while i < n && ok {
                self.g.make_move(pm[i]);
                let v = self.rec(false);
                self.g.unmake_move();
                if !v? {
                    ok = false;
                }
                i += 1;
            }
            if ok && self.g.wl[o] > 0 {
                // wall moves: classify against S's shortest-path corridor
                let mut d_s2 = [0u8; 81];
                self.g.compute_dist(s, &mut d_s2);
                let mut dp = [u8::MAX; 81];
                fill_ace_dist_from_pawn(self.g, self.g.pawn[s], &mut dp);
                let total = d_s2[self.g.pawn[s]] as i32;
                let p_s = self.g.pawn[s];
                let p_o = self.g.pawn[o];
                let mut wm = [0i16; 140];
                let mut wk = [0u8; 140];
                let mut wn = 0usize;
                for wtype in 0..2usize {
                    for slot in 0..64usize {
                        if !self.g.wall_legal(wtype, slot) {
                            continue;
                        }
                        let a_c = wall_cell_a(slot);
                        let (c1, c2, c3, c4, e1u, e1v, e2u, e2v);
                        if wtype == 0 {
                            c1 = a_c;
                            c2 = a_c + 1;
                            c3 = a_c + 9;
                            c4 = a_c + 10;
                            e1u = a_c;
                            e1v = a_c + 9;
                            e2u = a_c + 1;
                            e2v = a_c + 10;
                        } else {
                            c1 = a_c;
                            c2 = a_c + 9;
                            c3 = a_c + 1;
                            c4 = a_c + 10;
                            e1u = a_c;
                            e1v = a_c + 1;
                            e2u = a_c + 9;
                            e2v = a_c + 10;
                        }
                        // does the wall cut an edge on SOME shortest S path?
                        let dp = &dp;
                        let edge = |u: usize, v: usize| -> bool {
                            (dp[u] as i32 + 1 + d_s2[v] as i32 == total)
                                || (dp[v] as i32 + 1 + d_s2[u] as i32 == total)
                        };
                        let cuts = edge(e1u, e1v) || edge(e2u, e2v);
                        let mut near = cuts;
                        if !near {
                            let corridor = |c: usize| -> bool {
                                dp[c] as i32 + d_s2[c] as i32 <= total + self.slack
                            };
                            near = corridor(c1)
                                || corridor(c2)
                                || corridor(c3)
                                || corridor(c4)
                                || cheb_near(c1, p_s, 1)
                                || cheb_near(c4, p_s, 1)
                                || cheb_near(c2, p_s, 1)
                                || cheb_near(c3, p_s, 1)
                                || cheb_near(c1, p_o, 1)
                                || cheb_near(c4, p_o, 1)
                                || cheb_near(c2, p_o, 1)
                                || cheb_near(c3, p_o, 1);
                        }
                        if self.mode_pruned && !near {
                            continue;
                        }
                        wm[wn] = (if wtype == 0 {
                            crate::titanium::MOVE_HW_BASE
                        } else {
                            crate::titanium::MOVE_VW_BASE
                        }) + slot as i16;
                        wk[wn] = if cuts {
                            0
                        } else if near {
                            1
                        } else {
                            2
                        };
                        wn += 1;
                    }
                }
                insertion_sort(&mut wm[..wn], &mut wk[..wn]);
                let mut j = 0;
                while j < wn && ok {
                    self.g.make_move(wm[j]);
                    let v = self.rec(true); // wall just placed: S may re-commit
                    self.g.unmake_move();
                    if !v? {
                        ok = false;
                    }
                    j += 1;
                }
            }
            res = ok;
        }

        self.memo.insert((lo, hi_key), res);
        Ok(res)
    }
}

/// Insertion sort of `moves` keyed by `keys` (ascending), kept in lockstep.
/// Mirrors the JS tiny insertion sorts (cand/ck, pm/pk, wm/wk).
fn insertion_sort(moves: &mut [i16], keys: &mut [u8]) {
    for a in 1..moves.len() {
        let mv = moves[a];
        let kv = keys[a];
        let mut b = a as isize - 1;
        while b >= 0 && keys[b as usize] > kv {
            moves[(b + 1) as usize] = moves[b as usize];
            keys[(b + 1) as usize] = keys[b as usize];
            b -= 1;
        }
        moves[(b + 1) as usize] = mv;
        keys[(b + 1) as usize] = kv;
    }
}

// ── snapshot / restore (budget abort unwind, cf. JS snap/restore) ────────────

struct Snapshot {
    pawn: [usize; 2],
    wl: [i32; 2],
    turn: usize,
    hash_lo: u32,
    hash_hi: u32,
    hist_len: usize,
    last_wall_ply: usize,
    wall_stamp: i32,
    hw: [u8; 64],
    vw: [u8; 64],
    blocked: [u8; 81],
}

fn snap(g: &GameState) -> Snapshot {
    Snapshot {
        pawn: g.pawn,
        wl: g.wl,
        turn: g.turn,
        hash_lo: g.hash_lo,
        hash_hi: g.hash_hi,
        hist_len: g.hist_len,
        last_wall_ply: g.last_wall_ply,
        wall_stamp: g.wall_stamp,
        hw: g.hw,
        vw: g.vw,
        blocked: g.blocked,
    }
}

fn restore(g: &mut GameState, s: &Snapshot) {
    g.pawn = s.pawn;
    g.wl = s.wl;
    g.turn = s.turn;
    g.hash_lo = s.hash_lo;
    g.hash_hi = s.hash_hi;
    g.hist_len = s.hist_len;
    g.last_wall_ply = s.last_wall_ply;
    g.wall_stamp = s.wall_stamp;
    g.hw = s.hw;
    g.vw = s.vw;
    g.blocked = s.blocked;
}

/// Budget-capped static win certificate for the current position.
///
/// Returns `proven: Some(side)` iff that side's win is certified under the
/// 'all' (sound) mode. Tries the favored race winner first, then the other
/// side, within the shared node budget. 1:1 with the JS `certify(game, opts)`.
pub fn certify(game: &mut GameState, opts: &CertifyOpts) -> CertifyReport {
    let w = game.winner();
    if w >= 0 {
        return pack_report(game, Some(w as usize), 0);
    }
    if game.wl[0] + game.wl[1] > 3 {
        return pack_report(game, None, 0);
    }
    if game.wl[0] == 0 && game.wl[1] == 0 {
        let winner = no_wall_race_winner(game);
        let proven = match opts.side {
            Some(s @ (0 | 1)) if s != winner => None,
            _ => Some(winner),
        };
        return pack_report(game, proven, 0);
    }
    let mut d0 = [0u8; 81];
    let mut d1 = [0u8; 81];
    game.compute_dist(0, &mut d0);
    game.compute_dist(1, &mut d1);
    let favored = race_winner_stm(game.turn, d0[game.pawn[0]], d1[game.pawn[1]]);
    let order: Vec<usize> = match opts.side {
        Some(s @ (0 | 1)) => vec![s],
        _ => vec![favored, 1 - favored],
    };
    let budget = opts.budget;
    let mut total_nodes = 0u64;
    for s in order {
        if total_nodes >= budget {
            break;
        }
        let left = budget - total_nodes;
        let before = snap(game);
        let (verdict, nodes) = {
            let mut solver = Solver {
                g: game,
                s,
                budget: left,
                deadline: opts.deadline,
                mode_pruned: opts.mode_pruned,
                recommit: opts.recommit,
                slack: opts.slack,
                nodes: 0,
                memo: std::collections::HashMap::new(),
                race_memo: std::collections::HashMap::new(),
            };
            let v = solver.rec(false);
            (v, solver.nodes)
        };
        total_nodes += nodes;
        match verdict {
            Ok(true) => {
                return pack_report(game, Some(s), total_nodes);
            }
            Ok(false) => {}
            Err(Budget) => restore(game, &before),
        }
    }
    pack_report(game, None, total_nodes)
}

/// Upfront one-wall relaxation check — for the soundness counterexample ONLY
/// (this relaxation is NOT a certificate). With `wl[O] == 1`, "place O's wall
/// up front for free, then dist-race from the same stm" survives for S iff for
/// EVERY placeable wall the race still adjudicates to S. Ported for parity with
/// `certify_win.js`; not on the engine's hot path.
pub fn upfront_one_wall_survives(game: &mut GameState, s: usize) -> bool {
    let mut d0 = [0u8; 81];
    let mut d1 = [0u8; 81];
    for wtype in 0..2usize {
        for slot in 0..64usize {
            if !game.wall_fits(wtype, slot) {
                continue;
            }
            game.set_wall_bits(wtype, slot, true);
            let alive = game.has_path(0) && game.has_path(1);
            let mut refuted = false;
            if alive {
                game.compute_dist(0, &mut d0);
                game.compute_dist(1, &mut d1);
                let win = race_winner_stm(game.turn, d0[game.pawn[0]], d1[game.pawn[1]]);
                if win != s {
                    refuted = true;
                }
            }
            game.set_wall_bits(wtype, slot, false);
            if refuted {
                return false;
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn race_game(p0: usize, p1: usize, turn: usize) -> GameState {
        let mut g = GameState::new();
        g.pawn = [p0, p1];
        g.wl = [0, 0];
        g.turn = turn;
        g
    }

    #[test]
    fn no_wall_root_uses_race_shortcut_even_with_zero_budget() {
        // Board coords (3,1) vs (5,7): disjoint shortest paths, equal distance,
        // side to move wins by pure tempo math. ACE cell = (8-row)*9+col.
        let mut g = race_game(46, 34, 0);
        let report = certify(
            &mut g,
            &CertifyOpts {
                budget: 0,
                side: None,
                ..Default::default()
            },
        );
        assert_eq!(report.proven, Some(0));
        assert_eq!(report.nodes, 0);
        assert_eq!(report.bound, RaceBound::Lower(RACE_WIN_FLOOR));
        assert!(report.length.min_plies.is_some());
        // Certify does not invent Exact DTM for no-wall shortcut.
        assert!(report.length.max_plies.is_none());
    }

    #[test]
    fn no_wall_root_respects_forced_side_filter() {
        let mut g = race_game(46, 34, 0);
        let report = certify(
            &mut g,
            &CertifyOpts {
                budget: 0,
                side: Some(1),
                ..Default::default()
            },
        );
        assert_eq!(report.proven, None);
        assert_eq!(report.nodes, 0);
        assert_eq!(report.bound, RaceBound::Unknown);
        assert!(report.length.min_plies.is_some());
    }

    #[test]
    fn high_wall_positions_are_outside_certificate_scope() {
        let mut g = GameState::new();
        let report = certify(
            &mut g,
            &CertifyOpts {
                budget: 200_000,
                ..Default::default()
            },
        );
        assert_eq!(report.proven, None);
        assert_eq!(report.nodes, 0);
        assert_eq!(report.bound, RaceBound::Unknown);
    }

    #[test]
    fn terminal_position_emits_exact_zero_length() {
        let mut g = race_game(4, 40, 0); // p0 already on goal row
        assert!(g.winner() >= 0);
        let report = certify(&mut g, &CertifyOpts::default());
        assert_eq!(report.proven, Some(0));
        assert_eq!(report.length, LengthBound::exact(0));
        assert_eq!(report.bound, RaceBound::Lower(RACE_WIN_FLOOR));
    }
}
