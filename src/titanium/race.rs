//! pathfix/RaceProof — fixed-topology no-more-walls race system.
//!
//! Scope: **both wall hands are empty**, so the blocked-edge topology is frozen
//! permanently. Walls may already be on the board; only pawn moves, jumps and
//! diagonal jumps remain. Every API here is correct for *arbitrary* legal
//! fixed-wall topologies, not just the empty board.
//!
//! Two separate services, by design:
//!
//! **Service A — fast outcome / α-β bound** ([`race_outcome`]):
//!   Near-instant theorem deduction of the side-to-move's forced result, as an
//!   alpha-beta-native [`RaceBound`] (`Lower(RACE_WIN_FLOOR)` for a proven win,
//!   `Upper(-RACE_WIN_FLOOR)` for a proven loss, `Unknown` when it declines). It
//!   builds **no successor graph** and computes **no exact DTM**.
//!
//!   Sound decision rule (correct on ANY fixed-wall topology): if the two pawns'
//!   shortest-path SETS are **disjoint** they can never share a cell, so no jump
//!   / interception is possible and the race is a pure independent tempo race —
//!   the turn-adjusted faster pawn wins exactly ([`separated_pure_race_verdict`]).
//!   When the path sets **overlap**, interception can swing the result and no
//!   cheap proof is sound, so Service A returns `Unknown` and the caller falls
//!   back to ordinary search (or the exact service). It NEVER returns a false bound.
//!
//!   NOTE: two earlier cheap deciders were found **unsound on walled topologies**
//!   and are intentionally NOT used here: (a) the in-module winner-*sign*
//!   recursion (its sign disagreed with the retrograde oracle on random walled
//!   boards — masked because the old equality tests compared only the retrograde
//!   output, never the sign table); and (b) `cert_bridge::race_minimax`'s
//!   distance-decreasing-only forward proof (restricting the opponent's
//!   interception moves manufactures false wins). Both are exact only on the
//!   empty board, where optimal race play is always distance-decreasing.
//!
//! **Service B — optional exact DTM** ([`race_exact_dtm_on_demand`], [`solve_race_config`]):
//!   Exact `+k / −k` distance-to-mate, used only when a caller genuinely needs
//!   it (fastest-win / slowest-loss / stubborn-loser selection, UI, tests). Its
//!   ~160 KB successor-graph scratch is allocated on first use and reused; it is
//!   **never** invoked on the bound-only path. Computed by an exact ply-round
//!   retrograde over the live successor graph — the algorithm proven `+k/−k`-equal
//!   to the reference oracle on the empty board, all sample configs and 1,000
//!   random fixed topologies. (It is self-contained: it does NOT depend on any
//!   winner-sign field.)
//!
//! `solve_race_config_reference` remains a `#[cfg(test)]` oracle only.

use crate::titanium::cert_bridge::{paths_overlap, separated_pure_race_verdict, RaceVerdict};
use crate::titanium::game::GameState;

/// 81 × 81 × 2 (p0 cell, p1 cell, side to move).
pub const RACE_STATES: usize = 13_122;

/// Legal live pawn placements: p0 ∉ goal row, p1 ∉ goal row, p0 ≠ p1, both turns.
pub const RACE_LIVE_STATES: usize = 10_242;

/// Race-proof score band: above every heuristic eval, below the true-mate band.
/// Exact-DTM table values:
///   +k = side to move wins in k plies,
///   -k = side to move loses in k plies,
///    0 = illegal/unused state.
pub const RACE_MATE: i32 = 32_000;

/// Hard cap on race plies (the retrograde fixpoint bound). Every exact race
/// score therefore satisfies `RACE_MATE - RACE_MAX_PLIES < |score| ≤ RACE_MATE`.
pub const RACE_MAX_PLIES: i32 = 1_024;

/// Proven-outcome α-β bound magnitude. A theorem-proved win is a LOWER bound of
/// `RACE_WIN_FLOOR` (the true score is some exact `RACE_MATE - k ≥ RACE_WIN_FLOOR`);
/// a proven loss is an UPPER bound of `-RACE_WIN_FLOOR`. Chosen to sit strictly
///   - above every heuristic evaluation (race heuristic peaks well under 10 000),
///   - at or below every exact race-win score (`RACE_MATE - k`, `k < RACE_MAX_PLIES`),
///   - far below the real-mate band (`MATE − 1000 = 99 000`),
/// so it is always safe for fail-high / fail-low use and never collides with a
/// heuristic leaf or a true mate.
pub const RACE_WIN_FLOOR: i32 = RACE_MATE - RACE_MAX_PLIES;

/// Fast race outcome as an alpha-beta-native bound (Service A).
///
/// Never returns an invented exact score: a proven win is a LOWER bound, a proven
/// loss an UPPER bound. `Exact` is produced only by the on-demand exact service.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RaceBound {
    /// Side-to-move is a forced winner: true score ≥ this lower bound.
    Lower(i32),
    /// Side-to-move is a forced loser: true score ≤ this upper bound.
    Upper(i32),
    /// Genuine exact distance-to-mate (only from the exact service).
    Exact(i32),
    /// Not resolved by the fast theorem — caller must fall back to search.
    Unknown,
}

impl RaceBound {
    /// Proven win or loss → the bound's signum (+1 / −1); otherwise 0.
    #[inline]
    pub fn signum(self) -> i32 {
        match self {
            RaceBound::Lower(_) => 1,
            RaceBound::Upper(_) => -1,
            RaceBound::Exact(v) => v.signum(),
            RaceBound::Unknown => 0,
        }
    }
}

/// Reusable solver scratch.
///
/// The bound path ([`race_outcome`]) needs nothing from here — it uses the
/// classifier's own tiny transient scratch. The exact successor-graph tier
/// (~160 KB, Service B) is allocated lazily on first exact use and reused.
pub struct RaceScratch {
    /// Lazy exact-DTM successor graph (Service B), allocated on demand.
    exact: Option<Box<ExactScratch>>,
}

/// Exact-DTM successor-graph scratch — live-only, ~160 KB. Lazily allocated.
struct ExactScratch {
    graph_slot: Box<[u16]>,
    live: Box<[u16]>,
    nsucc: Box<[u8]>,
    succ: Box<[i16]>,
    buf: [i16; 16],
}

impl ExactScratch {
    fn new() -> Self {
        Self {
            graph_slot: vec![0u16; RACE_STATES].into_boxed_slice(),
            live: vec![0u16; RACE_LIVE_STATES].into_boxed_slice(),
            nsucc: vec![0u8; RACE_LIVE_STATES].into_boxed_slice(),
            succ: vec![0i16; RACE_LIVE_STATES * 5].into_boxed_slice(),
            buf: [0; 16],
        }
    }

    const fn bytes() -> usize {
        RACE_STATES * std::mem::size_of::<u16>()
            + RACE_LIVE_STATES * std::mem::size_of::<u16>()
            + RACE_LIVE_STATES * std::mem::size_of::<u8>()
            + RACE_LIVE_STATES * 5 * std::mem::size_of::<i16>()
            + std::mem::size_of::<[i16; 16]>()
    }
}

impl RaceScratch {
    pub fn new() -> Self {
        Self { exact: None }
    }

    /// Resident bytes on the bound-only path (the exact tier is not allocated).
    pub const fn scratch_bytes() -> usize {
        std::mem::size_of::<Option<Box<ExactScratch>>>()
    }

    /// Additional heap when the exact (Service B) tier is lazily allocated.
    pub const fn exact_scratch_bytes() -> usize {
        ExactScratch::bytes()
    }

    /// Whether the exact successor-graph tier is currently allocated.
    pub fn exact_allocated(&self) -> bool {
        self.exact.is_some()
    }
}

impl Default for RaceScratch {
    fn default() -> Self {
        Self::new()
    }
}

#[inline(always)]
fn state_id(p0: usize, p1: usize, turn: usize) -> usize {
    (p0 * 81 + p1) * 2 + turn
}

#[inline(always)]
fn decode_state(id: usize) -> (usize, usize, usize) {
    let turn = id % 2;
    let pp = id / 2;
    (pp / 81, pp % 81, turn)
}

#[inline(always)]
fn is_home(side: usize, cell: usize) -> bool {
    if side == 0 {
        cell < 9
    } else {
        cell >= 72
    }
}

// ---------------------------------------------------------------------------
// Service A — fast outcome / alpha-beta bound (no successor graph, no exact DTM).
// ---------------------------------------------------------------------------

/// Forced-outcome bound for the side to move at the current hands-empty state.
///
/// Sound on ANY fixed-wall topology: decides only when the pawns' shortest-path
/// sets are disjoint (pure tempo race), otherwise declines with
/// [`RaceBound::Unknown`]. No successor graph, no exact DTM, never a false bound.
/// The `_s` scratch is unused today but kept for API symmetry / future memoization.
///
/// Pre-condition (debug-checked): both pawns are off their own goal rows — the
/// caller handles terminal states.
pub fn race_outcome(g: &mut GameState, _s: &mut RaceScratch) -> RaceBound {
    debug_assert!(
        g.pawn[0] >= 9 && g.pawn[1] < 72,
        "race_outcome on terminal state"
    );
    let mut d0 = [0u8; 81];
    let mut d1 = [0u8; 81];
    g.compute_dist(0, &mut d0);
    g.compute_dist(1, &mut d1);
    if d0[g.pawn[0]] == u8::MAX || d1[g.pawn[1]] == u8::MAX {
        return RaceBound::Unknown;
    }
    // Overlapping shortest-path sets → interception possible → no cheap sound
    // proof; decline and let the caller search / use the exact service.
    if paths_overlap(g, &d0, &d1) {
        return RaceBound::Unknown;
    }
    match separated_pure_race_verdict(g) {
        RaceVerdict::Win => RaceBound::Lower(RACE_WIN_FLOOR),
        RaceVerdict::Loss => RaceBound::Upper(-RACE_WIN_FLOOR),
        RaceVerdict::NeedsProof => RaceBound::Unknown,
    }
}

/// Convenience: `Some(true)` = stm forced win, `Some(false)` = forced loss,
/// `None` = undecided (caller falls back to search).
#[inline]
pub fn race_outcome_stm_wins(g: &mut GameState, s: &mut RaceScratch) -> Option<bool> {
    match race_outcome(g, s) {
        RaceBound::Lower(_) => Some(true),
        RaceBound::Upper(_) => Some(false),
        RaceBound::Exact(v) => Some(v > 0),
        RaceBound::Unknown => None,
    }
}

// ---------------------------------------------------------------------------
// Service B — exact DTM (lazy successor-graph retrograde). Proven +k/-k-exact.
// ---------------------------------------------------------------------------

fn build_live_graph(
    g: &mut GameState,
    graph_slot: &mut [u16],
    live: &mut [u16],
    nsucc: &mut [u8],
    succ: &mut [i16],
    buf: &mut [i16; 16],
) -> usize {
    graph_slot.fill(0);
    let mut n = 0usize;
    let (saved_p0, saved_p1, saved_turn) = (g.pawn[0], g.pawn[1], g.turn);

    for p0 in 9..81usize {
        g.pawn[0] = p0;
        for p1 in 0..72usize {
            if p1 == p0 {
                continue;
            }
            g.pawn[1] = p1;

            for turn in 0..2usize {
                let id = state_id(p0, p1, turn);
                graph_slot[id] = n as u16;
                live[n] = id as u16;
                g.turn = turn;

                let nm = g.gen_pawn_moves(buf, 0);
                debug_assert!(nm <= 5);
                nsucc[n] = nm as u8;
                let off = n * 5;

                for j in 0..nm {
                    let c = buf[j] as usize;
                    succ[off + j] = if turn == 0 {
                        if c < 9 {
                            -1
                        } else {
                            state_id(c, p1, 1) as i16
                        }
                    } else if c >= 72 {
                        -1
                    } else {
                        state_id(p0, c, 0) as i16
                    };
                }
                n += 1;
            }
        }
    }

    g.pawn[0] = saved_p0;
    g.pawn[1] = saved_p1;
    g.turn = saved_turn;

    debug_assert_eq!(n, RACE_LIVE_STATES);
    n
}

/// Ply-round retrograde DTM over the live successor cache. Self-contained:
/// exact `+k = 1 + min losing-child magnitude`, `-k = 1 + max winning-child`.
fn fill_exact_dtm(g: &mut GameState, ex: &mut ExactScratch, tbl: &mut [i16]) {
    tbl.fill(0);

    let n_live = build_live_graph(
        g,
        &mut ex.graph_slot,
        &mut ex.live,
        &mut ex.nsucc,
        &mut ex.succ,
        &mut ex.buf,
    );
    let mut n_unresolved = n_live;
    let mut k = 1i32;

    while n_unresolved > 0 && k < RACE_MAX_PLIES {
        let mut assigned = 0usize;
        let mut keep = 0usize;

        for i in 0..n_unresolved {
            let id = ex.live[i] as usize;

            let gi = ex.graph_slot[id] as usize;
            let ns = ex.nsucc[gi] as usize;
            let off = gi * 5;

            let mut min_loss = i32::MAX;
            let mut all_win = ns > 0;
            let mut max_win = 0i32;

            for j in 0..ns {
                let nid = ex.succ[off + j];
                if nid < 0 {
                    min_loss = min_loss.min(0);
                    all_win = false;
                    continue;
                }

                let v = tbl[nid as usize] as i32;
                if v < 0 {
                    all_win = false;
                    min_loss = min_loss.min(-v);
                } else if v > 0 {
                    max_win = max_win.max(v);
                } else {
                    all_win = false;
                }
            }

            if min_loss != i32::MAX && min_loss + 1 == k {
                tbl[id] = k as i16;
                assigned += 1;
                continue;
            }

            if all_win && max_win + 1 == k {
                tbl[id] = -k as i16;
                assigned += 1;
                continue;
            }

            ex.live[keep] = id as u16;
            keep += 1;
        }

        n_unresolved = keep;
        if assigned == 0 {
            break;
        }
        k += 1;
    }

    debug_assert_eq!(
        n_unresolved, 0,
        "DTM pass left {n_unresolved} unresolved states"
    );
}

/// Fill the complete fixed-topology exact race table (Service B). Lazily
/// allocates/reuses the ~160 KB successor-graph scratch.
pub fn solve_race_config(g: &mut GameState, s: &mut RaceScratch, tbl: &mut [i16]) {
    debug_assert_eq!(tbl.len(), RACE_STATES);
    if s.exact.is_none() {
        s.exact = Some(Box::new(ExactScratch::new()));
    }
    let ex = s.exact.as_mut().expect("exact scratch");
    fill_exact_dtm(g, ex, tbl);
}

/// Exact distance-to-mate for the *current* state only (Service B, on demand).
///
/// Builds (or reuses) the exact full table for this topology into `tbl`, then
/// returns `+k / −k` for the current `(p0, p1, turn)`. `None` if the state is
/// off the live set. The caller owns `tbl` (it may cache it per topology); this
/// routine is never called on the bound-only search path.
pub fn race_exact_dtm_on_demand(
    g: &mut GameState,
    s: &mut RaceScratch,
    tbl: &mut [i16],
) -> Option<i16> {
    debug_assert_eq!(tbl.len(), RACE_STATES);
    solve_race_config(g, s, tbl);
    let v = tbl[state_id(g.pawn[0], g.pawn[1], g.turn)];
    if v == 0 {
        None
    } else {
        Some(v)
    }
}

// ---------------------------------------------------------------------------
// Test-only exhaustive reference oracle.
// ---------------------------------------------------------------------------

#[cfg(test)]
struct ReferenceScratch {
    succ: Box<[i16]>,
    nsucc: Box<[u8]>,
    live: Box<[i32]>,
    buf: [i16; 16],
}

#[cfg(test)]
impl ReferenceScratch {
    fn new() -> Self {
        Self {
            succ: vec![0i16; RACE_STATES * 5].into_boxed_slice(),
            nsucc: vec![0u8; RACE_STATES].into_boxed_slice(),
            live: vec![0i32; RACE_STATES].into_boxed_slice(),
            buf: [0; 16],
        }
    }
}

#[cfg(test)]
fn solve_race_config_reference(g: &mut GameState, s: &mut ReferenceScratch, tbl: &mut [i16]) {
    debug_assert_eq!(tbl.len(), RACE_STATES);
    let (sp0, sp1, sturn) = (g.pawn[0], g.pawn[1], g.turn);
    tbl.fill(0);

    let mut n_live = 0usize;
    for p0 in 9..81usize {
        g.pawn[0] = p0;
        for p1 in 0..72usize {
            if p1 == p0 {
                continue;
            }
            g.pawn[1] = p1;
            let base = state_id(p0, p1, 0);

            g.turn = 0;
            let nm = g.gen_pawn_moves(&mut s.buf, 0);
            debug_assert!(nm <= 5);
            s.nsucc[base] = nm as u8;
            let off = base * 5;
            for j in 0..nm {
                let c = s.buf[j] as usize;
                s.succ[off + j] = if c < 9 { -1 } else { state_id(c, p1, 1) as i16 };
            }
            s.live[n_live] = base as i32;
            n_live += 1;

            g.turn = 1;
            let nm = g.gen_pawn_moves(&mut s.buf, 0);
            debug_assert!(nm <= 5);
            s.nsucc[base + 1] = nm as u8;
            let off = (base + 1) * 5;
            for j in 0..nm {
                let c = s.buf[j] as usize;
                s.succ[off + j] = if c >= 72 {
                    -1
                } else {
                    state_id(p0, c, 0) as i16
                };
            }
            s.live[n_live] = (base + 1) as i32;
            n_live += 1;
        }
    }

    g.pawn[0] = sp0;
    g.pawn[1] = sp1;
    g.turn = sturn;

    let mut k = 1i32;
    while n_live > 0 && k < 1024 {
        let mut assigned = 0usize;
        let mut keep = 0usize;

        for i in 0..n_live {
            let id = s.live[i] as usize;
            let ns = s.nsucc[id] as usize;
            let mut min_loss = 32_767i32;
            let mut all_win = ns > 0;
            let mut max_win = 0i32;
            let off = id * 5;

            for j in 0..ns {
                let nid = s.succ[off + j];
                if nid < 0 {
                    min_loss = 0;
                    all_win = false;
                    continue;
                }

                let v = tbl[nid as usize] as i32;
                if v < 0 {
                    all_win = false;
                    min_loss = min_loss.min(-v);
                } else if v > 0 {
                    max_win = max_win.max(v);
                } else {
                    all_win = false;
                }
            }

            if min_loss + 1 == k {
                tbl[id] = k as i16;
                assigned += 1;
                continue;
            }

            if all_win && max_win + 1 == k {
                tbl[id] = -k as i16;
                assigned += 1;
                continue;
            }

            s.live[keep] = id as i32;
            keep += 1;
        }

        n_live = keep;
        if assigned == 0 {
            break;
        }
        k += 1;
    }
}

#[cfg(test)]
fn gen_successor_ids_for_test(
    g: &mut GameState,
    id: usize,
    buf: &mut [i16; 16],
    succ_out: &mut [i16; 5],
) -> usize {
    let (p0, p1, turn) = decode_state(id);
    g.pawn[0] = p0;
    g.pawn[1] = p1;
    g.turn = turn;

    let nm = g.gen_pawn_moves(buf, 0);
    debug_assert!(nm <= 5);

    for j in 0..nm {
        let c = buf[j] as usize;
        succ_out[j] = if turn == 0 {
            if c < 9 {
                -1
            } else {
                state_id(c, p1, 1) as i16
            }
        } else if c >= 72 {
            -1
        } else {
            state_id(p0, c, 0) as i16
        };
    }
    nm
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solved_empty_board() -> Vec<i16> {
        let mut g = GameState::new();
        let mut s = RaceScratch::new();
        let mut tbl = vec![0i16; RACE_STATES];
        solve_race_config(&mut g, &mut s, &mut tbl);
        tbl
    }

    /// Replay a shuffled wall/pawn prefix with both hands empty (deterministic LCG).
    fn random_fixed_topology(seed_state: &mut u64) -> GameState {
        use crate::titanium::algebraic_to_move_id;
        fn next_u64(rng: &mut u64) -> u64 {
            *rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            *rng
        }
        let pool: [&str; 24] = [
            "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "f6h", "c3h", "d4v", "e5v", "h6h", "a3h",
            "d6h", "f4v", "c5v", "h1h", "b4h", "g5h", "a7h", "f1h", "c7h", "d1h", "b1h",
        ];
        let mut order: Vec<usize> = (0..pool.len()).collect();
        for i in 0..order.len() {
            let j = (next_u64(seed_state) as usize) % order.len();
            order.swap(i, j);
        }
        let n_moves = 8 + (next_u64(seed_state) as usize) % (pool.len() - 7);
        let mut g = GameState::new();
        for &idx in &order[..n_moves] {
            g.make_move(algebraic_to_move_id(pool[idx]));
        }
        // Scope: both hands empty (no walls placeable). The race model ignores
        // wl, but set it so positions are faithful to the no-more-walls endgame.
        g.wl = [0, 0];
        g
    }

    fn compare_tables(
        fast: &[i16],
        reference: &[i16],
    ) -> (
        usize,
        usize,
        usize,
        Option<(usize, i16, i16)>,
        Option<(usize, i16, i16)>,
    ) {
        let mut live = 0usize;
        let mut sign_mismatches = 0usize;
        let mut exact_mismatches = 0usize;
        let mut first_sign = None;
        let mut first_exact = None;

        for id in 0..RACE_STATES {
            if reference[id] == 0 && fast[id] == 0 {
                continue;
            }
            live += 1;
            if fast[id].signum() != reference[id].signum() {
                sign_mismatches += 1;
                first_sign.get_or_insert((id, fast[id], reference[id]));
            }
            if fast[id] != reference[id] {
                exact_mismatches += 1;
                first_exact.get_or_insert((id, fast[id], reference[id]));
            }
        }

        (
            live,
            sign_mismatches,
            exact_mismatches,
            first_sign,
            first_exact,
        )
    }

    fn print_mismatch(label: &str, id: usize, fast: i16, reference: i16) {
        let (p0, p1, turn) = decode_state(id);
        eprintln!("{label}: id={id} p0={p0} p1={p1} turn={turn} fast={fast} ref={reference}");
    }

    // ── 1. Exhaustive empty-board exact equality (Service B) ──────────────────

    #[test]
    fn empty_board_exhaustive_exact_equality() {
        let mut g = GameState::new();

        let mut fast_scratch = RaceScratch::new();
        let mut fast = vec![0i16; RACE_STATES];
        solve_race_config(&mut g, &mut fast_scratch, &mut fast);

        let mut ref_scratch = ReferenceScratch::new();
        let mut reference = vec![0i16; RACE_STATES];
        solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);

        let (live, sign_m, exact_m, first_sign, first_exact) = compare_tables(&fast, &reference);

        if let Some((id, f, r)) = first_sign {
            print_mismatch("first sign mismatch", id, f, r);
        }
        if let Some((id, f, r)) = first_exact {
            print_mismatch("first exact mismatch", id, f, r);
        }

        eprintln!("empty-board: live={live} sign_mismatches={sign_m} exact_mismatches={exact_m}");

        assert_eq!(sign_m, 0, "sign mismatches on empty board");
        assert_eq!(exact_m, 0, "exact mismatches on empty board");
    }

    /// Service A (`race_outcome`) — on every DECISIVE live empty-board state its
    /// bound sign must match the exact oracle. (Unknown is allowed; it is never a
    /// false bound.) The bound path must allocate no exact graph.
    #[test]
    fn empty_board_race_outcome_bound_sign_audit() {
        let mut g = GameState::new();
        let mut ref_scratch = ReferenceScratch::new();
        let mut reference = vec![0i16; RACE_STATES];
        solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);

        let mut s = RaceScratch::new();
        let mut decisive = 0usize;
        let mut unknown = 0usize;
        for p0 in 9..81usize {
            for p1 in 0..72usize {
                if p0 == p1 {
                    continue;
                }
                for turn in 0..2usize {
                    let id = state_id(p0, p1, turn);
                    if reference[id] == 0 {
                        continue;
                    }
                    g.pawn[0] = p0;
                    g.pawn[1] = p1;
                    g.turn = turn;
                    let bound = race_outcome(&mut g, &mut s);
                    match bound {
                        RaceBound::Unknown => unknown += 1,
                        _ => {
                            decisive += 1;
                            assert_eq!(
                                bound.signum(),
                                reference[id].signum() as i32,
                                "race_outcome sign mismatch id={id} p0={p0} p1={p1} turn={turn} bound={bound:?} ref={}",
                                reference[id]
                            );
                        }
                    }
                    assert!(!s.exact_allocated(), "race_outcome allocated exact scratch");
                }
            }
        }
        eprintln!("race_outcome empty-board: decisive={decisive} unknown={unknown}");
        assert!(decisive > 0);
    }

    // ── 2. Fixed-wall sample configs ─────────────────────────────────────────

    #[test]
    fn exact_matches_reference_on_sample_configs() {
        use crate::titanium::algebraic_to_move_id;

        let configs: [&[&str]; 3] = [
            &[],
            &["e2", "e8", "e3h", "e6h"],
            &["e2", "e8", "c3h", "f6v", "d7h", "b4v"],
        ];

        for moves in configs {
            let mut g = GameState::new();
            for m in moves {
                g.make_move(algebraic_to_move_id(m));
            }

            let mut fast_scratch = RaceScratch::new();
            let mut fast = vec![0i16; RACE_STATES];
            solve_race_config(&mut g, &mut fast_scratch, &mut fast);

            let mut ref_scratch = ReferenceScratch::new();
            let mut reference = vec![0i16; RACE_STATES];
            solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);

            let (_, sign_m, exact_m, first_sign, first_exact) = compare_tables(&fast, &reference);

            assert_eq!(
                sign_m, 0,
                "sign mismatch; moves={moves:?}, first={first_sign:?}"
            );
            assert_eq!(
                exact_m, 0,
                "exact mismatch; moves={moves:?}, first={first_exact:?}"
            );
        }
    }

    // ── 3. Random legal fixed topologies (exact + bound sign) ────────────────

    #[test]
    fn random_fixed_topology_exact_equality_1000() {
        let seed: u64 = 0xACE5_2026;
        let mut rng = seed;

        const N: usize = 1_000;
        let mut fast_scratch = RaceScratch::new();
        let mut ref_scratch = ReferenceScratch::new();
        for trial in 0..N {
            let mut g = random_fixed_topology(&mut rng);

            let mut fast = vec![0i16; RACE_STATES];
            solve_race_config(&mut g, &mut fast_scratch, &mut fast);

            let mut reference = vec![0i16; RACE_STATES];
            solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);

            let (_, sign_m, exact_m, first_sign, first_exact) = compare_tables(&fast, &reference);
            if sign_m != 0 || exact_m != 0 {
                eprintln!(
                    "random topology failure trial={trial} seed={seed} pawns=({},{}) turn={}",
                    g.pawn[0], g.pawn[1], g.turn
                );
                if let Some((id, f, r)) = first_sign {
                    print_mismatch("sign", id, f, r);
                }
                if let Some((id, f, r)) = first_exact {
                    print_mismatch("exact", id, f, r);
                }
            }

            assert_eq!(sign_m, 0, "trial {trial} seed {seed} sign mismatch");
            assert_eq!(exact_m, 0, "trial {trial} seed {seed} exact mismatch");
        }
    }

    /// Service A soundness on WALLED topologies: across 1,000 random fixed
    /// topologies, EVERY decisive `race_outcome` bound must agree in sign with the
    /// exact oracle. (Unknown is allowed — it is never a false bound.) This is the
    /// gate that the in-module winner-sign recursion failed, motivating the switch
    /// to the proven cert_bridge resolver.
    #[test]
    fn random_fixed_topology_race_outcome_bound_sign_1000() {
        let seed: u64 = 0x71744E_1ACE;
        let mut rng = seed;
        const N: usize = 1_000;
        let mut s = RaceScratch::new();
        let mut ref_scratch = ReferenceScratch::new();
        let mut reference = vec![0i16; RACE_STATES];
        let mut decisive = 0usize;
        let mut unknown = 0usize;
        let mut g_probe = GameState::new();

        for trial in 0..N {
            let mut g = random_fixed_topology(&mut rng);
            solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);

            // Probe a deterministic spread of live states. Rebuild a *consistent*
            // GameState per probe by replaying onto a clone with the topology's
            // walls — so the classifier sees a valid position.
            for step in 0..24usize {
                let p0 = 9 + (step * 7 + trial) % 72;
                let p1 = (step * 13 + 2 * trial) % 72;
                if p0 == p1 {
                    continue;
                }
                let turn = step % 2;
                let id = state_id(p0, p1, turn);
                if reference[id] == 0 {
                    continue;
                }
                // Place pawns directly on a fresh clone of the walled topology.
                g_probe.clone_from(&g);
                g_probe.pawn[0] = p0;
                g_probe.pawn[1] = p1;
                g_probe.turn = turn;
                let bound = race_outcome(&mut g_probe, &mut s);
                match bound {
                    RaceBound::Unknown => unknown += 1,
                    _ => {
                        decisive += 1;
                        assert_eq!(
                            bound.signum(),
                            reference[id].signum() as i32,
                            "outcome sign trial={trial} seed={seed} p0={p0} p1={p1} turn={turn} bound={bound:?} ref={}",
                            reference[id]
                        );
                    }
                }
            }
            assert!(
                !s.exact_allocated(),
                "bound path must not allocate exact scratch"
            );
        }
        eprintln!(
            "race_outcome walled audit: decisive={decisive} unknown={unknown} (seed={seed})"
        );
        assert!(decisive > 0, "must exercise decisive bounds on walled boards");
    }

    // ── 4. Child-preservation audit (validates outcome-based move filtering) ──

    /// For every proven-winning state at least one legal child is a loss for the
    /// opponent; for every proven-losing state every legal child is a win for the
    /// opponent. Verified against the exact oracle on the empty board.
    #[test]
    fn child_preservation_audit_empty_board() {
        let mut g = GameState::new();
        let mut ref_scratch = ReferenceScratch::new();
        let mut reference = vec![0i16; RACE_STATES];
        solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);

        let mut buf = [0i16; 16];
        let mut succ = [0i16; 5];
        let mut win_states = 0usize;
        let mut loss_states = 0usize;

        for id in 0..RACE_STATES {
            let v = reference[id];
            if v == 0 {
                continue;
            }
            let ns = gen_successor_ids_for_test(&mut g, id, &mut buf, &mut succ);

            if v > 0 {
                win_states += 1;
                let preserves = (0..ns).any(|j| {
                    let nid = succ[j];
                    nid < 0 || reference[nid as usize] < 0
                });
                assert!(preserves, "winning state {id} has no winning child");
            } else {
                loss_states += 1;
                for j in 0..ns {
                    let nid = succ[j];
                    assert!(
                        nid >= 0,
                        "losing state {id} has an immediate-goal move (would be a win)"
                    );
                    assert!(
                        reference[nid as usize] > 0,
                        "losing state {id} has a non-winning child {nid}"
                    );
                }
            }
        }
        eprintln!("child-preservation: win_states={win_states} loss_states={loss_states}");
        assert!(win_states > 0 && loss_states > 0);
    }

    // ── 5. Alpha-beta bound correctness ──────────────────────────────────────

    const MATE_GUARD: i32 = 99_000;

    /// `race_outcome` lower/upper bounds must never cross the true exact score:
    /// a LOWER bound ≤ the exact race-win score; an UPPER bound ≥ the exact
    /// race-loss score. Both must stay above the heuristic band and below mate.
    #[test]
    fn race_outcome_bounds_never_cross_exact() {
        let mut g = GameState::new();
        let mut ref_scratch = ReferenceScratch::new();
        let mut reference = vec![0i16; RACE_STATES];
        solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);

        let mut s = RaceScratch::new();
        for p0 in 9..81usize {
            for p1 in 0..72usize {
                if p0 == p1 {
                    continue;
                }
                for turn in 0..2usize {
                    let id = state_id(p0, p1, turn);
                    let rv = reference[id] as i32;
                    if rv == 0 {
                        continue;
                    }
                    // Exact α-β score the engine assigns from this leaf.
                    let exact_score = if rv > 0 {
                        RACE_MATE - rv
                    } else {
                        -(RACE_MATE + rv) // rv<0 → -(RACE_MATE - |rv|)
                    };
                    g.pawn[0] = p0;
                    g.pawn[1] = p1;
                    g.turn = turn;
                    match race_outcome(&mut g, &mut s) {
                        RaceBound::Lower(b) => {
                            assert!(rv > 0, "LOWER bound on a non-win state {id}");
                            assert!(
                                b <= exact_score,
                                "LOWER bound {b} exceeds exact {exact_score} at {id}"
                            );
                            assert!(b > 9_000, "LOWER bound {b} not above heuristic band");
                            assert!(b < MATE_GUARD, "LOWER bound {b} reaches mate band");
                        }
                        RaceBound::Upper(b) => {
                            assert!(rv < 0, "UPPER bound on a non-loss state {id}");
                            assert!(
                                b >= exact_score,
                                "UPPER bound {b} below exact {exact_score} at {id}"
                            );
                            assert!(b < -9_000, "UPPER bound {b} not below heuristic band");
                            assert!(b > -MATE_GUARD, "UPPER bound {b} reaches mate band");
                        }
                        RaceBound::Exact(_) => panic!("Service A must not return Exact"),
                        RaceBound::Unknown => {} // allowed: no claim
                    }
                }
            }
        }
    }

    // ── 6. Existing regressions ──────────────────────────────────────────────

    #[test]
    fn empty_board_head_on_race_is_movers_loss() {
        let tbl = solved_empty_board();
        let p0 = 76;
        let p1 = 4;
        assert_eq!(tbl[state_id(p0, p1, 0)], -16);
        assert_eq!(tbl[state_id(p0, p1, 1)], -16);
    }

    #[test]
    fn immediate_jump_to_goal_wins_in_one_ply() {
        let tbl = solved_empty_board();
        let p0 = 18;
        let p1 = 9;
        assert_eq!(tbl[state_id(p0, p1, 0)], 1);
    }

    #[test]
    fn one_step_from_goal_wins_immediately() {
        let tbl = solved_empty_board();
        let p0 = 13;
        let p1 = 40;
        assert_eq!(tbl[state_id(p0, p1, 0)], 1);
    }

    #[test]
    fn race_table_is_bellman_consistent_on_sample_configs() {
        use crate::titanium::algebraic_to_move_id;

        let configs: [&[&str]; 3] = [
            &[],
            &["e2", "e8", "e3h", "e6h"],
            &["e2", "e8", "c3h", "f6v", "d7h", "b4v"],
        ];

        for moves in configs {
            let mut g = GameState::new();
            for m in moves {
                g.make_move(algebraic_to_move_id(m));
            }

            let mut fast_scratch = RaceScratch::new();
            let mut tbl = vec![0i16; RACE_STATES];
            solve_race_config(&mut g, &mut fast_scratch, &mut tbl);

            let mut buf = [0i16; 16];
            let mut succ = [0i16; 5];

            for id in 0..RACE_STATES {
                let v = tbl[id] as i32;
                if v == 0 {
                    continue;
                }

                let ns = gen_successor_ids_for_test(&mut g, id, &mut buf, &mut succ);
                let mut min_loss = i32::MAX;
                let mut all_resolved_win = ns > 0;
                let mut max_win = 0i32;

                for j in 0..ns {
                    let nid = succ[j];
                    if nid < 0 {
                        min_loss = min_loss.min(0);
                        all_resolved_win = false;
                        continue;
                    }

                    let sv = tbl[nid as usize] as i32;
                    if sv < 0 {
                        all_resolved_win = false;
                        min_loss = min_loss.min(-sv);
                    } else if sv > 0 {
                        max_win = max_win.max(sv);
                    } else {
                        all_resolved_win = false;
                    }
                }

                if v > 0 {
                    assert_eq!(v, min_loss + 1, "win value mismatch at state {id}");
                } else {
                    assert!(all_resolved_win, "loss state {id} has a non-win successor");
                    assert_eq!(-v, max_win + 1, "loss value mismatch at state {id}");
                }
            }
        }
    }

    #[test]
    fn ka_game_ply67_stubborn_loser_root_moves() {
        use crate::titanium::algebraic_to_move_id;
        use crate::titanium::move_id_to_algebraic;

        let moves = [
            "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "f6h", "c3h", "d4v", "e5v", "h6h", "a3h",
            "d6h", "f4v", "c5v", "h1h", "b4h", "g5h", "a7h", "f1h", "c7h", "d1h", "e5", "e6", "e4",
            "d6", "f4", "d5", "f5", "d4", "f6", "c4", "g6", "b4", "h6", "a4", "i6", "a5", "i5",
            "b5", "i4", "b6", "h4", "c6", "b6h", "b6", "h3", "a6", "g3", "a7", "f3", "b7", "e3",
            "c7", "d3", "d7", "d2", "e7", "c2", "b1h", "e7h", "d7", "b2", "c7", "a2",
        ];

        let mut g = GameState::new();
        for m in moves {
            g.make_move(algebraic_to_move_id(m));
        }

        let mut s = RaceScratch::new();
        let mut tbl = vec![0i16; RACE_STATES];
        solve_race_config(&mut g, &mut s, &mut tbl);

        let id = state_id(g.pawn[0], g.pawn[1], g.turn);
        let rv = tbl[id] as i32;
        let me = g.turn;
        let mut buf = [0i16; 16];
        let nm = g.gen_pawn_moves(&mut buf, 0);
        let mut best_key = i32::MIN;
        let mut best_alg = String::new();

        for &mv in &buf[..nm] {
            let c = mv as usize;
            let my_v = if is_home(me, c) {
                1
            } else {
                let child_id = if me == 0 {
                    state_id(c, g.pawn[1], 1)
                } else {
                    state_id(g.pawn[0], c, 0)
                };

                let v = tbl[child_id] as i32;
                if v == 0 {
                    continue;
                }

                if v > 0 {
                    -(v + 1)
                } else {
                    1 - v
                }
            };

            let key = if my_v > 0 {
                1_000_000 - my_v
            } else {
                -1_000_000 - my_v
            };

            if key > best_key {
                best_key = key;
                best_alg = move_id_to_algebraic(mv);
            }
        }

        assert!(rv < 0, "white must be in a proven loss");
        assert_eq!(
            best_alg, "b7",
            "b7 and d7 tie on race plies; b7 wins move-order tie-break"
        );
    }

    // ── 7. On-demand exact API + lazy lifecycle ──────────────────────────────

    #[test]
    fn on_demand_exact_matches_full_table_and_is_lazy() {
        let mut g = GameState::new();
        let mut s = RaceScratch::new();

        // Bound queries first: no exact graph yet.
        g.pawn[0] = 40;
        g.pawn[1] = 41;
        g.turn = 0;
        let _ = race_outcome(&mut g, &mut s);
        assert!(!s.exact_allocated(), "bound query must stay graph-free");

        // On-demand exact: allocates the graph, returns the same value as the
        // full table for this state, and agrees with the oracle.
        let mut tbl = vec![0i16; RACE_STATES];
        let v = race_exact_dtm_on_demand(&mut g, &mut s, &mut tbl);
        assert!(s.exact_allocated(), "exact request must allocate the graph");
        let id = state_id(g.pawn[0], g.pawn[1], g.turn);
        assert_eq!(v, Some(tbl[id]));

        let mut ref_scratch = ReferenceScratch::new();
        let mut reference = vec![0i16; RACE_STATES];
        solve_race_config_reference(&mut g, &mut ref_scratch, &mut reference);
        assert_eq!(v, Some(reference[id]));
    }

    // ── Benchmarks (printed; assert correctness) ─────────────────────────────

    #[test]
    fn benchmark_services_and_scratch() {
        let mut g = GameState::new();

        const ITERS: u32 = 200;
        let n = u128::from(ITERS);

        // (1/4) ordinary bound path: one lazy outcome query.
        let mut bound_ns = 0u128;
        let mut s = RaceScratch::new();
        for _ in 0..ITERS {
            g.pawn[0] = 40;
            g.pawn[1] = 41;
            g.turn = 0;
            let t = std::time::Instant::now();
            let _ = race_outcome(&mut g, &mut s);
            bound_ns += t.elapsed().as_nanos();
        }
        assert!(!s.exact_allocated(), "bound path must not allocate exact graph");

        // (7) exact cold (fresh scratch each iter — includes the lazy alloc).
        let mut exact_cold_us = 0u128;
        for _ in 0..ITERS {
            let mut s = RaceScratch::new();
            let mut tbl = vec![0i16; RACE_STATES];
            let t = std::time::Instant::now();
            solve_race_config(&mut g, &mut s, &mut tbl);
            exact_cold_us += t.elapsed().as_micros();
        }

        // (8) exact cached (graph already allocated; reused).
        let mut exact_cached_us = 0u128;
        {
            let mut s = RaceScratch::new();
            let mut tbl = vec![0i16; RACE_STATES];
            solve_race_config(&mut g, &mut s, &mut tbl);
            for _ in 0..ITERS {
                let t = std::time::Instant::now();
                solve_race_config(&mut g, &mut s, &mut tbl);
                exact_cached_us += t.elapsed().as_micros();
            }
        }

        eprintln!(
            "race-bench: bound_query_ns={} exact_cold_us={} exact_cached_us={} bound_scratch_bytes={} exact_scratch_bytes={}",
            bound_ns / n,
            exact_cold_us / n,
            exact_cached_us / n,
            RaceScratch::scratch_bytes(),
            RaceScratch::exact_scratch_bytes(),
        );
    }
}
