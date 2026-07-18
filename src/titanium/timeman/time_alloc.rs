//! Sudden-death move budget from remaining game clock (`go rem`).
//!
//! Base plan ≈ mean completed-game length (~60 plies → **30 own moves**). Do
//! **not** plan the long tail (P90/P95); when a game runs long, raise the
//! horizon with `min(dist_to_win_p0, dist_to_win_p1)` from eval/jump-aware
//! distances. Site-style spendFactor/shareCap + Stockfish `MAX_RATIO` hard
//! ceiling. Soft-stop 85/92% still applies inside search on hard `move_ms`.
//!
//! Corpus P95 is telemetry + a light leftover guard only — never the divisor.

/// Minimum think slice so a near-empty clock still searches something.
pub const MIN_MOVE_MS: u64 = 50;

/// Baseline own-move plan from mean game length (~60 plies / 2). Extensions
/// via eval `min(d0,d1)` cover longer games; do not bake in P95 here.
pub const PLAN_OWN_MOVES: f64 = 30.0;

/// Hard ceiling vs optimum (Stockfish-style steal room).
pub const MAX_RATIO: f64 = 1.25;

/// Remaining-game length bound for TM / horizon (not an αβ score cut).
///
/// Separate from [`crate::titanium::race::RaceBound`]: min/max plies must not be confused
/// with Lower/Upper mate scores. `certify()` fills both (C1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct LengthBound {
    /// Optimistic lower bound on plies until some terminal (geom / exact DTM).
    pub min_plies: Option<u32>,
    /// Upper bound when a forced end is known; `None` = unknown.
    pub max_plies: Option<u32>,
}

impl LengthBound {
    /// Board-only optimistic min from pure race geometry (no walls cooperation).
    pub fn optimistic_board(stm: usize, pawn: [usize; 2]) -> Self {
        Self {
            min_plies: Some(geometric_terminal_ply_lb(stm, pawn)),
            max_plies: None,
        }
    }

    /// Exact remaining length (forced end in exactly `plies`).
    #[inline]
    pub fn exact(plies: u32) -> Self {
        Self {
            min_plies: Some(plies),
            max_plies: Some(plies),
        }
    }

    /// Raise `min_plies` only (never invents a max).
    #[inline]
    pub fn with_min(plies: u32) -> Self {
        Self {
            min_plies: Some(plies),
            max_plies: None,
        }
    }

    /// Tighten two bounds: max of mins, min of maxes. Drops `max` if it would
    /// fall below `min` (refuse an unsound short-game dump).
    #[inline]
    pub fn merge(self, other: Self) -> Self {
        let min_plies = match (self.min_plies, other.min_plies) {
            (Some(a), Some(b)) => Some(a.max(b)),
            (a, b) => a.or(b),
        };
        let max_plies = match (self.max_plies, other.max_plies) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let max_plies = match (min_plies, max_plies) {
            (Some(lo), Some(hi)) if lo > hi => None,
            _ => max_plies,
        };
        Self {
            min_plies,
            max_plies,
        }
    }

    /// Own-move floor implied by `min_plies` (STM moves on odd remaining plies).
    pub fn min_own_moves(&self) -> u32 {
        self.min_plies.map(own_moves_lb_from_plies).unwrap_or(0)
    }

    /// Own-move ceiling from `max_plies` when a forced end is known.
    pub fn max_own_moves(&self) -> Option<u32> {
        self.max_plies.map(|p| own_moves_lb_from_plies(p).max(1))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoveBudget {
    /// Hard deadline passed to `think` (optimum × MAX_RATIO, capped).
    pub move_ms: u64,
    /// Soft target before steal room (site gross − handoff).
    pub optimum_ms: u64,
    pub remaining_ms: u64,
    pub safety_ms: u64,
    pub spendable_ms: u64,
    /// Plan/race horizon used as the spend divisor.
    pub expected_own_moves: f64,
    /// Corpus P95 remaining own moves (diagnostic / leftover guard only).
    pub p95_own_moves: f64,
    pub spend_factor: f64,
    /// Geometric earliest-terminal ply lower bound (diagnostic).
    pub geom_ply_lb: u32,
    /// Typed length bound fed into the horizon (O2b).
    pub length: LengthBound,
    pub ply: u32,
}

/// Rows a player still needs toward their goal row (jump-aware optimistic).
#[inline]
pub fn rows_to_goal(player: usize, pawn: usize) -> u32 {
    if player == 0 {
        (pawn / 9) as u32
    } else {
        (8 - pawn / 9) as u32
    }
}

#[inline]
fn own_turns_lb(rows: u32) -> u32 {
    (rows + 1) / 2
}

/// Strict optimistic lower bound on plies until some player can reach goal
/// if every future wall cooperates and jumps maximize progress.
pub fn geometric_terminal_ply_lb(stm: usize, pawn: [usize; 2]) -> u32 {
    let mut best = u32::MAX;
    for p in 0..2 {
        let rows = rows_to_goal(p, pawn[p]);
        let turns = own_turns_lb(rows);
        let ply = if turns == 0 {
            0
        } else if p == stm {
            2 * turns - 1
        } else {
            2 * turns
        };
        best = best.min(ply);
    }
    if best == u32::MAX {
        0
    } else {
        best
    }
}

/// Corpus P95 remaining own moves by current ply (60s completed games).
pub fn reserve_own_moves_p95(ply: usize) -> f64 {
    const KNOTS: [(usize, f64); 7] = [
        (0, 38.5),
        (20, 28.5),
        (40, 18.5),
        (50, 14.0),
        (60, 11.0),
        (70, 9.0),
        (80, 5.0),
    ];
    if ply >= KNOTS[KNOTS.len() - 1].0 {
        return KNOTS[KNOTS.len() - 1].1;
    }
    for i in 0..KNOTS.len() - 1 {
        let (p0, v0) = KNOTS[i];
        let (p1, v1) = KNOTS[i + 1];
        if ply >= p0 && ply <= p1 {
            if p1 == p0 {
                return v0;
            }
            let t = (ply - p0) as f64 / (p1 - p0) as f64;
            return v0 + t * (v1 - v0);
        }
    }
    KNOTS[0].1
}

fn safety_ms(remaining_ms: u64) -> u64 {
    remaining_ms.saturating_div(200).max(100).min(remaining_ms)
}

fn spend_policy(remaining_ms: u64) -> (f64, f64) {
    // Absolute bands (go rem has remaining only, not the original total).
    if remaining_ms >= 45_000 {
        (1.35, 0.20)
    } else if remaining_ms >= 15_000 {
        (1.0, 0.20)
    } else {
        (0.75, 0.10)
    }
}

/// Own-move floor implied by an optimistic remaining-ply lower bound.
/// STM moves on remaining plies 1,3,5,… so `ceil(plies/2)` own turns remain.
#[inline]
pub fn own_moves_lb_from_plies(plies: u32) -> u32 {
    if plies == 0 {
        0
    } else {
        (plies + 1) / 2
    }
}

/// Eval/PV race lower bound: `min(dist_to_win_p0, dist_to_win_p1)`.
/// Invalid / missing sides are skipped; returns 0 when neither is usable.
#[inline]
pub fn eval_race_own_moves_lb(dist0: Option<u32>, dist1: Option<u32>) -> u32 {
    const INF: u32 = u32::MAX;
    let d0 = dist0.filter(|&d| d > 0 && d < INF).unwrap_or(INF);
    let d1 = dist1.filter(|&d| d > 0 && d < INF).unwrap_or(INF);
    let m = d0.min(d1);
    if m == INF {
        0
    } else {
        m
    }
}

/// Allocate one move's hard think budget from the side's remaining game clock.
///
/// Horizon = `max(plan_30 − own_moves_played, eval_race_lb)` where
/// `eval_race_lb = min(dist_to_win_p0, dist_to_win_p1)` from eval/PV.
/// Eval may only **raise** the mean-game plan — never shrink it; long games
/// rely on that extension instead of planning P95 up front.
pub fn allocate_move_budget(
    remaining_ms: u64,
    ply: usize,
    stm: usize,
    pawn: [usize; 2],
) -> MoveBudget {
    allocate_move_budget_with_dists(remaining_ms, ply, stm, pawn, None, None)
}

/// Same as [`allocate_move_budget`] with eval/PV distances for both players.
pub fn allocate_move_budget_with_leaf(
    remaining_ms: u64,
    ply: usize,
    stm: usize,
    pawn: [usize; 2],
    pv_leaf_own_moves: Option<u32>,
) -> MoveBudget {
    // Legacy single-floor API: treat as both sides sharing the same lb.
    allocate_move_budget_with_dists(
        remaining_ms,
        ply,
        stm,
        pawn,
        pv_leaf_own_moves,
        pv_leaf_own_moves,
    )
}

/// Preferred API: plan-30 plus `min(d0, d1)` from eval/PV.
pub fn allocate_move_budget_with_dists(
    remaining_ms: u64,
    ply: usize,
    stm: usize,
    pawn: [usize; 2],
    dist0: Option<u32>,
    dist1: Option<u32>,
) -> MoveBudget {
    allocate_move_budget_with_length(
        remaining_ms,
        ply,
        stm,
        pawn,
        dist0,
        dist1,
        LengthBound::optimistic_board(stm, pawn),
    )
}

/// Like [`allocate_move_budget_with_dists`], merging an oracle/certify
/// [`LengthBound`]. `max_plies` may shrink the spend horizon (forced short game);
/// `min_plies` may only raise it.
pub fn allocate_move_budget_with_length(
    remaining_ms: u64,
    ply: usize,
    stm: usize,
    pawn: [usize; 2],
    dist0: Option<u32>,
    dist1: Option<u32>,
    length_hint: LengthBound,
) -> MoveBudget {
    let geom_ply_lb = geometric_terminal_ply_lb(stm, pawn);
    let length = LengthBound::optimistic_board(stm, pawn).merge(length_hint);
    let p95 = reserve_own_moves_p95(ply);
    let own_played = (ply / 2) as f64;
    // Mean-game own-move plan (~60 plies / 2) → remaining plan tail.
    let plan_tail = (PLAN_OWN_MOVES - own_played).max(1.0);
    // Lower bound from eval: smaller distance-to-win of the two players.
    let eval_lb = eval_race_own_moves_lb(dist0, dist1) as f64;
    // LengthBound min → own-move floor (always applied; may only raise plan).
    let length_own = length.min_own_moves() as f64;
    let mut expected_own_moves = plan_tail.max(eval_lb).max(length_own).max(1.0);
    // Forced short game: do not reserve for mean-game plan beyond known end.
    if let Some(max_own) = length.max_own_moves() {
        expected_own_moves = expected_own_moves
            .min(max_own as f64)
            .max(length_own)
            .max(1.0);
    }

    let safety = if remaining_ms == 0 {
        0
    } else {
        safety_ms(remaining_ms)
    };
    let spendable = remaining_ms.saturating_sub(safety);
    let (spend_factor, share_cap) = spend_policy(remaining_ms);

    let (optimum_ms, hard_ms) = if remaining_ms == 0 || spendable == 0 {
        (1u64, 1u64)
    } else {
        let by_share = spendable as f64 * share_cap;
        let by_horizon = (spendable as f64 / expected_own_moves) * spend_factor;
        let gross = by_share.min(by_horizon);
        let handoff = (gross * 0.05).clamp(50.0, 300.0);
        let mut opt = (gross - handoff).round().max(1.0) as u64;

        // Light leftover guard from corpus P95 (not a divisor).
        let leave = (((p95 - 1.0).max(0.0)) * (MIN_MOVE_MS as f64) * 0.25).round() as u64;
        let leave_cap = spendable.saturating_sub(MIN_MOVE_MS.min(spendable));
        let leave = leave.min(leave_cap);
        opt = opt.min(spendable.saturating_sub(leave).max(1));

        let lo = MIN_MOVE_MS.min(spendable).max(1);
        opt = opt.clamp(lo, spendable);

        let hard = ((opt as f64) * MAX_RATIO)
            .round()
            .max(opt as f64) as u64;
        let hard = hard.clamp(opt, spendable);
        (opt, hard)
    };

    MoveBudget {
        move_ms: hard_ms,
        optimum_ms,
        remaining_ms,
        safety_ms: safety,
        spendable_ms: spendable,
        expected_own_moves,
        p95_own_moves: p95,
        spend_factor,
        geom_ply_lb,
        length,
        ply: ply as u32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn p95_knots_exact() {
        assert!((reserve_own_moves_p95(0) - 38.5).abs() < 1e-9);
        assert!((reserve_own_moves_p95(20) - 28.5).abs() < 1e-9);
        assert!((reserve_own_moves_p95(40) - 18.5).abs() < 1e-9);
        assert!((reserve_own_moves_p95(80) - 5.0).abs() < 1e-9);
        assert!((reserve_own_moves_p95(100) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn startpos_budget_near_site_band() {
        // Start: P0=76 (row 8), P1=4 (row 0) → 8 rows each → geom ply lb 7.
        assert_eq!(geometric_terminal_ply_lb(0, [76, 4]), 7);
        let b = allocate_move_budget(60_000, 0, 0, [76, 4]);
        assert_eq!(b.geom_ply_lb, 7);
        // Plan-30 (= mean ~60 plies / 2), not corpus P95 38.5.
        assert!((b.expected_own_moves - 30.0).abs() < 1e-9);
        assert!((b.p95_own_moves - 38.5).abs() < 1e-9);
        assert!((b.spend_factor - 1.35).abs() < 1e-9);
        // Plan-30 optimum ~2.2–3.2s; hard = optimum × 1.25.
        assert!(
            b.optimum_ms >= 2_200 && b.optimum_ms <= 3_200,
            "optimum_ms={}",
            b.optimum_ms
        );
        assert!(
            b.move_ms >= 2_700 && b.move_ms <= 4_000,
            "move_ms={}",
            b.move_ms
        );
        assert!(b.move_ms >= b.optimum_ms);
        assert!(b.move_ms <= b.spendable_ms);
        // Must not regress to TM2 ~1551.
        assert!(b.move_ms > 2_000, "TM2-style starvation: {}", b.move_ms);
    }

    #[test]
    fn never_exceeds_spendable() {
        for rem in [1u64, 50, 100, 500, 5_000, 60_000] {
            let b = allocate_move_budget(rem, 10, 0, [76, 4]);
            assert!(b.move_ms <= b.spendable_ms.max(1), "{b:?}");
            assert!(b.optimum_ms <= b.move_ms, "{b:?}");
        }
    }

    #[test]
    fn length_bound_raises_late_game_horizon() {
        let lb = LengthBound::optimistic_board(0, [76, 4]);
        assert_eq!(lb.min_plies, Some(7));
        assert!(lb.max_plies.is_none());
        assert_eq!(lb.min_own_moves(), 4);
        // Late ply: plan_tail=1, length own=4 raises expected.
        let b = allocate_move_budget(30_000, 80, 0, [76, 4]);
        assert_eq!(b.length.min_plies, Some(b.geom_ply_lb));
        assert!(b.expected_own_moves >= 4.0, "{b:?}");
    }

    #[test]
    fn eval_min_dist_raises_plan_never_shrinks() {
        let base = allocate_move_budget(60_000, 0, 0, [76, 4]);
        assert!((base.expected_own_moves - 30.0).abs() < 1e-9);

        // min(8, 12) = 8 < 30 → stay on mean-game plan.
        let below = allocate_move_budget_with_dists(
            60_000, 0, 0, [76, 4], Some(8), Some(12),
        );
        assert!((below.expected_own_moves - 30.0).abs() < 1e-9);

        // min(40, 55) = 40 raises above plan-30 (long game via extension).
        let raised = allocate_move_budget_with_dists(
            60_000, 0, 0, [76, 4], Some(40), Some(55),
        );
        assert!((raised.expected_own_moves - 40.0).abs() < 1e-9);
        assert!(raised.optimum_ms <= base.optimum_ms);

        // Only one side: that value is the lb.
        let one = allocate_move_budget_with_dists(
            60_000, 0, 0, [76, 4], Some(45), None,
        );
        assert!((one.expected_own_moves - 45.0).abs() < 1e-9);
    }

    #[test]
    fn exact_max_plies_shrinks_horizon() {
        let base = allocate_move_budget(60_000, 0, 0, [76, 4]);
        assert!((base.expected_own_moves - 30.0).abs() < 1e-9);

        // Startpos geom min=7; exact(12) tightens to min=max=12 → 6 own moves.
        let short = allocate_move_budget_with_length(
            60_000,
            0,
            0,
            [76, 4],
            None,
            None,
            LengthBound::exact(12),
        );
        assert_eq!(short.length, LengthBound::exact(12));
        assert!((short.expected_own_moves - 6.0).abs() < 1e-9);
        assert!(short.optimum_ms > base.optimum_ms);

        // max < geom min → drop max (refuse unsound short-game dump).
        let bad = allocate_move_budget_with_length(
            60_000,
            0,
            0,
            [76, 4],
            None,
            None,
            LengthBound {
                min_plies: None,
                max_plies: Some(3),
            },
        );
        assert!(bad.length.max_plies.is_none());
        assert!((bad.expected_own_moves - 30.0).abs() < 1e-9);
    }

    #[test]
    fn length_bound_merge_tightens() {
        let a = LengthBound::with_min(7).merge(LengthBound::exact(12));
        // exact(12) min=12 beats with_min(7); max=12 kept.
        assert_eq!(a, LengthBound::exact(12));
        let b = LengthBound::exact(8).merge(LengthBound {
            min_plies: Some(3),
            max_plies: Some(20),
        });
        assert_eq!(b.min_plies, Some(8));
        assert_eq!(b.max_plies, Some(8));
    }
}
