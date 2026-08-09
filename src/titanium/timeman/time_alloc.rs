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


// ── Value-of-thinking planner ────────────────────────────────────────────────
//
// The clock is split in proportion to how much a longer search can still change
// the outcome, rather than evenly. Every term is measured on the gate corpus,
// and weights are normalised against the projected future so the shares sum to
// ~1. An earlier attempt multiplied the early game by 1.25 with nothing
// compensating, which is just overspending, and lost 11-27.

/// Moves at the end of a game the certificate answers outright. Measured after
/// the solved-position fix: plies 46+ cost 0-1ms per move however much they are
/// offered, so they need a token reserve rather than a proportional one. The
/// difference is budget the middlegame keeps.
const TAIL_MOVES: f64 = 5.0;
const TAIL_MS_PER_MOVE: f64 = 100.0;

/// Walls still on the board at `ply`, fitted to the corpus (mean walls left at
/// ply 0/8/16/24/32/40/48 = 19.9/17.0/11.1/7.0/4.9/3.4/1.9).
#[inline]
fn projected_walls(ply: usize) -> f64 {
    20.0 * (-(ply as f64) / 22.0).exp()
}

/// How much a longer search can still change this move. A pure function of the
/// board, so the schedule stays predictable and testable.
///
/// Measured P(the search changed its best move):
///   walls left 0-3 -> 23%, 4-7 -> 36%, 8-11 -> 45%, 12-15 -> 58%, 16+ -> 57%
///   nearer pawn within 2 of goal -> 13.5%, versus ~42% mid-board
///   by ply: 1.00 at 6-12, 0.75 by 18, 0.55 by 42, 0.12 by 60, 0.01 by 66
///
/// Walls dominate by 2.5x: they are the branching resource, and a wall-less
/// position is a deterministic race where extra nodes buy nothing.
fn think_value(ply: usize, walls_total: f64, leader_dist: Option<u32>) -> f64 {
    // Opening is booked or near-forced, so it sits below the curve despite a
    // high change-rate: those changes are cheap to find.
    let ply_term = if ply < 8 {
        0.25
    } else if ply < 20 {
        1.00
    } else if ply < 30 {
        0.90
    } else if ply < 40 {
        0.70
    } else if ply < 50 {
        0.40
    } else {
        0.15
    };
    let wall_term = 0.40 + 0.60 * (walls_total.max(0.0) / 12.0).min(1.0);
    let lead_term = match leader_dist {
        Some(d) if d <= 2 => 0.35,
        Some(d) if d <= 4 => 0.70,
        _ => 1.0,
    };
    ply_term * wall_term * lead_term
}

/// Sum of `think_value` over the moves still to be played, walking the board
/// forward. This denominator is what makes the schedule a true split of the
/// clock rather than a multiplier on it.
fn future_think_value(ply: usize, own_moves_left: f64, leader_dist: Option<u32>) -> f64 {
    let n = own_moves_left.ceil().max(1.0) as usize;
    (0..n)
        .map(|k| {
            let p = ply + 2 * k;
            // The leader closes on its goal as the game runs on, so each future
            // move is weighted for the position it will actually be played in.
            let d = leader_dist.map(|d| d.saturating_sub(k as u32));
            think_value(p, projected_walls(p), d)
        })
        .sum()
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

/// Allocate with current wall counts, which the value planner needs.
pub fn allocate_move_budget_with_dists_and_walls(
    remaining_ms: u64,
    ply: usize,
    stm: usize,
    pawn: [usize; 2],
    dist0: Option<u32>,
    dist1: Option<u32>,
    walls_remaining: [i32; 2],
) -> MoveBudget {
    allocate_move_budget_with_length_and_walls(
        remaining_ms, ply, stm, pawn, dist0, dist1,
        LengthBound::optimistic_board(stm, pawn), walls_remaining,
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
    // Unknown walls: assume a full board rather than an empty one, so callers
    // that do not pass walls are not silently starved by the planner.
    allocate_move_budget_with_length_and_walls(
        remaining_ms, ply, stm, pawn, dist0, dist1, length_hint, [10, 10],
    )
}

fn allocate_move_budget_with_length_and_walls(
    remaining_ms: u64,
    ply: usize,
    stm: usize,
    pawn: [usize; 2],
    dist0: Option<u32>,
    dist1: Option<u32>,
    length_hint: LengthBound,
    walls_remaining: [i32; 2],
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
        // Reserve only a token amount for the moves the certificate answers
        // outright; they measure 0-1ms however much they are given, so
        // reserving for them at the mean rate was money left on the table.
        let tail_reserve = (TAIL_MOVES.min(expected_own_moves) * TAIL_MS_PER_MOVE) as u64;
        let planned = spendable.saturating_sub(tail_reserve).max(MIN_MOVE_MS) as f64;

        // Split the plannable clock by how much this move can still change,
        // against every move still to come. Shares sum to ~1, so weighting the
        // early game necessarily takes the time from later rather than
        // inventing it.
        let leader = match (dist0, dist1) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (a, b) => a.or(b),
        };
        let walls_now = (walls_remaining[0] + walls_remaining[1]).clamp(0, 20) as f64;
        let value_now = think_value(ply, walls_now, leader);
        let value_future = future_think_value(ply, expected_own_moves, leader).max(1e-9);
        let share = (value_now / value_future).clamp(0.0, 1.0);

        // spend_factor must NOT scale a normalised share: the shares already
        // sum to ~1, so multiplying by 1.35 is a 35% across-the-board
        // overspend -- precisely the mistake that made the old phase
        // multiplier lose 11-27. It survives only as a low-clock brake:
        // full share above 45s, progressively cautious below.
        let caution = spend_factor / 1.35;
        let by_share = spendable as f64 * share_cap;
        let by_plan = planned * share * caution;
        let gross = by_share.min(by_plan);
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

    /// Budget for a 3min+2s game at `ply`, with given walls and leader distance.
    fn plan_at(rem_ms: u64, ply: usize, walls: [i32; 2], lead: Option<u32>) -> u64 {
        allocate_move_budget_with_dists_and_walls(
            rem_ms, ply, ply % 2, [76, 4], lead, lead, walls,
        )
        .optimum_ms
    }

    #[test]
    fn opening_is_cheaper_than_the_decision_zone() {
        // Opening play is booked or near-forced, so it sits below the curve
        // even though its move-change rate is high: those changes are cheap to
        // find. No magic band here -- the number is re-tuned deliberately, and
        // pinning it only produces false failures.
        let open = plan_at(60_000, 4, [10, 10], None);
        let peak = plan_at(60_000, 14, [10, 10], None);
        assert!(peak > open * 2, "opening {open} vs peak {peak}");
        assert!(open > MIN_MOVE_MS * 4, "opening starved: {open}");
    }

    #[test]
    fn fewer_walls_buy_less_time() {
        // Measured: P(the search changed its move) is 23% with 0-3 walls left
        // against 58% with 12-15. A wall-less race is deterministic and extra
        // nodes buy almost nothing.
        let rich = plan_at(60_000, 24, [10, 10], None);
        let poor = plan_at(60_000, 24, [1, 0], None);
        assert!(poor < rich, "wall-less {poor} should undercut wall-rich {rich}");
    }

    #[test]
    fn a_leader_at_the_goal_line_is_cheap() {
        // 13.5% move-change rate once the nearer pawn is within 2 of home,
        // against ~42% mid-board: nothing left to decide.
        let far = plan_at(60_000, 30, [5, 5], Some(9));
        let near = plan_at(60_000, 30, [5, 5], Some(1));
        assert!(near < far, "near-goal {near} should undercut mid-board {far}");
    }

    #[test]
    fn the_schedule_spends_the_clock_without_hoarding_it() {
        // Walk a whole game: shares sum to ~1, so the clock should be mostly
        // consumed and the decision zone should dominate. The old fixed divisor
        // banked 21-31% unspent.
        let mut rem: u64 = 60_000;
        let mut spend = [0u64; 4]; // <8, 8-30, 30-46, 46+
        for ply in (0..56).step_by(2) {
            let walls = (20.0 * (-(ply as f64) / 22.0).exp()).round() as i32;
            let b = plan_at(rem, ply, [walls / 2, walls - walls / 2], None);
            // Past the decision horizon the certificate answers instantly.
            let used = if ply < 46 { b } else { b / 20 };
            let bucket = match ply {
                0..=7 => 0,
                8..=29 => 1,
                30..=45 => 2,
                _ => 3,
            };
            spend[bucket] += used;
            rem = rem.saturating_sub(used);
        }
        let total: u64 = spend.iter().sum();
        let pct = |i: usize| 100 * spend[i] / total.max(1);
        assert!(pct(1) >= 45, "decision zone underfunded: {:?} -> {}%", spend, pct(1));
        assert!(pct(0) <= 20, "opening overfunded: {}%", pct(0));
        assert!(rem < 20_000, "hoarded {rem}ms of a 60s clock");
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
