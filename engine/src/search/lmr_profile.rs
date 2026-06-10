//! Adaptive LMR profile and mate-zone controller for iterative deepening.

use crate::core::board::Board;
use crate::search::pipeline::walls_placed;

/// Absolute cm floor for “cold” extra reduction (fringe walls / off-corridor).
pub const COLD_CM_OPENING: u16 = 90;
pub const COLD_CM_MID: u16 = 55;
/// Top heat fraction at this node that skips LMR (higher = only hottest moves stay full-depth).
pub const HOT_RATIO_OPENING_PCT: u16 = 97;
pub const HOT_RATIO_MID_PCT: u16 = 84;

pub const MATE_REFINE_SLACK: u32 = 4;
pub const MATE_SPIN_MAX_MARGINAL_NODES: u64 = 15_000;
pub const MATE_MAX_TRUSTED_DIST: u32 = 64;

/// Non-mate: stop ID when root score is flat for several depths (ply37 d53 spin case).
pub const EVAL_SPIN_STABLE_ITERS: u32 = 3;
/// Centipawns (×100 cm) — max root-score change to count as "stable".
pub const EVAL_STABLE_SCORE_DELTA: i32 = 200;

/// Reference think budget for neutral LMR (10s/move in UI/benchmarks).
pub const TIME_REFERENCE_MS: u64 = 10_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MateStopReason {
    RefineCeiling,
    MateSpin,
    ForcedOutcome,
}

#[derive(Debug, Clone, Copy)]
pub struct LmrProfile {
    pub stage_t: f32,
    pub aggression: f32,
    pub lmr_after_move: usize,
    pub cat_heat_lmr_slope: f32,
    /// Skip LMR when `move_heat * 100 >= node_max_heat * hot_ratio_pct`.
    pub hot_ratio_pct: u16,
    pub cold_cm: u16,
    pub depth_balance_floor: u32,
    /// Marginal-node ceiling for depth-push feedback (opening layers are expensive).
    pub depth_push_marginal_cap: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct MateZoneState {
    pub last_mate_dist: Option<u32>,
    pub stable_iters: u32,
    pub refine_ceiling: Option<u32>,
}

/// Detects eval-flat ID spin outside mate scores (benchmark ply38 d53 @ -3.63).
#[derive(Debug, Clone, Copy, Default)]
pub struct EvalZoneState {
    pub last_score: Option<i32>,
    pub stable_iters: u32,
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t.clamp(0.0, 1.0)
}

fn compute_gates(stage_t: f32) -> (u16, u16, u64) {
    let hot_ratio = lerp(
        HOT_RATIO_OPENING_PCT as f32,
        HOT_RATIO_MID_PCT as f32,
        stage_t,
    ) as u16;
    let cold = lerp(COLD_CM_OPENING as f32, COLD_CM_MID as f32, stage_t) as u16;
    let push_cap = lerp(400_000.0, 8_000.0, stage_t) as u64;
    (hot_ratio, cold, push_cap)
}

/// Default aggression — gentle LMR, fuller tree (legacy baseline ≈1.0).
fn aggression_default() -> f32 {
    1.0
}

/// Push ID depth when eval is stable and iterations are cheap (opening prep).
fn aggression_depth_push() -> f32 {
    1.45
}

/// Empty / early opening — maximize ID depth; CAT ranks cold walls for extra cut.
fn aggression_opening_max() -> f32 {
    2.5
}

/// Endgame pawn race — narrow tree, chase forcing lines.
fn aggression_endgame_race() -> f32 {
    1.6
}

/// Tactical / eval-volatile — widen search, sacrifice ID depth for move quality.
fn aggression_tactical_wide() -> f32 {
    0.82
}

impl LmrProfile {
    /// Depth-first default — opening (low `stage_t`) is the most LMR-aggressive phase.
    pub fn depth_first_default(stage_t: f32) -> Self {
        let (hot_ratio, cold, push_cap) = compute_gates(stage_t);
        let open_blend = (1.0 - stage_t / 0.35).clamp(0.0, 1.0);
        let aggression = lerp(
            aggression_opening_max(),
            aggression_default(),
            1.0 - open_blend,
        );
        let lmr_after = if stage_t < 0.35 { 1 } else { 4 };
        let slope = lerp(0.052, 0.014, stage_t);
        Self {
            stage_t,
            aggression,
            lmr_after_move: lmr_after,
            cat_heat_lmr_slope: slope,
            hot_ratio_pct: hot_ratio,
            cold_cm: cold,
            depth_balance_floor: if stage_t < 0.25 { 56 } else { 40 },
            depth_push_marginal_cap: push_cap,
        }
    }

    pub fn first_iteration() -> Self {
        Self::depth_first_default(0.0)
    }

    /// Reproduces legacy static LMR when stage is neutral.
    pub fn baseline() -> Self {
        let (hot_ratio, cold, push_cap) = compute_gates(0.5);
        Self {
            stage_t: 0.5,
            aggression: 1.0,
            lmr_after_move: 4,
            cat_heat_lmr_slope: 0.015,
            hot_ratio_pct: hot_ratio,
            cold_cm: cold,
            depth_balance_floor: 70,
            depth_push_marginal_cap: push_cap,
        }
    }

    pub fn mate_refine() -> Self {
        let (hot_ratio, cold, _) = compute_gates(0.5);
        Self {
            stage_t: 0.5,
            aggression: 0.85,
            lmr_after_move: 8,
            cat_heat_lmr_slope: 0.005,
            hot_ratio_pct: hot_ratio,
            cold_cm: cold,
            depth_balance_floor: 0,
            depth_push_marginal_cap: 8_000,
        }
    }

    pub fn from_stage(stage_t: f32, endgame_race: bool, mate_refine: bool) -> Self {
        if mate_refine {
            return Self::mate_refine();
        }
        let mut p = Self::depth_first_default(stage_t);
        if endgame_race {
            p.aggression = aggression_endgame_race();
            p.lmr_after_move = 3;
        } else if stage_t >= 0.40 {
            // Complex middlegame — slightly wider by default.
            p.lmr_after_move = 5;
            p.cat_heat_lmr_slope = 0.008;
        }
        p
    }

    /// Widen LMR — accuracy over depth (tactical mess, eval swinging).
    pub fn apply_tactical_wide(&mut self) {
        self.aggression = aggression_tactical_wide();
        self.lmr_after_move = self.lmr_after_move.max(7).min(8);
        self.cat_heat_lmr_slope *= 0.85;
    }

    /// Narrow LMR — push ID depth when iterations are cheap and eval is stable.
    pub fn apply_depth_push(&mut self) {
        self.aggression = (self.aggression * 1.08)
            .max(aggression_depth_push())
            .min(2.6);
        self.lmr_after_move = self.lmr_after_move.saturating_sub(1).max(2);
        self.cat_heat_lmr_slope = (self.cat_heat_lmr_slope * 1.10).min(0.060);
    }

    /// Scale LMR for per-move think budget — less time → more pruning, chase depth.
    pub fn apply_time_budget(&mut self, time_ms: u64) {
        let t = time_pressure_from_ms(time_ms);
        if t < 0.02 {
            return;
        }
        let mul = 1.0 + t * 0.55;
        self.aggression = (self.aggression * mul).min(3.4);
        self.hot_ratio_pct = (self.hot_ratio_pct as f32 + t * 5.0).min(99.0) as u16;
        self.cold_cm = (self.cold_cm as f32 + t * 30.0).min(150.0) as u16;
        let cut = ((t * 2.5) as usize).min(self.lmr_after_move.saturating_sub(1));
        self.lmr_after_move = self.lmr_after_move.saturating_sub(cut).max(1);
        self.cat_heat_lmr_slope = (self.cat_heat_lmr_slope * (1.0 + t * 0.40)).min(0.090);
        self.depth_push_marginal_cap =
            ((self.depth_push_marginal_cap as f32) * (1.0 - t * 0.50).max(0.15)) as u64;
        self.depth_balance_floor =
            ((self.depth_balance_floor as f32) * (1.0 - t * 0.25)).max(24.0) as u32;
    }

    /// Extra pruning when the clock is running low within a move.
    pub fn apply_time_urgency(&mut self, fraction_elapsed: f32, time_ms: u64) {
        let base = time_pressure_from_ms(time_ms);
        let urgency = ((fraction_elapsed - 0.55) / 0.45).clamp(0.0, 1.0);
        let t = (base + urgency * (1.0 - base * 0.5)).clamp(0.0, 1.0);
        let extra = (t - base).max(0.0);
        if extra < 0.04 {
            return;
        }
        self.aggression = (self.aggression * (1.0 + extra * 0.35)).min(3.6);
        self.hot_ratio_pct = (self.hot_ratio_pct as f32 + extra * 3.0).min(99.0) as u16;
        self.cold_cm = (self.cold_cm as f32 + extra * 12.0).min(160.0) as u16;
    }
}

/// 0 = full 10s budget, 1 = severe crunch (~2s). Linear in budget fraction —
/// at 8s/10s pressure≈0.2, at 5s≈0.5 (depth-first under time handicap).
pub fn time_pressure_from_ms(time_ms: u64) -> f32 {
    let frac = (time_ms as f32 / TIME_REFERENCE_MS as f32).clamp(0.15, 1.0);
    (1.0 - frac).clamp(0.0, 1.0)
}

pub fn compute_stage_t(
    board: &Board,
    our_dist: u8,
    opp_dist: u8,
    root_cat_max: u16,
    root_cat_p75: u16,
) -> f32 {
    let walls_n = walls_placed(board) as f32 / 20.0;
    let min_dist = our_dist.min(opp_dist) as f32;
    let race_n = 1.0 - (min_dist / 16.0).clamp(0.0, 1.0);
    let spread_n = if root_cat_max > 0 {
        1.0 - (root_cat_max.saturating_sub(root_cat_p75)) as f32 / root_cat_max as f32
    } else {
        0.5
    };
    (0.35 * walls_n + 0.35 * race_n + 0.30 * spread_n).clamp(0.0, 1.0)
}

pub fn build_lmr_table(aggression: f32) -> [[u32; 64]; 64] {
    let mut table = [[0u32; 64]; 64];
    let ag = aggression as f64;
    for depth in 1usize..64 {
        for mv_count in 1usize..64 {
            let r_raw = 0.75 + (depth as f64).ln() * (mv_count as f64).ln() * (ag / 1.85);
            let cap = (depth / 2) as u32;
            let r = (r_raw.max(0.0) as u32).min(cap);
            table[depth][mv_count] = r;
        }
    }
    table
}

impl MateZoneState {
    pub fn update_after_depth(
        &mut self,
        verified: i32,
        depth: u32,
        marginal_nodes: u64,
        mate_proven_at_depth: bool,
        pv_verified: bool,
    ) -> Option<MateStopReason> {
        if !is_mate_score(verified) {
            self.last_mate_dist = None;
            self.stable_iters = 0;
            self.refine_ceiling = None;
            return None;
        }

        let Some(dist) = mate_distance(verified) else {
            return None;
        };
        if dist == 0 || dist > MATE_MAX_TRUSTED_DIST {
            self.last_mate_dist = None;
            self.stable_iters = 0;
            return None;
        }

        if mate_proven_at_depth || pv_verified {
            let ceiling = dist.saturating_add(MATE_REFINE_SLACK);
            self.refine_ceiling = Some(self.refine_ceiling.map_or(ceiling, |c| c.min(ceiling)));
        }

        if self.last_mate_dist == Some(dist) {
            self.stable_iters = self.stable_iters.saturating_add(1);
        } else {
            self.last_mate_dist = Some(dist);
            self.stable_iters = 1;
        }

        if let Some(ceiling) = self.refine_ceiling {
            if depth >= ceiling {
                return Some(MateStopReason::RefineCeiling);
            }
        } else if depth >= dist {
            self.refine_ceiling = Some(dist.saturating_add(MATE_REFINE_SLACK));
            if depth >= dist.saturating_add(MATE_REFINE_SLACK) {
                return Some(MateStopReason::RefineCeiling);
            }
        }

        if depth >= dist.saturating_add(MATE_REFINE_SLACK) {
            return Some(MateStopReason::RefineCeiling);
        }

        if self.stable_iters >= 2 && marginal_nodes < MATE_SPIN_MAX_MARGINAL_NODES && depth >= dist
        {
            return Some(MateStopReason::MateSpin);
        }

        None
    }
}

pub fn apply_depth_feedback(
    profile: &mut LmrProfile,
    completed_depth: u32,
    marginal_nodes: u64,
    prev_marginal_nodes: u64,
    fraction_elapsed: f32,
    score_delta: i32,
    aspiration_fails_delta: u32,
) {
    let eval_volatile = score_delta.abs() > EVAL_STABLE_SCORE_DELTA;
    // Opening pawn-PV ±1.21 oscillation is benign — don't widen LMR mid-ID.
    let tactical = profile.stage_t >= 0.38 && (eval_volatile || aspiration_fails_delta >= 2);

    if tactical {
        profile.apply_tactical_wide();
        return;
    }

    // Cheap stable iterations → push depth (opening prep / quiet positions).
    if completed_depth < profile.depth_balance_floor
        && marginal_nodes < profile.depth_push_marginal_cap
        && prev_marginal_nodes > 0
        && fraction_elapsed < 0.85
    {
        profile.apply_depth_push();
    }

    // Branching explosion — widen so next depth can finish in budget (not in open prep).
    if profile.stage_t >= 0.35
        && prev_marginal_nodes > 0
        && marginal_nodes > prev_marginal_nodes.saturating_mul(4)
    {
        profile.apply_tactical_wide();
    }
}

impl EvalZoneState {
    pub fn update_after_depth(&mut self, verified: i32, depth: u32, marginal_nodes: u64) -> bool {
        if is_mate_score(verified) {
            self.last_score = None;
            self.stable_iters = 0;
            return false;
        }

        if let Some(prev) = self.last_score {
            if (verified - prev).abs() <= EVAL_STABLE_SCORE_DELTA {
                self.stable_iters = self.stable_iters.saturating_add(1);
            } else {
                self.stable_iters = 1;
            }
        } else {
            self.stable_iters = 1;
        }
        self.last_score = Some(verified);

        let _ = marginal_nodes;
        self.stable_iters >= EVAL_SPIN_STABLE_ITERS && depth >= 12
    }
}

// Mirror mate helpers used in alphabeta (avoid circular deps).
const MATE: i32 = 20_000;
const MATE_WINDOW: i32 = 500;

fn is_mate_score(score: i32) -> bool {
    score > MATE - MATE_WINDOW || score < -MATE + MATE_WINDOW
}

fn mate_distance(score: i32) -> Option<u32> {
    if score > MATE - MATE_WINDOW {
        Some((MATE - score).max(0) as u32)
    } else if score < -MATE + MATE_WINDOW {
        Some((MATE + score).max(0) as u32)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::Board;

    #[test]
    fn race_n_long_race_gives_lower_t_than_short_race() {
        let board = Board::new();
        let long_race = compute_stage_t(&board, 12, 12, 200, 80);
        let short_race = compute_stage_t(&board, 4, 4, 200, 80);
        assert!(
            long_race < short_race,
            "long={long_race} short={short_race}"
        );
    }

    #[test]
    fn time_pressure_increases_with_shorter_budget() {
        let p10 = time_pressure_from_ms(10_000);
        let p8 = time_pressure_from_ms(8_000);
        let p3 = time_pressure_from_ms(3_000);
        assert!(p10 < p8, "10s={p10} 8s={p8}");
        assert!(p8 < p3, "8s={p8} 3s={p3}");
        assert!((p8 - 0.2).abs() < 0.01, "8s pressure ~0.2 got {p8}");
    }

    #[test]
    fn apply_time_budget_raises_aggression() {
        let mut open = LmrProfile::depth_first_default(0.0);
        let base = open.aggression;
        open.apply_time_budget(1_000);
        assert!(
            open.aggression > base,
            "base={base} after={}",
            open.aggression
        );
        assert!(open.hot_ratio_pct > HOT_RATIO_OPENING_PCT);
    }

    #[test]
    fn spread_n_zero_cat_max_uses_neutral_guard() {
        let board = Board::new();
        let t = compute_stage_t(&board, 8, 8, 0, 0);
        assert!((0.0..=1.0).contains(&t));
        let flat = compute_stage_t(&board, 8, 8, 200, 200);
        assert!(
            flat > t,
            "flat heat spread should increase t; guard={t} flat={flat}"
        );
    }

    #[test]
    fn baseline_profile_near_legacy() {
        let p = LmrProfile::baseline();
        assert!((p.aggression - 1.0).abs() < 0.01);
    }

    #[test]
    fn opening_profile_is_most_aggressive() {
        let open = LmrProfile::depth_first_default(0.0);
        let mid = LmrProfile::depth_first_default(0.5);
        assert!(
            open.aggression > mid.aggression,
            "opening aggression {} should exceed mid {}",
            open.aggression,
            mid.aggression
        );
        assert!(open.aggression >= aggression_opening_max() - 0.01);
        assert_eq!(open.lmr_after_move, 1);
        assert!(open.cat_heat_lmr_slope > mid.cat_heat_lmr_slope);
        assert!(open.depth_push_marginal_cap > 100_000);
    }

    #[test]
    fn eval_zone_stops_flat_eval_spin() {
        let mut zone = EvalZoneState::default();
        let score = -363;
        let mut stopped = false;
        for depth in 1..=40 {
            if zone.update_after_depth(score, depth, 5_000) {
                stopped = true;
                assert!(depth >= 12);
                break;
            }
        }
        assert!(stopped);
    }

    #[test]
    fn eval_zone_stops_even_when_depth_is_expensive() {
        let mut zone = EvalZoneState::default();
        let score = -169;
        let mut stopped = false;
        for depth in 1..=40 {
            if zone.update_after_depth(score, depth, 800_000) {
                stopped = true;
                assert!(depth >= 12);
                break;
            }
        }
        assert!(
            stopped,
            "stable eval should stop ID even when each depth is costly"
        );
    }

    #[test]
    fn mate_zone_stops_at_dist_plus_slack() {
        let mut zone = MateZoneState::default();
        let score = -(MATE - 8);
        let mut stopped_at = None;
        for depth in 1..=20 {
            if zone
                .update_after_depth(score, depth, 20_000, depth >= 8, false)
                .is_some()
            {
                stopped_at = Some(depth);
                break;
            }
        }
        assert_eq!(stopped_at, Some(12));
    }
}
