//! Titanium αβ search with iterative deepening, transposition tables, LMR/EME,
//! RaceProof, repetition-safe TT handling, and the supporting move-ordering,
//! evaluation, and timing heuristics used by the native engine.

use crate::titanium::dist::{
    fill_ace_dist_from_pawn, fill_ace_dist_layers_to_goal_p0, fill_ace_dist_layers_to_goal_p1,
    fill_ace_dist_to_goal_with_masks_p0, fill_ace_dist_to_goal_with_masks_p1, fill_choke_points,
    fill_contested, fill_corridor_delta, fill_sparse_route_masks, materialize_distance_layers,
    shortest_route_bits, wall_incr_refresh_flags, width_in_layers,
};
use crate::titanium::{
    is_hwall_move, is_pawn_move, is_vwall_move, is_wall_move, move_id_to_board, wall_slot,
    MOVE_HW_BASE, MOVE_VW_BASE,
};
use crate::util::clock::{Duration, Instant};

use super::cat_index_lmr::apply_lmr_path_correction;
use super::v16_lmr::{
    plan_v16_pawn_lmr, plan_v16_wall_lmr, V16HardOverride, ACE_LMR_AFTER_MOVE, ACE_LMR_MIN_DEPTH,
};
use crate::cat::prune::{
    cat_v16_lmr_fringe_pct_for_worker, gap_play_zone_mask, get_shortest_path,
    move_corridor_attention_with_denial, move_corridor_attention_with_path, move_impact_heat,
    wall_in_dead_zone, wall_should_search,
};
use crate::cat::CorridorAttention;
use crate::core::board::{Board, Move as BoardMove, Player, Undo, WallOrientation};
use crate::movegen::{
    generate_legal_moves_slice_cached, GeometricWallCache, GeometricWallCacheStats, MAX_LEGAL_MOVES,
};
use crate::pathfinding::bff::expand_frontier;
use crate::pathfinding::masks::DirMasks;
use crate::pathfinding::BfsScratch;
use crate::titanium::certify::{certify, CertifyOpts};
use crate::titanium::game::{GameState, ZOBRIST};
use crate::titanium::net::{net, net_frozen, Net, MAX_NET_H, NET_BKT, NET_MIRC, NET_MIRS};
use crate::titanium::packed_state::FEATURE_SCHEMA;
use crate::titanium::race::{
    bff_tempo_margin_close, jump_aware_goal_distances, race_outcome_detailed,
    race_outcome_with_dist, solve_race_config, PlyEstimate, RaceBound, RaceOutcomeStats,
    RaceScratch, RACE_MATE, RACE_STATES, RACE_WIN_FLOOR,
};
use crate::titanium::reduction_sidecar::ReductionSidecar;
use crate::util::grid::FLOOD_PLAYABLE;
use std::collections::HashMap;
#[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
use std::sync::Mutex;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use std::sync::{
    atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering},
    Arc, RwLock,
};
pub const MATE: i32 = 100_000;
pub const MAX_PLY: usize = 64;
const INF: i32 = 2 * MATE;
const HIST_SPAN: usize = 256;
/// TT search-depth field width: 8 bits → physical max 255.
/// Game plies are a separate limit (~128 with 3-fold / match `max_plies`);
/// iterative deepening is also clamped to this.
pub const TT_DEPTH_MAX: i32 = 255;
const TT_DEPTH_SHIFT: i32 = 12;
const TT_DEPTH_MASK: i32 = 0xFF;

#[inline]
fn tt_pack_depth(depth: i32) -> i32 {
    depth.clamp(0, TT_DEPTH_MAX) << TT_DEPTH_SHIFT
}

#[inline]
fn tt_unpack_depth(meta: i32) -> i32 {
    (meta >> TT_DEPTH_SHIFT) & TT_DEPTH_MASK
}

#[inline]
fn reverse_futility_margin(
    depth: i32,
    improving: bool,
    ace_rfp: bool,
    ace_rfp_max_depth: i32,
) -> Option<i32> {
    if ace_rfp {
        (depth <= ace_rfp_max_depth).then_some(100 * depth)
    } else {
        (depth <= 4).then_some((if improving { 70 } else { 90 }) * depth)
    }
}

#[inline]
fn rfp_depth_for_budget(tc_adaptive: bool, allotted_ms: u64) -> i32 {
    if tc_adaptive && allotted_ms <= 200 {
        4
    } else {
        3
    }
}

/// Default CAT-index LMR tuning percent:
/// -500 = strongest CAT-shaped cuts, 100 = current/default, 150 = full depth.
pub const CAT_LMR_DEFAULT_TUNING_PERCENT: i32 = -177;

fn cat_lmr_tuning_percent() -> i32 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        if let Ok(raw) = std::env::var("TITANIUM_CAT_LMR_TUNING_PERCENT") {
            if let Ok(value) = raw.parse::<i32>() {
                return value.clamp(-500, 150);
            }
        }
    }
    CAT_LMR_DEFAULT_TUNING_PERCENT
}

/// Late-move reduction plies — re-exported for LMR vision (`legacy_search::lmr_viz`).
pub use super::v16_lmr::ace_graduated_lmr_reduction;

/// EME extends only the first ordered wall moves after the TT/best move.
/// Index 0 (TT move) already gets full depth; extending more siblings
/// compounds multiplicatively down the tree and explodes the node count.
const ACE_EME_TOP_MOVES: usize = 2;

/// Flat move-ordering bonus for a wall touching either player's shortest-
/// route cell set (see `route_touch_ordering`). Small relative to typical
/// CAT impact-heat magnitudes -- a cold-start nudge among otherwise-similar
/// moves, not a signal meant to override a strong CAT read or distort
/// iterative deepening.
const ROUTE_TOUCH_ORDER_BONUS: i32 = 20;

/// Default quiescence extension cap (Ka-AB `qMax`).
const Q_SEARCH_MAX_DEFAULT: i32 = 4;
/// Static-eval swing threshold in cp (Ka `qSwing=0.15` on a ~400cp net scale).
const Q_SWING_CP_DEFAULT: i32 = 60;
/// saturated wall at ~1.05M ordering score — the same neighborhood the legacy
/// counter reaches on hot walls (and just above pawn-progress scores), so the
/// relative ordering bands match the legacy mode and the A/B isolates the
/// side-split/gravity/malus semantics, not a rescale of everything.
const SF_HIST_MAX: i32 = 1 << 20;

/// Pawn progress changes ordering by 1000 points for each shortest-path step.
/// Keep the optional history tie-break strictly inside half that gap: even the
/// most-disfavoured one-step-faster pawn move still outranks the most-favoured
/// one-step-slower move (a minimum two-point margin).
const SF_PAWN_HISTORY_TIEBREAK_MAX: i32 = 499;

/// Compress the saturated SF history range into a pawn-only ordering tie-break.
/// The clamp also makes the bound hold if a caller seeds a table entry directly.
#[inline]
fn sf_pawn_history_tiebreak(history: i32) -> i32 {
    let bounded = history.clamp(-SF_HIST_MAX, SF_HIST_MAX) as i64;
    (bounded * SF_PAWN_HISTORY_TIEBREAK_MAX as i64 / SF_HIST_MAX as i64) as i32
}

/// Correction-history table size per side (wall-structure-hash buckets).
const CORR_SIZE: usize = 1 << 14;
/// Correction magnitude clamp (cp). Net eval band is ±2000; a correction
/// beyond ±256 means the static eval is systematically wrong by more than a
/// wall's worth — cap it rather than let one bucket dominate.
const CORR_MAX: i32 = 256;

/// Opt-in ProbCut is deliberately narrow: static eval must already exceed the
/// null-window beta by this many centipawns before it may spend a shallow
/// verification search. A 200cp margin keeps this experimental cutoff well
/// away from ordinary evaluation noise.
const PROBCUT_STATIC_MARGIN: i32 = 200;
/// The verification search is four plies shallower than the full node. This
/// leaves at least two plies at the minimum eligible depth (six).
const PROBCUT_REDUCTION: i32 = 4;
const PROBCUT_MIN_DEPTH: i32 = 6;
/// ProbCut is only meaningful for ordinary net-evaluation windows. Keeping it
/// inside this band excludes mate, race, and certificate score conventions.
const PROBCUT_MAX_ABS_BETA: i32 = 2_000;

/// Pure eligibility gate for the opt-in ProbCut experiment. `ply > 0` keeps
/// root search exact, the one-point window excludes PV nodes, and `allow_null`
/// prevents the verification itself from recursively launching another
/// speculative cutoff search.
#[inline]
fn probcut_is_eligible(
    depth: i32,
    alpha: i32,
    beta: i32,
    ply: usize,
    allow_null: bool,
    static_ev: i32,
) -> bool {
    depth >= PROBCUT_MIN_DEPTH
        && ply > 0
        && allow_null
        && beta.saturating_sub(alpha) == 1
        && beta.abs() < PROBCUT_MAX_ABS_BETA
        && static_ev >= beta.saturating_add(PROBCUT_STATIC_MARGIN)
}

/// Pure predicate, factored out for direct unit testing: does this wall
/// touch a cell on either player's shortest-route set? `route0`/`route1`
/// are the 81-cell boolean masks from `fill_sparse_route_masks`.
fn wall_touches_route(
    row: u8,
    col: u8,
    orientation: crate::core::board::WallOrientation,
    route0: &[u8; 81],
    route1: &[u8; 81],
) -> bool {
    crate::util::grid::wall_touch_squares(row, col, orientation)
        .iter()
        .any(|&(r, c)| {
            let sq = (r as usize) * 9 + c as usize;
            route0[sq] != 0 || route1[sq] != 0
        })
}

/// Early Move Extension — +1 ply for the top ordered walls; +2 only for
/// the very first non-TT wall when there is real depth left to spend.
fn ace_graduated_eme_extension(move_index: usize, depth: i32) -> i32 {
    if move_index == 1 && depth >= 8 {
        2
    } else {
        1
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod lazy_smp_tests {
    use super::*;

    fn fresh() -> Box<TitaniumSearch> {
        TitaniumSearch::grafted(GameState::new(), Some(18))
    }

    #[test]
    fn root_width_calculation_uses_ceiling_and_min_one() {
        let cases = [
            (30, 100, 30),
            (30, 80, 24),
            (30, 60, 18),
            (30, 40, 12),
            (30, 20, 6),
            (1, 20, 1),
            (2, 20, 1),
            (3, 20, 1),
            (4, 20, 1),
            (5, 20, 1),
            (6, 20, 2),
            (0, 100, 0),
        ];
        for (root_count, percent, expected) in cases {
            assert_eq!(lazy_smp_allowed_root_moves(root_count, percent), expected);
        }
    }

    /// `allowed_root_moves()` used to re-apply the CAT-heat threshold as a
    /// move-count percentage on top of the already-filtered `root_move_count`,
    /// so `unique.len() <= allowed` could fail even though the real search
    /// (which uses `root_move_count` directly, not `allowed_root_moves()`)
    /// never visited outside its actual retained set. This test now checks
    /// the real invariants: visits resolve to real retained move IDs, unique
    /// visit count never exceeds the real retained count, and retained sets
    /// are monotonic by threshold (a stricter worker never keeps a move a
    /// looser one dropped).
    #[test]
    fn root_filtering_limits_each_worker_to_its_width() {
        let mut search = fresh();
        let result = search.think_with_threads(1_000, 1, true, false, "titanium-v15", 5);
        assert_eq!(result.root_widths.len(), 5);
        assert_eq!(result.root_move_ids.len(), 5);

        let retained_sets: Vec<std::collections::HashSet<i16>> = result
            .root_move_ids
            .iter()
            .map(|ids| ids.iter().copied().collect())
            .collect();

        for plan in &result.root_widths {
            let visits = &result.root_visits[plan.worker_id];
            let retained = &result.root_move_ids[plan.worker_id];
            assert_eq!(retained.len(), plan.root_move_count);

            for &idx in visits {
                assert!(
                    idx < retained.len(),
                    "worker {} visited index {} outside its retained list of {}",
                    plan.worker_id,
                    idx,
                    retained.len()
                );
            }

            let unique = visits
                .iter()
                .copied()
                .collect::<std::collections::HashSet<_>>();
            assert!(
                unique.len() <= plan.root_move_count,
                "worker {} visited {} unique root moves but only retained {}",
                plan.worker_id,
                unique.len(),
                plan.root_move_count
            );

            // Real percentage, derived from real counts -- not re-derived
            // from a move-count formula applied to the heat threshold.
            let expected_pct = if plan.root_moves_before_filter == 0 {
                0.0
            } else {
                100.0 * plan.root_move_count as f64 / plan.root_moves_before_filter as f64
            };
            assert!((plan.root_moves_retained_pct() - expected_pct).abs() < 1e-9);

            if plan.worker_id == 0 {
                assert_eq!(plan.root_value_threshold_pct, 20);
            }
        }

        // Monotonic superset across the whole pool: worker 0 (20%) retains a
        // superset of worker 1 (30%), which retains a superset of every
        // worker on the 40% floor (workers 2-4 here) -- chained pairwise
        // since threshold is non-decreasing by worker_id in this schedule.
        for pair in result.root_widths.windows(2) {
            let looser = &retained_sets[pair[0].worker_id];
            let stricter = &retained_sets[pair[1].worker_id];
            if pair[0].root_value_threshold_pct <= pair[1].root_value_threshold_pct {
                assert!(
                    stricter.is_subset(looser),
                    "worker {} (threshold {}) retained a move worker {} (threshold {}) dropped",
                    pair[1].worker_id,
                    pair[1].root_value_threshold_pct,
                    pair[0].worker_id,
                    pair[0].root_value_threshold_pct
                );
            }
        }
    }

    /// Regression guard for the diagnostics-only cleanup: retained counts for
    /// this exact 20-ply position were captured via CLI trace (`titanium
    /// genmove --engine titanium-v17 --time 10 --threads 8`) as
    /// [16,14,9,9,9,9,9,9]. `root_moves_before_filter` is 79 here -- that is
    /// `root_moves_raw.len()` from `ordered_root_moves_snapshot` (already
    /// past dead-zone/CAT-corridor root pruning upstream), NOT the naive
    /// full-legal-move count (132, from plain `titanium moves`) which
    /// includes moves this cleanup never had visibility into. This cleanup
    /// touches naming, `allowed_root_moves()`, and diagnostics JSON only --
    /// `lazy_smp_value_threshold_pct` and `lazy_smp_value_filtered_moves`
    /// (the actual filtering logic) are untouched, so these counts must be
    /// identical after the cleanup.
    #[test]
    fn root_filter_retained_counts_match_pre_cleanup_baseline_on_20ply_position() {
        let moves = [
            "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e6h", "c3h", "g6h", "g3h", "c6h", "a3h",
            "e5", "h4v", "h7h", "f5v", "a6h", "e4h", "c4v",
        ];
        let mut g = GameState::new();
        for mv in moves {
            g.make_move(crate::titanium::algebraic_to_move_id(mv));
        }
        let mut search = TitaniumSearch::grafted(g, Some(18));
        let result = search.think_with_threads(1_000, 1, true, false, "titanium-v17", 8);

        assert_eq!(result.root_widths.len(), 8);
        let counts: Vec<usize> = result
            .root_widths
            .iter()
            .map(|p| p.root_move_count)
            .collect();
        assert_eq!(
            counts,
            vec![16, 14, 9, 9, 9, 9, 9, 9],
            "retained root move counts drifted from the pre-cleanup baseline"
        );
        for plan in &result.root_widths {
            assert_eq!(
                plan.root_moves_before_filter, 79,
                "root_moves_raw count at this position drifted"
            );
        }
    }

    /// Every worker's threshold (from `lazy_smp_value_threshold_pct`) filters
    /// from the SAME (root_moves, heat_by_id, max_heat) inputs, only
    /// `threshold_pct` differs -- so a stricter threshold's kept set must be
    /// a subset of a looser one's, UNLESS the stricter one hit the
    /// "kept.is_empty() -> keep all" fallback (which can make it a superset
    /// instead). This guards against the value-filter schedule silently
    /// stopping being monotonic in width.
    #[test]
    fn value_filtered_moves_are_monotonic_by_threshold() {
        let root_moves: Vec<i16> = (0..40).collect();
        let mut heat_by_id = [0i32; HIST_SPAN];
        // Deliberately non-uniform, non-monotonic-by-id heat so the test
        // isn't accidentally satisfied by move ordering alone.
        for (i, &mv) in root_moves.iter().enumerate() {
            heat_by_id[mv as usize] = ((i * 37) % 200) as i32;
        }
        let max_heat = heat_by_id.iter().copied().max().unwrap();

        let mut thresholds: Vec<i32> = (0..8).map(lazy_smp_value_threshold_pct).collect();
        thresholds.sort_unstable();
        thresholds.dedup();

        let kept_sets: Vec<std::collections::HashSet<i16>> = thresholds
            .iter()
            .map(|&pct| {
                lazy_smp_value_filtered_moves(&root_moves, &heat_by_id, max_heat, pct)
                    .into_iter()
                    .collect()
            })
            .collect();

        for w in kept_sets.windows(2) {
            let (looser, stricter) = (&w[0], &w[1]);
            // The empty-kept-set fallback (`lazy_smp_value_filtered_moves`
            // returning everything) is the only case allowed to break strict
            // subset-ness; with this heat distribution no threshold empties
            // out, so the invariant must hold exactly.
            assert!(
                stricter.is_subset(looser),
                "a stricter threshold retained a move the looser one dropped: \
                 looser={looser:?} stricter={stricter:?}"
            );
        }
    }

    #[test]
    fn helper_root_profiles_are_diversified() {
        let root_moves = (0..20).collect::<Vec<i16>>();
        let (main_moves, main_idx) =
            TitaniumSearch::lazy_smp_profile_root_moves(&root_moves, 0, 20, false);
        let (helper_moves, helper_idx) =
            TitaniumSearch::lazy_smp_profile_root_moves(&root_moves, 1, 12, false);
        assert_eq!(main_moves, root_moves);
        assert_eq!(main_idx, (0..20).collect::<Vec<_>>());
        assert_eq!(helper_moves.len(), 12);
        assert_eq!(helper_idx.len(), 12);
        assert_ne!(helper_idx, (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn top_n_worker_plan_limits_effective_root_width() {
        let plan = WorkerPlan {
            worker_id: 4,
            root_move_count: 12,
            root_moves_before_filter: 20,
            root_value_threshold_pct: 40,
            top_n_override: Some(LAZY_SMP_LAST_WORKER_TOP_N),
        };
        assert_eq!(plan.allowed_root_moves(), 3);
        assert_eq!(
            TitaniumSearch::lazy_smp_profile_root_moves(&(0..12).collect::<Vec<i16>>(), 4, 3, true),
            (vec![0, 1, 2], vec![0, 1, 2])
        );
    }

    #[test]
    fn lazy_topn_flag_defaults_off_and_propagates_to_workers() {
        let mut search = fresh();
        assert!(!search.lazy_topn_enabled());
        search.enable_lazy_topn();
        assert!(search.lazy_topn_enabled());
        let worker = search.fork_lazy_worker(&search.g);
        assert!(worker.lazy_topn_enabled());
    }

    #[test]
    fn ace_lmp_defaults_off_and_propagates_to_workers() {
        let mut search = fresh();
        assert!(!search.ace_lmp_enabled());
        search.set_ace_lmp(true);
        assert!(search.ace_lmp_enabled());
        let worker = search.fork_lazy_worker(&search.g);
        assert!(worker.ace_lmp_enabled());
    }

    #[test]
    fn lazy_topn_only_limits_last_worker_when_pool_has_three_threads() {
        let mut search = fresh();
        search.enable_lazy_topn();
        let result = search.think_with_threads(100, 1, true, false, "titanium-v17-lazy-topn", 3);
        assert_eq!(result.root_widths.len(), 3);
        assert_eq!(result.root_widths[0].top_n_override, None);
        assert_eq!(result.root_widths[1].top_n_override, None);
        assert_eq!(
            result.root_widths[2].top_n_override,
            Some(LAZY_SMP_LAST_WORKER_TOP_N)
        );
        assert_eq!(result.root_widths[2].allowed_root_moves(), 3);
        assert!(
            result.root_visits[2].iter().copied().max().unwrap_or(0) < 3,
            "last worker visited outside its effective top-N"
        );
    }

    #[test]
    fn cat_v16_worker_profiles_raise_fringe_threshold() {
        let mut search = *TitaniumSearch::grafted_v17(GameState::new(), Some(18));
        search.set_cat_lmr_worker_profile(0);
        assert_eq!(search.cat_lmr_fringe_pct, 5);
        search.set_cat_lmr_worker_profile(1);
        assert_eq!(search.cat_lmr_fringe_pct, 10);
        search.set_cat_lmr_worker_profile(2);
        assert_eq!(search.cat_lmr_fringe_pct, 20);
        search.set_cat_lmr_worker_profile(4);
        assert_eq!(search.cat_lmr_fringe_pct, 40);
        search.set_cat_lmr_worker_profile(8);
        assert_eq!(search.cat_lmr_fringe_pct, 70);
    }

    #[test]
    fn shared_tt_allocation_and_probe_are_shared() {
        let mut search = fresh();
        search.resize_tt(18);
        let shared = Arc::new(SharedTitaniumTt::from_search(&search));
        let runtime = Arc::new(LazySmpRuntime::new(
            Instant::now() + Duration::from_millis(100),
        ));
        let root_moves = Arc::new(vec![0i16]);
        let root_visit_map = Arc::new(vec![0usize]);
        let mut worker = search.fork_lazy_worker(&GameState::new());
        search.install_lazy_smp_context(
            0,
            shared.clone(),
            runtime.clone(),
            root_moves.clone(),
            root_visit_map.clone(),
            1,
        );
        worker.install_lazy_smp_context(1, shared.clone(), runtime, root_moves, root_visit_map, 1);
        assert!(Arc::ptr_eq(
            search.shared_tt.as_ref().expect("main shared TT"),
            worker.shared_tt.as_ref().expect("helper shared TT")
        ));

        shared.store(
            123,
            456,
            7,
            false,
            SharedTtEntry {
                key_hi: 456,
                key_lo: 123,
                meta: 42 | (0 << 10) | (5 << 12),
                score: 99,
                rep: 0,
                anc_lo: 0,
                anc_hi: 0,
                entry_gen: 7,
            },
        );
        let entry = shared.probe(123, 456).expect("stored helper entry");
        assert_eq!(entry.score, 99);
        assert_eq!(tt_unpack_depth(entry.meta), 5);
    }

    #[test]
    fn tt_depth_field_is_eight_bits_clamped_to_255() {
        assert_eq!(tt_unpack_depth(tt_pack_depth(0)), 0);
        assert_eq!(tt_unpack_depth(tt_pack_depth(128)), 128);
        assert_eq!(tt_unpack_depth(tt_pack_depth(255)), 255);
        assert_eq!(tt_unpack_depth(tt_pack_depth(256)), 255);
        assert_eq!(tt_unpack_depth(tt_pack_depth(10_000)), 255);
        // High garbage bits above the 8-bit depth field must be ignored.
        let dirty = tt_pack_depth(42) | (0x7F << 20);
        assert_eq!(tt_unpack_depth(dirty), 42);
    }

    #[test]
    fn shared_stop_flag_is_observed() {
        let mut search = fresh();
        let runtime = Arc::new(LazySmpRuntime::new(Instant::now() + Duration::from_secs(1)));
        runtime.stop.store(true, Ordering::Relaxed);
        search.lazy_runtime = Some(runtime);
        assert!(search.check_time().is_err());
    }

    #[test]
    fn helper_depth_does_not_replace_main_authority() {
        let mut search = fresh();
        let result = search.think_with_threads(1_000, 2, true, false, "titanium-v15", 4);
        assert_eq!(result.depth, result.main_completed_depth);
        assert_eq!(
            result.main_thread_nodes + result.helper_nodes.iter().sum::<u64>(),
            result.total_nodes
        );
        assert_eq!(result.nodes, result.total_nodes);
    }

    #[test]
    fn helper_partial_is_used_only_when_main_has_no_completed_move() {
        fn result(mv: i16, depth: i32, nodes: u64) -> ThinkResult {
            ThinkResult {
                mv,
                score: depth * 10,
                root_moves: Vec::new(),
                depth,
                nodes,
                main_thread_nodes: 0,
                helper_nodes: Vec::new(),
                total_nodes: 0,
                main_completed_depth: 0,
                helper_completed_depths: Vec::new(),
                root_widths: Vec::new(),
                root_visits: Vec::new(),
                root_move_ids: Vec::new(),
                ms: 0,
                white_dist: 0,
                black_dist: 0,
                depth_log: Vec::new(),
                stop_reason: "test",
                race_outcome_stats: RaceOutcomeStats::default(),
                opening_book: None,
                root_defense_diag: Vec::new(),
                race: RaceResultInfo::from_score(depth * 10),
                timing: TimingDiag::default(),
            }
        }

        let legal_roots = [11, 22, 33];
        let helpers = vec![
            (1, result(99, 8, 99), Vec::new()),
            (2, result(22, 3, 300), Vec::new()),
            (3, result(33, 4, 200), Vec::new()),
        ];
        let main_ready = result(11, 1, 10);
        assert!(
            TitaniumSearch::lazy_smp_helper_partial(&main_ready, &helpers, &legal_roots).is_none()
        );

        let main_empty = result(crate::titanium::TITANIUM_NO_MOVE, 0, 0);
        let adopted = TitaniumSearch::lazy_smp_helper_partial(&main_empty, &helpers, &legal_roots)
            .expect("legal helper result");
        assert_eq!(adopted.mv, 33);
        assert_eq!(adopted.depth, 4);
    }

    #[test]
    fn one_thread_matches_existing_search_at_fixed_depth() {
        let mut old = fresh();
        let mut new = fresh();
        let a = old.think(10_000, 2, true, false, "titanium-v15");
        let b = new.think_with_threads(10_000, 2, true, false, "titanium-v15", 1);
        assert_eq!(a.mv, b.mv);
        assert_eq!(a.score, b.score);
        assert_eq!(a.depth, b.depth);
        assert_eq!(a.nodes, b.nodes);
        assert_eq!(a.depth_log.len(), b.depth_log.len());
    }

    #[test]
    fn race_stress_no_illegal_moves_or_hangs() {
        for _ in 0..8 {
            let mut search = fresh();
            let result = search.think_with_threads(250, 3, true, false, "titanium-v15", 4);
            let mut legal = [0i16; 160];
            let n = search.gen_moves(0, 1, result.mv, &mut legal);
            assert!(n > 0);
            assert!(legal[..n].contains(&result.mv));
        }
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod route_touch_tests {
    use super::*;
    use crate::core::board::WallOrientation;

    fn mask_with(cells: &[(u8, u8)]) -> [u8; 81] {
        let mut m = [0u8; 81];
        for &(r, c) in cells {
            m[(r as usize) * 9 + c as usize] = 1;
        }
        m
    }

    #[test]
    fn wall_touching_route_cell_is_detected() {
        // Horizontal wall at (row=3, col=4) touches cells
        // (3,4) (3,5) (4,4) (4,5). Put a route cell at one of them.
        let route0 = mask_with(&[(4, 5)]);
        let route1 = [0u8; 81];
        assert!(wall_touches_route(
            3,
            4,
            WallOrientation::Horizontal,
            &route0,
            &route1
        ));
    }

    #[test]
    fn wall_touching_neither_players_route_is_not_detected() {
        let route0 = mask_with(&[(0, 0)]);
        let route1 = mask_with(&[(8, 8)]);
        assert!(!wall_touches_route(
            3,
            4,
            WallOrientation::Horizontal,
            &route0,
            &route1
        ));
    }

    #[test]
    fn wall_touching_either_players_route_counts() {
        let route0 = [0u8; 81];
        let route1 = mask_with(&[(3, 4)]); // top-left corner of the wall's 4 cells
        assert!(wall_touches_route(
            3,
            4,
            WallOrientation::Vertical,
            &route0,
            &route1
        ));
    }

    #[test]
    fn route_touch_ordering_flag_defaults_off() {
        let g = GameState::new();
        let search = TitaniumSearch::with_ti_movegen(g);
        assert!(!search.route_touch_ordering);
    }

    #[test]
    fn enable_route_touch_ordering_sets_flag_and_propagates_to_workers() {
        let mut search = TitaniumSearch::with_ti_movegen(GameState::new());
        search.enable_route_touch_ordering();
        assert!(search.route_touch_ordering);
        let worker = search.fork_lazy_worker(&search.g);
        assert!(worker.route_touch_ordering);
    }

    #[test]
    fn q_search_defaults_off_and_has_an_explicit_setter() {
        let search = TitaniumSearch::with_ti_movegen(GameState::new());
        assert!(!search.q_search_enabled());
        let mut search = search;
        search.enable_q_search();
        assert!(search.q_search_enabled());
    }

    #[test]
    fn enable_q_search_propagates_to_workers() {
        let mut search = TitaniumSearch::grafted_v17(GameState::new(), None);
        search.enable_q_search();
        let worker = search.fork_lazy_worker(&search.g);
        assert!(worker.q_search);
        assert_eq!(worker.q_max, Q_SEARCH_MAX_DEFAULT);
    }

    #[test]
    fn ace_rfp_defaults_off_and_propagates_to_workers() {
        let mut search = TitaniumSearch::grafted_v17(GameState::new(), None);
        assert!(!search.ace_rfp_enabled());
        search.set_ace_rfp(true);
        assert!(search.ace_rfp_enabled());
        let worker = search.fork_lazy_worker(&search.g);
        assert!(worker.ace_rfp_enabled());
    }

    #[test]
    fn v17_ab_controls_default_on_and_propagate_to_workers() {
        let search = TitaniumSearch::grafted_v17(GameState::new(), None);
        assert!(search.partial_iter_enabled());
        assert!(search.predict_stop_enabled());
        let worker = search.fork_lazy_worker(&search.g);
        assert!(worker.partial_iter_enabled());
        assert!(worker.predict_stop_enabled());
    }

    #[test]
    fn v17_ab_controls_can_be_disabled_independently() {
        let mut search = TitaniumSearch::grafted_v17(GameState::new(), None);
        search.set_partial_iter(false);
        assert!(!search.partial_iter_enabled());
        assert!(search.predict_stop_enabled());
        search.set_predict_stop(false);
        assert!(!search.partial_iter_enabled());
        assert!(!search.predict_stop_enabled());
        let worker = search.fork_lazy_worker(&search.g);
        assert!(!worker.partial_iter_enabled());
        assert!(!worker.predict_stop_enabled());
    }

    #[test]
    fn rfp_margin_preserves_v17_and_uses_ace_candidate_only_when_enabled() {
        assert_eq!(reverse_futility_margin(3, true, false, 3), Some(210));
        assert_eq!(reverse_futility_margin(4, false, false, 3), Some(360));
        assert_eq!(reverse_futility_margin(4, true, true, 3), None);
        assert_eq!(reverse_futility_margin(3, false, true, 3), Some(300));
        assert_eq!(reverse_futility_margin(4, true, true, 4), Some(400));
        assert_eq!(rfp_depth_for_budget(true, 200), 4);
        assert_eq!(rfp_depth_for_budget(true, 201), 3);
        assert_eq!(rfp_depth_for_budget(false, 100), 3);
    }
}

/// Production: 21-bit f32 (~32MB table at 16B/entry). Feature `eval_cache_baseline`
/// keeps the pre-B+C 21-bit f64 path for A/B benches only. wasm stays 16 bits.
#[cfg(not(target_arch = "wasm32"))]
const EVAL_CACHE_BITS: usize = 21;
#[cfg(target_arch = "wasm32")]
const EVAL_CACHE_BITS: usize = 16;
const EVAL_CACHE_SIZE: usize = 1 << EVAL_CACHE_BITS;

/// Static-eval cache entry. Baseline feature uses f64; production uses f32.
#[derive(Clone, Copy)]
struct EvalCacheEntry {
    key: u64,
    #[cfg(feature = "eval_cache_baseline")]
    val: f64,
    #[cfg(not(feature = "eval_cache_baseline"))]
    val: f32,
    /// wl0<<8|wl1 verify tag; u16::MAX = empty slot.
    meta: u16,
}

impl Default for EvalCacheEntry {
    fn default() -> Self {
        Self {
            key: 0,
            val: 0.0,
            meta: u16::MAX,
        }
    }
}

/// Wall-topology → goal-distance-fields cache (both players). Sibling walls and
/// iterative-deepening re-searches revisit the same topologies constantly; a hit
/// turns a two-sided BFS reflood into a short memcpy. Inline layers are
/// `[u128; DIST_LAYER_INLINE]` per side (entry size depends on inline capacity;
/// `dist_layers_full81` uses full 81); depths beyond that spill
/// to a freelist pool (≈99.96% of stores fit inline).
///
/// Adaptive sizing mirrors the TT strategy (`enable_adaptive_tt`/`tt_grow`):
/// start small so short searches never pay allocation/zeroing for a table they
/// won't fill, grow on 50% occupancy with a live-entry rehash. Measured on the
/// i7-4900MQ: 9 bits thrashes at depth 14+ (75% miss, ~1 BFS reflood/node);
/// 13 bits cuts refloods 25-45%; 14-15 bits are a wash (TLB pressure eats the
/// extra hits), so 13 caps native. wasm caps lower — browser memory is the
/// scarce resource there (256MB threaded-build ceiling, one table per worker).
// One-shot callers (genmove, match, bench) never get to grow past a cold
// start before they exit, so the adaptive grow-on-occupancy path below barely
// runs. Starting directly at the steady-state size (~6MB/process with inline
// layers) skips that cold-ramp entirely instead of thrashing at 512 entries.
// Measured +38-49% single-thread NPS on an idle i7-4900MQ vs starting at 1<<9
// (startpos, fixed move/score/nodes-shape unchanged — search-identical, cache
// size only).
#[cfg(not(target_arch = "wasm32"))]
const DIST_LRU_MIN_BITS: usize = 13;
#[cfg(not(target_arch = "wasm32"))]
const DIST_LRU_MAX_BITS: usize = 13;
#[cfg(target_arch = "wasm32")]
const DIST_LRU_MIN_BITS: usize = 7;
#[cfg(target_arch = "wasm32")]
const DIST_LRU_MAX_BITS: usize = 10;

/// Inline layer capacity in each DistTopoEntry. Working ply arrays stay
/// `[u128; 81]`; only the cache entry shrinks. Default 16 covers 99.96% ≤16.
/// Features (A/B only, not production):
/// - `dist_layers_full81` → 81
/// - `dist_layers_inline12` → 12 (~98% ≤12; cache-line probe)
#[cfg(feature = "dist_layers_full81")]
const DIST_LAYER_INLINE: usize = 81;
#[cfg(all(not(feature = "dist_layers_full81"), feature = "dist_layers_inline12"))]
const DIST_LAYER_INLINE: usize = 12;
#[cfg(all(not(feature = "dist_layers_full81"), not(feature = "dist_layers_inline12")))]
const DIST_LAYER_INLINE: usize = 16;

/// Heap tails for rare DistTopoEntry depths beyond [`DIST_LAYER_INLINE`].
/// Keyed by wall-topology `wkey` in `TitaniumSearch::dist_layer_spill` — not
/// stored on the hot entry (avoids paying spill metadata on every slot).
struct DistLayerSpill {
    /// layers[DIST_LAYER_INLINE .. d0_depth]
    d0_tail: Vec<u128>,
    /// layers[DIST_LAYER_INLINE .. d1_depth]
    d1_tail: Vec<u128>,
}

#[derive(Clone)]
struct DistTopoEntry {
    /// wall_topology_key (hi<<32|lo); u64::MAX = empty.
    key: u64,
    d0_depth: u16,
    d1_depth: u16,
    d0: [u8; 81],
    d1: [u8; 81],
    d0_layers: [u128; DIST_LAYER_INLINE],
    d1_layers: [u128; DIST_LAYER_INLINE],
}

impl Default for DistTopoEntry {
    fn default() -> Self {
        Self {
            key: u64::MAX,
            d0_depth: 0,
            d1_depth: 0,
            d0: [255; 81],
            d1: [255; 81],
            d0_layers: [0; DIST_LAYER_INLINE],
            d1_layers: [0; DIST_LAYER_INLINE],
        }
    }
}

/// Bytes of a full DistTopoEntry allocation (scalar + inline layers).
#[inline]
pub fn dist_topo_entry_size_bytes() -> usize {
    std::mem::size_of::<DistTopoEntry>()
}

/// Approximate scalar payload: key(8)+d0_depth(2)+d1_depth(2)+d0(81)+d1(81)=174.
#[inline]
pub fn dist_topo_scalar_bytes() -> usize {
    8 + 2 + 2 + 81 + 81
}

/// Inline layer arrays only: 2 * DIST_LAYER_INLINE * size_of::<u128>() = 512.
#[inline]
pub fn dist_topo_layer_bytes() -> usize {
    2 * DIST_LAYER_INLINE * std::mem::size_of::<u128>()
}

#[inline]
fn eval_cache_slot(hash64: u64, bits: usize) -> usize {
    if bits == 0 {
        0
    } else {
        (hash64.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - bits)) as usize
    }
}

#[inline]
fn dist_lru_slot(wkey: u64, bits: usize) -> usize {
    if bits == 0 {
        0
    } else {
        (wkey.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> (64 - bits)) as usize
    }
}

/// FxHash (the algorithm rustc itself uses for its internal maps): a
/// non-cryptographic hasher for small POD keys. `std::collections::HashMap`
/// defaults to SipHash-1-3, which costs ~100-300ns/lookup on a heterogeneous
/// tuple — measured as a flat ~390ns/node floor in `evaluate_tail` regardless
/// of position (wall count, search depth), the signature of hash overhead
/// rather than real work. `cw_cache` keys are internal-only (never attacker
/// controlled), so DoS-resistance buys nothing here.
#[derive(Default)]
struct FxHasher {
    hash: u64,
}

impl FxHasher {
    const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

    #[inline(always)]
    fn add(&mut self, w: u64) {
        self.hash = (self.hash.rotate_left(5) ^ w).wrapping_mul(Self::SEED);
    }
}

impl std::hash::Hasher for FxHasher {
    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        for chunk in bytes.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            self.add(u64::from_ne_bytes(buf));
        }
    }
    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline(always)]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline(always)]
    fn write_i32(&mut self, i: i32) {
        self.add(i as u64);
    }
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.hash
    }
}

type FxBuildHasher = std::hash::BuildHasherDefault<FxHasher>;

const TT_BITS: usize = 20;
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: u32 = (TT_SIZE - 1) as u32;

// Root-move width percent per worker. Reverted from the "narrow-first"
// schedule (worker 0 at just 10%, commit 3daf94c) -- that left main's
// iterative deepening authoritative over only its top ~10% of root moves by
// initial move-ordering guess, so a deep, confident-looking main search could
// be completely blind to the true best move whenever ordering ranked it
// outside that slice. Main now keeps almost the full root (95%, leaving a
// small margin to skip only moves ordering is very confident are losing);
// helpers narrow progressively for deeper per-move lookahead, floored at 40%
// (not 20%) since helper results only ever matter as an emergency fallback
// when main produces nothing (see lazy_smp_helper_partial).
const LAZY_SMP_WIDTHS: [usize; 4] = [95, 80, 80, 80];

// The very last worker (worker_id == threads - 1, when there are at least 3
// threads so it's a distinct role from main and the uniform-80% helpers)
// skips the percentage schedule entirely and searches only this many
// root moves -- the top-N by move ordering. Its job isn't breadth, it's
// squeezing maximum depth out of whatever the rest of the pool already
// agrees are the most promising candidates.
const LAZY_SMP_LAST_WORKER_TOP_N: usize = 3;

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
pub const LAZY_SMP_MAX_THREADS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerPlan {
    pub worker_id: usize,
    /// Root moves actually retained for this worker AFTER CAT-value filtering
    /// (i.e. `filtered_by_worker[worker_id].len()`) -- this is the real count
    /// the search runs with, not a count to be further percentage-cut.
    pub root_move_count: usize,
    /// Total legal root moves at this position BEFORE any per-worker filtering.
    pub root_moves_before_filter: usize,
    /// The CAT impact-heat cutoff (percent of the position's max root-move
    /// heat) used to build `root_move_count` -- NOT a percent of move COUNT.
    /// Renamed from `root_width_percent`, which the diagnostics printer and
    /// `allowed_root_moves()` used to misread as a move-count percentage and
    /// re-apply on top of the already-filtered count (main.rs's old `"allowed"`
    /// JSON field, and the now-rewritten `root_filtering_limits_each_worker_to_its_width`
    /// test, both computed a second, bogus cut this way).
    pub root_value_threshold_pct: usize,
    pub top_n_override: Option<usize>,
}

impl WorkerPlan {
    /// Real number of root moves this worker searches. The CAT-value threshold
    /// has already been applied when `root_move_count` was computed (see
    /// `filtered_by_worker` in `think_lazy_smp`) -- this must NOT re-apply it
    /// as a move-count percentage on top.
    pub fn allowed_root_moves(&self) -> usize {
        if let Some(top_n) = self.top_n_override {
            return top_n.max(1).min(self.root_move_count.max(1));
        }
        self.root_move_count
    }

    pub fn root_moves_retained_pct(&self) -> f64 {
        if self.root_moves_before_filter == 0 {
            return 0.0;
        }
        100.0 * self.root_move_count as f64 / self.root_moves_before_filter as f64
    }
}

// EXPERIMENT (not the shipped schedule): per-worker root-move cutoff by CAT
// impact-heat VALUE relative to the best root move's heat, not by move count.
// A move survives worker `w` iff heat(move) >= pct[w]% * max(heat(any root
// move)). This is a genuinely different criterion than LAZY_SMP_WIDTHS: in a
// position with one dominant move it collapses hard (real tail-cut); in a
// flat position where many moves are nearly as good it keeps most of them,
// regardless of raw count. main=20% (only drop clearly-useless tail moves),
// then progressively stricter per worker, floored at 40%.
const LAZY_SMP_VALUE_THRESHOLD_PCTS: [i32; 4] = [20, 30, 40, 40];

fn lazy_smp_value_threshold_pct(worker_id: usize) -> i32 {
    LAZY_SMP_VALUE_THRESHOLD_PCTS
        .get(worker_id)
        .copied()
        .unwrap_or(*LAZY_SMP_VALUE_THRESHOLD_PCTS.last().expect("non-empty"))
}

/// Keep only root moves whose CAT impact heat clears `threshold_pct`% of the
/// best root move's heat. Falls back to keeping everything when there's no
/// CAT signal at all (max_heat <= 0) -- absence of signal is not evidence a
/// move is useless.
fn lazy_smp_value_filtered_moves(
    root_moves: &[i16],
    heat_by_id: &[i32; HIST_SPAN],
    max_heat: i32,
    threshold_pct: i32,
) -> Vec<i16> {
    if max_heat <= 0 {
        return root_moves.to_vec();
    }
    let kept: Vec<i16> = root_moves
        .iter()
        .copied()
        .filter(|&m| {
            let h = heat_by_id[m as usize].max(0);
            h.saturating_mul(100) >= threshold_pct.saturating_mul(max_heat)
        })
        .collect();
    if kept.is_empty() {
        root_moves.to_vec()
    } else {
        kept
    }
}

pub fn lazy_smp_allowed_root_moves(root_move_count: usize, root_width_percent: usize) -> usize {
    if root_move_count == 0 {
        return 0;
    }
    let allowed = root_move_count
        .saturating_mul(root_width_percent)
        .saturating_add(99)
        / 100;
    allowed.max(1).min(root_move_count)
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
#[derive(Debug, Clone, Copy, Default)]
struct SharedTtEntry {
    key_hi: u32,
    key_lo: u32,
    meta: i32,
    score: i32,
    rep: u8,
    anc_lo: u32,
    anc_hi: u32,
    entry_gen: u8,
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
struct SharedTitaniumTt {
    slots: Vec<RwLock<SharedTtEntry>>,
    mask: u32,
    bits: usize,
    filled: AtomicUsize,
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
impl SharedTitaniumTt {
    fn from_search(search: &TitaniumSearch) -> Self {
        let slots = (0..search.tt_meta.len())
            .map(|i| {
                RwLock::new(SharedTtEntry {
                    key_hi: search.tt_key_hi[i],
                    key_lo: search.tt_key_lo[i],
                    meta: search.tt_meta[i],
                    score: search.tt_score[i],
                    rep: search.tt_rep[i],
                    anc_lo: search.tt_anc_lo[i],
                    anc_hi: search.tt_anc_hi[i],
                    entry_gen: search.tt_entry_gen[i],
                })
            })
            .collect();
        Self {
            slots,
            mask: search.tt_mask,
            bits: search.tt_bits,
            filled: AtomicUsize::new(search.tt_filled),
        }
    }

    fn probe(&self, hash_lo: u32, hash_hi: u32) -> Option<SharedTtEntry> {
        let idx = (hash_lo & self.mask) as usize;
        let entry = *self.slots[idx]
            .read()
            .expect("shared TT read lock poisoned");
        if entry.meta != 0 && entry.key_hi == hash_hi && entry.key_lo == hash_lo {
            Some(entry)
        } else {
            None
        }
    }

    fn store(&self, hash_lo: u32, hash_hi: u32, tt_gen: u8, pure_mode: bool, entry: SharedTtEntry) {
        let idx = (hash_lo & self.mask) as usize;
        let mut slot = self.slots[idx]
            .write()
            .expect("shared TT write lock poisoned");
        let was_empty = slot.meta == 0;
        let stale_gen = !pure_mode && !was_empty && slot.entry_gen != tt_gen;
        let deeper =
            !was_empty && !stale_gen && tt_unpack_depth(entry.meta) >= tt_unpack_depth(slot.meta);
        if was_empty || stale_gen || deeper {
            *slot = entry;
            if was_empty {
                self.filled.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
struct LazySmpRuntime {
    stop: AtomicBool,
    global_nodes: AtomicU64,
    deadline: Instant,
}

#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
impl LazySmpRuntime {
    fn new(deadline: Instant) -> Self {
        Self {
            stop: AtomicBool::new(false),
            global_nodes: AtomicU64::new(0),
            deadline,
        }
    }
}

/// Time-abort marker — propagates like the JS `throw "time"`.
pub struct TimeUp;

/// Titanium `Board` kept in sync with the ACE game — fast movegen + optional CAT.
pub struct TiBridge {
    pub board: Board,
    pub bfs: BfsScratch,
    undo_stack: Vec<Undo>,
    geometric_walls: Option<GeometricWallCache>,
    pub wall_cache_stats: GeometricWallCacheStats,
}

impl TiBridge {
    fn from_game(g: &GameState) -> Box<Self> {
        let mut board = Board::new();
        for i in 0..g.hist_len {
            let _ = board.make_move(move_id_to_board(g.hist_m[i]));
        }
        Box::new(Self {
            board,
            bfs: BfsScratch::new(),
            undo_stack: Vec::with_capacity(256),
            geometric_walls: None,
            wall_cache_stats: GeometricWallCacheStats::default(),
        })
    }

    fn push(&mut self, m: i16) {
        let undo = self.board.make_move(move_id_to_board(m));
        self.undo_stack.push(undo);
    }

    fn pop(&mut self) {
        if let Some(undo) = self.undo_stack.pop() {
            self.board.unmake_move(undo);
        }
    }

    /// Full legal moves via Titanium `movegen` → dense encoding.
    fn gen_legal_ace(&mut self, out: &mut [i16; 160]) -> usize {
        let mut ti_buf = [BoardMove::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n = generate_legal_moves_slice_cached(
            &mut self.geometric_walls,
            &mut self.board,
            &mut ti_buf,
            &mut self.bfs,
            Some(&mut self.wall_cache_stats),
        );
        for i in 0..n {
            out[i] = board_move_to_move_id(ti_buf[i]);
        }
        n
    }
}

/// Titanium board move → ACE numeric encoding.
pub fn board_move_to_move_id(mv: BoardMove) -> i16 {
    match mv {
        BoardMove::Pawn { row, col } => ((8 - row as i16) * 9 + col as i16) as i16,
        BoardMove::Wall {
            row,
            col,
            orientation,
        } => {
            let slot = (7 - row as i16) * 8 + col as i16;
            match orientation {
                WallOrientation::Horizontal => MOVE_HW_BASE + slot,
                WallOrientation::Vertical => MOVE_VW_BASE + slot,
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct AceDepthLogEntry {
    pub depth: i32,
    pub score: i32,
    pub nodes: u64,
    pub elapsed_ms: u64,
    pub marginal_nodes: u64,
    pub pv: String,
}

/// Race semantics attached to a final result.  A proof bound and an exact DTM
/// are deliberately different states: an approximate ETA must never enter the
/// exact `RACE_MATE - dtm` score band.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RaceResultInfo {
    /// `1` = side to move at the root is proven to win, `-1` = proven to lose.
    pub outcome: i8,
    /// Cheap winner-arrival estimate.  This is metadata, never an exact score.
    pub approximate_plies: Option<u16>,
    /// Symmetric uncertainty of `approximate_plies`, in plies.
    pub approximation_tolerance: u8,
    /// Exact retrograde DTM when it was actually requested and completed.
    pub exact_dtm: Option<u16>,
}

impl RaceResultInfo {
    #[inline]
    fn approximate(outcome: i8, plies: Option<u16>) -> Self {
        Self {
            outcome,
            approximate_plies: plies,
            approximation_tolerance: u8::from(plies.is_some()),
            exact_dtm: None,
        }
    }

    #[inline]
    fn exact(outcome: i8, dtm: u16) -> Self {
        Self {
            outcome,
            approximate_plies: None,
            approximation_tolerance: 0,
            exact_dtm: Some(dtm),
        }
    }

    #[inline]
    fn from_score(score: i32) -> Self {
        let abs = score.abs();
        if abs > RACE_WIN_FLOOR && abs <= RACE_MATE {
            Self::exact(score.signum() as i8, (RACE_MATE - abs) as u16)
        } else {
            Self::default()
        }
    }
}

/// Post-search clock/ID diagnostics for time-management telemetry.
/// Filled after search from existing depth_log + stop path; does not change
/// search decisions. Cheap enough for default release match builds.
#[derive(Clone, Copy, Debug, Default)]
pub struct TimingDiag {
    /// Caller-supplied hard budget (`go` time_ms).
    pub allocated_hard_ms: u64,
    /// Soft stop budget = hard * (0.85 or 0.92 when losing).
    pub allocated_soft_ms: u64,
    /// Hard budget minus RaceProof gate reserve.
    pub searchable_ms: u64,
    pub gate_reserve_ms: u64,
    pub elapsed_ms: u64,
    /// `elapsed - searchable` (negative = unfinished budget).
    pub hard_overshoot_ms: i64,
    /// `elapsed - soft`.
    pub soft_overshoot_ms: i64,
    pub last_iter_ms: u64,
    pub prev_iter_ms: u64,
    pub best_move_changes: u32,
    pub partial_iter_used: bool,
    /// Soft fraction in basis points (8500 or 9200).
    pub soft_fraction_bp: u16,
}

impl TimingDiag {
    fn from_think(
        time_ms: u64,
        gate_reserve_ms: u64,
        _last_score: i32,
        elapsed_ms: u64,
        depth_log: &[AceDepthLogEntry],
        best_move_changes: u32,
        partial_iter_used: bool,
        soft_ms: u64,
    ) -> Self {
        let soft = soft_ms.min(time_ms).max(1);
        let searchable = time_ms.saturating_sub(gate_reserve_ms);
        let (last_iter_ms, prev_iter_ms) = iter_costs_ms(depth_log);
        let frac = soft as f64 / time_ms.max(1) as f64;
        Self {
            allocated_hard_ms: time_ms,
            allocated_soft_ms: soft,
            searchable_ms: searchable,
            gate_reserve_ms,
            elapsed_ms,
            hard_overshoot_ms: elapsed_ms as i64 - searchable as i64,
            soft_overshoot_ms: elapsed_ms as i64 - soft as i64,
            last_iter_ms,
            prev_iter_ms,
            best_move_changes,
            partial_iter_used,
            soft_fraction_bp: (frac * 10_000.0).round() as u16,
        }
    }
}

fn iter_costs_ms(depth_log: &[AceDepthLogEntry]) -> (u64, u64) {
    let n = depth_log.len();
    if n == 0 {
        return (0, 0);
    }
    let last = depth_log[n - 1].elapsed_ms.saturating_sub(if n >= 2 {
        depth_log[n - 2].elapsed_ms
    } else {
        0
    });
    let prev = if n >= 2 {
        depth_log[n - 2].elapsed_ms.saturating_sub(if n >= 3 {
            depth_log[n - 3].elapsed_ms
        } else {
            0
        })
    } else {
        0
    };
    (last, prev)
}

#[derive(Clone)]
pub struct ThinkResult {
    pub mv: i16,
    pub score: i32,
    pub root_moves: Vec<(i16, i32)>,
    pub depth: i32,
    pub nodes: u64,
    pub main_thread_nodes: u64,
    pub helper_nodes: Vec<u64>,
    pub total_nodes: u64,
    pub main_completed_depth: i32,
    pub helper_completed_depths: Vec<i32>,
    pub root_widths: Vec<WorkerPlan>,
    pub root_visits: Vec<Vec<usize>>,
    /// Retained root move IDs per worker (index-parallel to `root_widths` /
    /// `root_visits`) AFTER CAT-value filtering -- empty except from
    /// `think_lazy_smp`, which is the only path that filters per worker.
    pub root_move_ids: Vec<Vec<i16>>,
    pub ms: u64,
    pub white_dist: u8,
    pub black_dist: u8,
    pub depth_log: Vec<AceDepthLogEntry>,
    pub stop_reason: &'static str,
    pub race_outcome_stats: RaceOutcomeStats,
    pub opening_book: Option<crate::titanium::opening_book::OpeningBookDiagnostics>,
    /// Last lost-position root defense pass (one entry per legal root move searched).
    pub root_defense_diag: Vec<RootDefenseDiag>,
    pub race: RaceResultInfo,
    pub timing: TimingDiag,
}

/// One complete late-move pipeline observation. These records are emitted only
/// by the offline counterfactual collector; production search leaves probing off.
#[derive(Debug, Clone)]
pub struct ReductionProbeEvent {
    pub ordinal: u64,
    pub parent_hash_lo: u32,
    pub parent_hash_hi: u32,
    pub child_hash_lo: u32,
    pub child_hash_hi: u32,
    pub mv: i16,
    pub depth: i32,
    pub ply: usize,
    pub alpha: i32,
    pub beta: i32,
    pub move_index: usize,
    pub base_reduction: i32,
    pub applied_extra_reduction: bool,
    pub verification_triggered: bool,
    pub self_gain: i32,
    pub opponent_delay: i32,
    pub race_gain: i32,
    pub path_adjustment: i32,
    pub final_reduction: i32,
    pub thread_aggression_percent: i32,
    pub score: i32,
    pub nodes: u64,
    pub hidden: [f64; MAX_NET_H],
    /// Total legal moves generated at this node (enables rank_percentile computation).
    pub total_legal_moves: usize,
    /// Raw history-table score for this wall move (proxy for ordering confidence).
    pub history_score: i32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ReductionShadowStats {
    pub evaluations: u64,
    pub hypothetical_activations: u64,
    pub inference_nanos: u64,
}

pub fn score_label(score: i32) -> String {
    let abs = score.abs();
    if abs >= MATE - 1_000 {
        let plies = MATE - abs;
        if score > 0 {
            format!("mate in {}", plies.max(0))
        } else {
            format!("mated in {}", plies.max(0))
        }
    } else if abs > RACE_WIN_FLOOR && abs <= RACE_MATE {
        let plies = RACE_MATE - abs;
        if score > 0 {
            format!("race win in {}", plies.max(0))
        } else {
            format!("race loss in {}", plies.max(0))
        }
    } else if score == RACE_WIN_FLOOR {
        "proven race win".to_owned()
    } else if score == -RACE_WIN_FLOOR {
        "proven race loss".to_owned()
    } else {
        format!("cp {score}")
    }
}

#[cfg(test)]
mod score_label_tests {
    use super::*;

    #[test]
    fn labels_race_scores_as_forced_races() {
        assert_eq!(score_label(RACE_MATE - 30), "race win in 30");
        assert_eq!(score_label(-(RACE_MATE - 17)), "race loss in 17");
    }

    #[test]
    fn stability_soft_extends_on_instability_and_shortens_when_quiet() {
        let base = TitaniumSearch::stability_soft_fraction(0, 0, 0, 0);
        assert!((base - 0.85).abs() < 1e-9);
        let unstable = TitaniumSearch::stability_soft_fraction(0, 3, -50, 0);
        assert!(unstable > base);
        assert!(unstable <= 1.0);
        let quiet = TitaniumSearch::stability_soft_fraction(0, 0, 0, 3);
        assert!(quiet < base);
        assert!(quiet >= 0.55);
        let losing_base = TitaniumSearch::stability_soft_fraction(-100, 0, 0, 0);
        assert!((losing_base - 0.92).abs() < 1e-9);
    }

    #[test]
    fn root_moves_progress_json_includes_rank_and_multipv() {
        let root_moves = vec![(76i16, 80i32), (4i16, 123i32)];
        let json = ace_progress_json(
            "test",
            &[],
            1,
            0,
            0,
            &[],
            0,
            123,
            &root_moves,
            5,
            5,
            0,
            true,
            2,
        );
        assert!(json.contains(r#""rank":1"#));
        assert!(json.contains(r#""rank":2"#));
        assert!(json.contains(r#""move":"e9"#));
        assert!(json.contains(r#""move":"e1"#));
        assert!(json.contains(r#""multiPv":["#));
        assert!(json.contains(r#""score":123"#));
        assert!(json.contains(r#""score":80"#));

        let json_off = ace_progress_json(
            "test",
            &[],
            1,
            0,
            0,
            &[],
            0,
            123,
            &root_moves,
            5,
            5,
            0,
            false,
            2,
        );
        assert!(json_off.contains(r#""rootMoves":[]"#));
        assert!(json_off.contains(r#""multiPv":["#));
    }

    #[test]
    fn bound_is_proven_but_never_decoded_as_exact_dtm() {
        assert_eq!(score_label(RACE_WIN_FLOOR), "proven race win");
        assert_eq!(score_label(-RACE_WIN_FLOOR), "proven race loss");
        assert_eq!(proven_score_dtm(RACE_WIN_FLOOR), None);
        assert_eq!(proven_score_dtm(-RACE_WIN_FLOOR), None);
    }

    #[test]
    fn exact_race_band_uses_the_declared_maximum_not_a_magic_thousand() {
        let longest_exact = RACE_WIN_FLOOR + 1;
        assert_eq!(
            proven_score_dtm(longest_exact),
            Some(RACE_MATE - longest_exact)
        );
    }

    #[test]
    fn approximate_race_tie_break_requires_disjoint_tolerance_intervals() {
        let fast = RaceRootCandidate {
            mv: 1,
            root_wins: true,
            approximate_plies: Some(5),
            exact_dtm: None,
        };
        let overlaps = RaceRootCandidate {
            mv: 2,
            root_wins: true,
            approximate_plies: Some(7),
            exact_dtm: None,
        };
        let separated = RaceRootCandidate {
            mv: 3,
            root_wins: true,
            approximate_plies: Some(8),
            exact_dtm: None,
        };
        assert!(!race_candidate_definitely_best(fast, &[fast, overlaps]));
        assert!(race_candidate_definitely_best(fast, &[fast, separated]));

        let slow_loss = RaceRootCandidate {
            mv: 4,
            root_wins: false,
            approximate_plies: Some(12),
            exact_dtm: None,
        };
        let quick_loss = RaceRootCandidate {
            mv: 5,
            root_wins: false,
            approximate_plies: Some(9),
            exact_dtm: None,
        };
        assert!(race_candidate_definitely_best(
            slow_loss,
            &[slow_loss, quick_loss]
        ));
    }

    fn two_wall_fixture(p0: &str, p1: &str, hands: [i32; 2], turn: usize) -> GameState {
        use crate::titanium::game::ZOBRIST;
        let mut game = GameState::new();
        game.pawn = [
            crate::titanium::algebraic_to_move_id(p0) as usize,
            crate::titanium::algebraic_to_move_id(p1) as usize,
        ];
        game.wl = hands;
        game.turn = turn;
        game.hash_lo = ZOBRIST.pawn_lo[0][game.pawn[0]] ^ ZOBRIST.pawn_lo[1][game.pawn[1]];
        game.hash_hi = ZOBRIST.pawn_hi[0][game.pawn[0]] ^ ZOBRIST.pawn_hi[1][game.pawn[1]];
        if turn != 0 {
            game.hash_lo ^= ZOBRIST.turn_lo;
            game.hash_hi ^= ZOBRIST.turn_hi;
        }
        game
    }

    fn pawn_sq_fixture(p0: usize, p1: usize, hands: [i32; 2], turn: usize) -> GameState {
        use crate::titanium::game::ZOBRIST;
        let mut game = GameState::new();
        game.pawn = [p0, p1];
        game.wl = hands;
        game.turn = turn;
        game.hash_lo = ZOBRIST.pawn_lo[0][p0] ^ ZOBRIST.pawn_lo[1][p1];
        game.hash_hi = ZOBRIST.pawn_hi[0][p0] ^ ZOBRIST.pawn_hi[1][p1];
        if turn != 0 {
            game.hash_lo ^= ZOBRIST.turn_lo;
            game.hash_hi ^= ZOBRIST.turn_hi;
        }
        game
    }

    /// Oracle-certified empty-board (p0, p1) with turn=0 where STM wins (`stm_wins`)
    /// or loses the pure race. Builds the empty topology race table once.
    fn find_empty_board_pure_race_pair(stm_wins: bool) -> (usize, usize) {
        let mut search = enabled_two_wall_search(GameState::new());
        search.race_proof = true;
        search.g.wl = [0, 0];
        let slot = search
            .race_tbl(true)
            .expect("empty-board race_tbl must build under test caps");
        for p0 in 0..81usize {
            for p1 in 0..81usize {
                if p0 == p1 {
                    continue;
                }
                search.g.pawn = [p0, p1];
                search.g.turn = 0;
                let rv = search.race_value(slot);
                if rv == 0 {
                    continue;
                }
                if (rv > 0) == stm_wins {
                    return (p0, p1);
                }
            }
        }
        panic!("no empty-board pure-race fixture for stm_wins={stm_wins}");
    }

    fn enabled_two_wall_search(game: GameState) -> Box<TitaniumSearch> {
        let mut search = TitaniumSearch::new(game);
        search.two_wall_race_resolved = Some(true);
        search.rp_build_ok = true;
        search.rc_count_cap = u32::MAX;
        search.rc_solve_cap = f64::INFINITY;
        search.deadline = Instant::now() + Duration::from_secs(30);
        search
    }

    fn enabled_one_wall_search(game: GameState) -> Box<TitaniumSearch> {
        let mut search = enabled_two_wall_search(game);
        search.one_wall_race_resolved = Some(true);
        search.two_wall_race_resolved = Some(false);
        search
    }

    #[test]
    fn one_wall_subset_accepts_holder_pure_race_win_as_bound() {
        let game = two_wall_fixture("a6", "i4", [1, 0], 0);
        let mut search = enabled_one_wall_search(game);
        assert_eq!(
            search.one_wall_race_bound(),
            RaceBound::Lower(RACE_WIN_FLOOR)
        );
        assert_eq!(search.race_outcome_stats.one_wall_decisive, 1);
    }

    #[test]
    fn one_wall_subset_declines_when_delayed_wall_can_matter() {
        // The holder loses the pure race, but the opponent is not one step
        // from goal. A delayed placement may matter, so the subset must fall
        // back instead of manufacturing a loss.
        let game = two_wall_fixture("a2", "i5", [1, 0], 0);
        let mut search = enabled_one_wall_search(game);
        assert_eq!(search.one_wall_race_bound(), RaceBound::Unknown);
    }

    #[test]
    fn two_wall_subset_is_default_off() {
        let game = two_wall_fixture("a7", "i4", [2, 0], 0);
        let mut search = TitaniumSearch::new(game);
        search.two_wall_race_resolved = Some(false);
        assert_eq!(search.two_wall_monopoly_race_bound(), RaceBound::Unknown);
        assert_eq!(search.race_outcome_stats.two_wall_calls, 0);
    }

    #[test]
    fn two_wall_subset_accepts_monopoly_pure_race_win_as_bound() {
        let game = two_wall_fixture("a7", "i4", [2, 0], 0);
        let mut search = enabled_two_wall_search(game);
        assert_eq!(
            search.two_wall_monopoly_race_bound(),
            RaceBound::Lower(RACE_WIN_FLOOR)
        );
        assert_eq!(search.race_outcome_stats.two_wall_decisive, 1);
    }

    #[test]
    fn two_wall_forced_placement_fixture_matches_ordinary_search_winner() {
        let game = two_wall_fixture("a7", "i2", [2, 0], 0);
        let mut proof = enabled_two_wall_search(game.clone());
        let bound = proof.two_wall_monopoly_race_bound();
        assert_ne!(bound, RaceBound::Unknown, "forced-wall subset declined");

        let mut ordinary = TitaniumSearch::new(game);
        ordinary.two_wall_race_resolved = Some(false);
        let result = ordinary.think(30_000, 5, true, false, "titanium-v17");
        assert!(
            result.score.abs() >= MATE - 1_000,
            "ordinary depth-5 search did not prove the fixture: {}",
            result.score
        );
        assert_eq!(bound.signum(), result.score.signum());
    }

    #[test]
    fn two_wall_subset_rejects_split_nonholder_and_interacting_pawns() {
        let cases = [
            two_wall_fixture("a7", "i2", [1, 1], 0),
            two_wall_fixture("a7", "i2", [0, 2], 0),
            two_wall_fixture("e5", "e6", [2, 0], 0),
        ];
        for game in cases {
            let mut search = enabled_two_wall_search(game);
            assert_eq!(search.two_wall_monopoly_race_bound(), RaceBound::Unknown);
        }
    }

    #[test]
    fn broke_side_lower_when_opp_out_of_walls_and_stm_wins_pure_race() {
        let (p0, p1) = find_empty_board_pure_race_pair(true);
        let game = pawn_sq_fixture(p0, p1, [3, 0], 0);
        let mut search = enabled_two_wall_search(game);
        search.race_proof = true;
        search.rp_build_ok = true;
        match search.one_side_broke_race_bound() {
            RaceBound::Lower(v) => {
                assert!(
                    v > RACE_WIN_FLOOR && v <= RACE_MATE,
                    "expected DTM-preserving Lower, got {v}"
                );
            }
            other => panic!("expected Lower, got {other:?} (p0={p0} p1={p1})"),
        }
        assert_eq!(search.race_outcome_stats.broke_decisive, 1);
    }

    #[test]
    fn broke_side_lower_jump_path_fixture() {
        // P0 can jump over P1 (46 -> 28) on the exact empty-board race path.
        // With P1 unable to place, the refuse-to-place theorem must preserve
        // that jump-aware winner as a Lower bound.
        let game = pawn_sq_fixture(46, 37, [3, 0], 0);
        let mut search = enabled_two_wall_search(game);
        search.race_proof = true;
        search.rp_build_ok = true;
        match search.one_side_broke_race_bound() {
            RaceBound::Lower(v) => assert!(v > RACE_WIN_FLOOR && v <= RACE_MATE),
            other => panic!("expected jump-aware Lower, got {other:?}"),
        }
        assert_eq!(search.race_outcome_stats.broke_decisive, 1);
    }

    #[test]
    fn broke_side_upper_when_stm_out_of_walls_and_loses_pure_race() {
        let (p0, p1) = find_empty_board_pure_race_pair(false);
        let game = pawn_sq_fixture(p0, p1, [0, 4], 0);
        let mut search = enabled_two_wall_search(game);
        search.race_proof = true;
        search.rp_build_ok = true;
        match search.one_side_broke_race_bound() {
            RaceBound::Upper(v) => {
                assert!(
                    v < -RACE_WIN_FLOOR && v >= -RACE_MATE,
                    "expected DTM-preserving Upper, got {v}"
                );
            }
            other => panic!("expected Upper, got {other:?} (p0={p0} p1={p1})"),
        }
        assert_eq!(search.race_outcome_stats.broke_decisive, 1);
    }

    #[test]
    fn broke_side_declines_when_wallless_player_wins_but_opp_still_has_walls() {
        // STM has 0 walls and wins pure race, but opp still has walls and can
        // spoil — must NOT emit Lower.
        let (p0, p1) = find_empty_board_pure_race_pair(true);
        let game = pawn_sq_fixture(p0, p1, [0, 3], 0);
        let mut search = enabled_two_wall_search(game);
        search.race_proof = true;
        search.rp_build_ok = true;
        assert_eq!(search.one_side_broke_race_bound(), RaceBound::Unknown);
        assert_eq!(search.race_outcome_stats.broke_unknown, 1);
    }

    #[test]
    fn broke_side_declines_when_walled_stm_loses_pure_race() {
        // STM has walls, opp has none, but STM loses pure race — STM may still
        // spend walls to reverse. Must NOT emit Upper (and must not Lower).
        let (p0, p1) = find_empty_board_pure_race_pair(false);
        let game = pawn_sq_fixture(p0, p1, [4, 0], 0);
        let mut search = enabled_two_wall_search(game);
        search.race_proof = true;
        search.rp_build_ok = true;
        assert_eq!(search.one_side_broke_race_bound(), RaceBound::Unknown);
        assert_eq!(search.race_outcome_stats.broke_unknown, 1);
    }

    #[test]
    fn broke_side_declines_when_both_armed_or_both_broke() {
        let (p0, p1) = find_empty_board_pure_race_pair(true);
        let both_armed = pawn_sq_fixture(p0, p1, [2, 2], 0);
        let mut search = enabled_two_wall_search(both_armed);
        search.race_proof = true;
        assert_eq!(search.one_side_broke_race_bound(), RaceBound::Unknown);
        assert_eq!(search.race_outcome_stats.broke_calls, 0);

        let both_broke = pawn_sq_fixture(p0, p1, [0, 0], 0);
        let mut search = enabled_two_wall_search(both_broke);
        search.race_proof = true;
        assert_eq!(search.one_side_broke_race_bound(), RaceBound::Unknown);
        assert_eq!(search.race_outcome_stats.broke_calls, 0);
    }

    #[test]
    #[ignore = "diagnostic counter-oracle: explicit release invocation"]
    fn two_wall_subset_random_empty_topology_counter_oracle() {
        let mut checked = 0usize;
        let mut ordinary_unresolved = 0usize;
        for p0 in [18usize, 27, 36, 45, 54] {
            for p1 in 63usize..72 {
                if p0 == p1 {
                    continue;
                }
                let mut game = two_wall_fixture("a7", "i2", [2, 0], 0);
                game.pawn = [p0, p1];
                game.hash_lo = crate::titanium::game::ZOBRIST.pawn_lo[0][p0]
                    ^ crate::titanium::game::ZOBRIST.pawn_lo[1][p1];
                game.hash_hi = crate::titanium::game::ZOBRIST.pawn_hi[0][p0]
                    ^ crate::titanium::game::ZOBRIST.pawn_hi[1][p1];

                let mut proof = enabled_two_wall_search(game.clone());
                let bound = proof.two_wall_monopoly_race_bound();
                if bound == RaceBound::Unknown {
                    continue;
                }

                let mut ordinary = TitaniumSearch::new(game);
                ordinary.two_wall_race_resolved = Some(false);
                let result = ordinary.think(30_000, 8, true, false, "titanium-v17");
                if result.score.abs() < MATE - 1_000 {
                    ordinary_unresolved += 1;
                    continue;
                }
                assert_eq!(
                    bound.signum(),
                    result.score.signum(),
                    "winner mismatch p0={p0} p1={p1}"
                );
                checked += 1;
            }
        }
        assert!(checked >= 10, "insufficient decisive samples: {checked}");
        eprintln!(
            "two-wall counter-oracle: {checked} independently proven states, 0 mismatches, {ordinary_unresolved} ordinary-search unknown"
        );
    }

    #[test]
    #[ignore = "diagnostic performance comparison: explicit release invocation"]
    fn two_wall_subset_ab_measurement() {
        let fixtures = [
            ("pure-win", two_wall_fixture("a7", "i4", [2, 0], 0), 8),
            ("forced-wall", two_wall_fixture("a7", "i2", [2, 0], 0), 7),
            ("reversed", two_wall_fixture("a8", "i3", [0, 2], 1), 7),
        ];
        for (name, game, depth) in fixtures {
            let mut baseline = TitaniumSearch::new(game.clone());
            baseline.two_wall_race_resolved = Some(false);
            let t0 = Instant::now();
            let baseline_result = baseline.think(30_000, depth, true, false, "titanium-v17");
            let baseline_ms = t0.elapsed().as_millis();

            let mut candidate = TitaniumSearch::new(game);
            candidate.two_wall_race_resolved = Some(true);
            let t1 = Instant::now();
            let candidate_result = candidate.think(30_000, depth, true, false, "titanium-v17");
            let candidate_ms = t1.elapsed().as_millis();

            assert_eq!(
                baseline_result.score.signum(),
                candidate_result.score.signum()
            );
            eprintln!(
                "two-wall AB {name}: depth={depth} baseline nodes={} ms={} candidate nodes={} ms={} calls={} decisive={} unknown={}",
                baseline_result.nodes,
                baseline_ms,
                candidate_result.nodes,
                candidate_ms,
                candidate_result.race_outcome_stats.two_wall_calls,
                candidate_result.race_outcome_stats.two_wall_decisive,
                candidate_result.race_outcome_stats.two_wall_unknown,
            );
        }
    }

    #[test]
    fn labels_true_mate_scores_separately_from_races() {
        assert_eq!(score_label(MATE - 5), "mate in 5");
        assert_eq!(score_label(-(MATE - 9)), "mated in 9");
        assert_eq!(score_label(42), "cp 42");
    }

    #[test]
    fn probcut_defaults_off_and_has_an_explicit_setter() {
        let mut search = TitaniumSearch::new(crate::titanium::game::GameState::new());
        assert!(!search.probcut);
        search.set_probcut(true);
        assert!(search.probcut);
    }

    #[test]
    fn sf_history_defaults_off_and_has_an_explicit_setter() {
        let mut search = TitaniumSearch::new(crate::titanium::game::GameState::new());
        assert!(!search.sf_history);
        search.set_sf_history(true);
        assert!(search.sf_history);
    }

    #[test]
    fn dense_history_codes_cover_all_move_ids() {
        for m in [0, 80, 81, 144, 145, 208] {
            assert!(is_pawn_move(m) || is_wall_move(m));
            assert_eq!(m as usize, m as usize);
        }
        for invalid in [-2, 209, 511] {
            assert!(!is_pawn_move(invalid) && !is_wall_move(invalid));
        }
    }

    #[test]
    fn sf_pawn_history_tiebreak_cannot_overturn_one_step_of_progress() {
        assert_eq!(sf_pawn_history_tiebreak(SF_HIST_MAX), 499);
        assert_eq!(sf_pawn_history_tiebreak(-SF_HIST_MAX), -499);
        assert_eq!(sf_pawn_history_tiebreak(i32::MAX), 499);
        assert_eq!(sf_pawn_history_tiebreak(i32::MIN), -499);

        let faster_with_worst_history = 1_000_000 - 5 * 1000 - 499;
        let slower_with_best_history = 1_000_000 - 6 * 1000 + 499;
        assert!(faster_with_worst_history > slower_with_best_history);
    }

    #[test]
    fn sf_history_switch_reads_pawn_destination_from_the_selected_table() {
        let mut search = TitaniumSearch::new(crate::titanium::game::GameState::new());
        let pawn_destination = 42i16;
        let pawn_hist = pawn_destination as usize;
        search.history_tbl[pawn_hist] = 17;
        search.hist_sf[0][pawn_hist] = -29;

        assert_eq!(search.move_hist(0, pawn_destination), 17);
        search.set_sf_history(true);
        assert_eq!(search.move_hist(0, pawn_destination), -29);
    }

    #[test]
    fn probcut_gate_requires_a_safe_non_root_fail_high_plausibility() {
        assert!(probcut_is_eligible(6, 99, 100, 1, true, 300));
        assert!(!probcut_is_eligible(5, 99, 100, 1, true, 300));
        assert!(!probcut_is_eligible(6, 99, 100, 0, true, 300));
        assert!(!probcut_is_eligible(6, 99, 100, 1, false, 300));
        assert!(!probcut_is_eligible(6, 98, 100, 1, true, 300));
        assert!(!probcut_is_eligible(
            6,
            PROBCUT_MAX_ABS_BETA - 1,
            PROBCUT_MAX_ABS_BETA,
            1,
            true,
            2_300
        ));
        assert!(!probcut_is_eligible(6, 99, 100, 1, true, 299));
    }

    #[test]
    #[ignore = "diagnostic: root defense pass requires proven-loss classification at W23"]
    fn w23_root_defense_fully_searches_all_pawns_and_picks_longest_loss() {
        use crate::titanium::algebraic_to_move_id;
        use crate::titanium::game::GameState;
        use crate::titanium::move_id_to_algebraic;

        let moves = [
            "e2", "e8", "e3", "e7", "e4", "e6", "a3h", "a6h", "e3h", "c3v", "c1h", "c6h", "e6v",
            "e4h", "d5h", "f5h", "d4", "d6", "c5v", "e6", "d7h", "e7", "b7h", "d7", "d3", "c7",
            "e3", "b7", "e2", "a7", "f2", "a8", "g2", "b8", "h2", "h2v", "h3h", "c8", "g2", "f2h",
            "d2h", "d8", "g1", "e8",
        ];
        let mut g = GameState::new();
        for m in moves {
            g.make_move(algebraic_to_move_id(m));
        }

        let mut search = TitaniumSearch::grafted_v17(g, Some(18));
        let result = search.think(60_000, 12, true, false, "titanium-v17");

        assert!(
            !result.root_defense_diag.is_empty(),
            "expected defense pass diagnostics"
        );
        let pawn_entries: Vec<_> = result
            .root_defense_diag
            .iter()
            .filter(|e| is_pawn_move(e.mv))
            .collect();
        assert_eq!(pawn_entries.len(), 3, "W23 has three pawn root moves");

        for entry in &pawn_entries {
            assert!(
                entry.full_depth_searched,
                "{} must be fully searched",
                move_id_to_algebraic(entry.mv)
            );
            assert_eq!(
                entry.child_depth_used,
                11,
                "{} childDepthUsed={} expected full defense depth",
                move_id_to_algebraic(entry.mv),
                entry.child_depth_used
            );
        }

        let f1 = pawn_entries
            .iter()
            .find(|e| move_id_to_algebraic(e.mv) == "f1")
            .expect("f1 in defense table");
        assert_ne!(
            f1.child_depth_used, 1,
            "f1 must not be LMR-reduced to depth 1"
        );
        assert!(
            f1.nodes > 1000,
            "f1 nodes={} expected full-depth search",
            f1.nodes
        );

        // Stubborn-loser policy (replaces pure DTM-maximization): among proven-loss
        // root moves, never prefer one that worsens our own distance-to-goal over
        // one that doesn't; among those tied on that, maximize the opponent's
        // distance-to-goal; only tie-break by sprinting (minimizing our own
        // distance) once the opponent's distance can't be improved on further.
        let loss_entries: Vec<&RootDefenseDiag> = result
            .root_defense_diag
            .iter()
            .filter(|e| is_proven_loss_score(e.search_score))
            .collect();
        assert!(
            !loss_entries.is_empty(),
            "expected at least one proven-loss root move"
        );

        let non_worsening: Vec<&RootDefenseDiag> = loss_entries
            .iter()
            .copied()
            .filter(|e| e.own_dist_after <= e.own_dist_before)
            .collect();
        let pool: &Vec<&RootDefenseDiag> = if non_worsening.is_empty() {
            &loss_entries
        } else {
            &non_worsening
        };
        let best_opp_dist = pool.iter().map(|e| e.opp_dist_after).max().unwrap();
        let best_own_dist = pool
            .iter()
            .filter(|e| e.opp_dist_after == best_opp_dist)
            .map(|e| e.own_dist_after)
            .min()
            .unwrap();

        if is_proven_loss_score(result.score) {
            let chosen = result
                .root_defense_diag
                .iter()
                .find(|e| e.mv == result.mv)
                .expect("selected move must be in the defense diag");
            if !non_worsening.is_empty() {
                assert!(
                    chosen.own_dist_after <= chosen.own_dist_before,
                    "must not select a move that worsens our own distance-to-goal when an alternative avoids it"
                );
            }
            assert_eq!(
                chosen.opp_dist_after, best_opp_dist,
                "selected move must maximize the opponent's distance-to-goal among non-worsening moves"
            );
            assert_eq!(
                chosen.own_dist_after, best_own_dist,
                "tie-break must sprint (minimize our own distance-to-goal)"
            );
        }
    }
}

/// Proven forced loss in the race or true-mate band.
#[inline]
pub fn is_proven_loss_score(score: i32) -> bool {
    if score >= 0 {
        return false;
    }
    let abs = score.abs();
    abs >= MATE - 1_000 || (abs > RACE_WIN_FLOOR && abs <= RACE_MATE)
}

/// Pack a 0/1 wall-slot byte array into a u64 bitboard (bit s = slot s occupied).
/// ACE wall slots only ever hold 0 or 1; eight bytes gather at a time via the
/// bit-0 multiply trick so the NNUE accumulator diff is a couple of XORs instead
/// of a 128-slot byte scan per eval.
#[inline]
fn wall_slot_bits(slots: &[u8; 64]) -> u64 {
    const LSB: u64 = 0x0101_0101_0101_0101;
    const GATHER: u64 = 0x0102_0408_1020_4080;
    let mut out = 0u64;
    let mut i = 0;
    while i < 8 {
        let w = u64::from_le_bytes(slots[i * 8..i * 8 + 8].try_into().unwrap());
        debug_assert!(w & !LSB == 0, "wall slot bytes must be 0/1");
        out |= ((w & LSB).wrapping_mul(GATHER) >> 56) << (i * 8);
        i += 1;
    }
    out
}

/// Proven forced win in the race or true-mate band.
#[inline]
pub fn is_proven_win_score(score: i32) -> bool {
    let abs = score.abs();
    (abs >= MATE - 1_000 || (abs > RACE_WIN_FLOOR && abs <= RACE_MATE)) && score > 0
}

/// Distance-to-mate plies encoded in a proven race/mate score.
#[inline]
pub fn proven_score_dtm(score: i32) -> Option<i32> {
    let abs = score.abs();
    if abs >= MATE - 1_000 {
        Some(MATE - abs)
    } else if abs > RACE_WIN_FLOOR && abs <= RACE_MATE {
        Some(RACE_MATE - abs)
    } else {
        None
    }
}

#[inline]
pub fn score_result_class(score: i32) -> &'static str {
    if is_proven_win_score(score) {
        if score.abs() >= MATE - 1_000 {
            "mate_win"
        } else {
            "race_win"
        }
    } else if is_proven_loss_score(score) {
        if score.abs() >= MATE - 1_000 {
            "mate_loss"
        } else {
            "race_loss"
        }
    } else {
        "cp"
    }
}

/// Selection key for lost-position root defense (higher = preferred). Loss
/// candidates rank by the stubborn-loser priorities (see
/// `better_defense_candidate`): never worsen own distance-to-goal, then
/// maximize the opponent's distance-to-goal, then sprint (minimize own
/// distance). This key is display-only (JSON diag); the real tie-break logic
/// lives in `better_defense_candidate`.
#[inline]
pub fn defense_selection_key(
    score: i32,
    static_eval: i32,
    own_worsens: bool,
    own_dist_after: i32,
    opp_dist_after: i32,
) -> i32 {
    if is_proven_loss_score(score) {
        let worsen_penalty = if own_worsens { -1_000_000 } else { 0 };
        -2_000_000 + worsen_penalty + opp_dist_after * 100 - own_dist_after
    } else if is_proven_win_score(score) {
        1_000_000 - proven_score_dtm(score).unwrap_or(0)
    } else {
        static_eval
    }
}

/// Stubborn-loser root move selection: when the root is a proven loss, never
/// pick a move that makes our own distance-to-goal worse than the position
/// already searched (no backward shuffling or wasted walls just because a
/// jump looks scary) -- among moves that hold our own distance, prefer
/// whichever maximizes the opponent's distance-to-goal (delay them when we
/// can), and if the opponent's distance can't be improved on, sprint (pick
/// whichever leaves us closest to our own goal).
#[inline]
#[allow(clippy::too_many_arguments)]
fn better_defense_candidate(
    score: i32,
    static_eval: i32,
    own_dist_before: i32,
    own_dist_after: i32,
    opp_dist_after: i32,
    order: usize,
    best_score: i32,
    best_static: i32,
    best_own_dist_after: i32,
    best_opp_dist_after: i32,
    best_order: usize,
) -> bool {
    if best_score == i32::MIN {
        return true;
    }
    let loss = is_proven_loss_score(score);
    let best_loss = is_proven_loss_score(best_score);
    if loss != best_loss {
        return !loss;
    }
    if loss {
        let worsens = own_dist_after > own_dist_before;
        let best_worsens = best_own_dist_after > own_dist_before;
        if worsens != best_worsens {
            return !worsens;
        }
        if opp_dist_after != best_opp_dist_after {
            return opp_dist_after > best_opp_dist_after;
        }
        if own_dist_after != best_own_dist_after {
            return own_dist_after < best_own_dist_after;
        }
    } else if score != best_score {
        return score > best_score;
    }
    if static_eval != best_static {
        return static_eval > best_static;
    }
    order < best_order
}

/// Clean-winner root move selection: when the root is a proven win and two
/// candidate moves score EXACTLY the same, prefer the one that doesn't waste
/// a tempo. Pawn moves make forced-win progress; a wall placement that
/// achieves the identical proven score is by definition unnecessary (the win
/// didn't need it), so it only adds branching complexity a human wouldn't
/// bother with. If both tied candidates are walls (no pawn move ties),
/// prefer whichever pushes the opponent's distance-to-goal further back —
/// the mirror of the stubborn-loser opponent-delay preference above, but for
/// making an already-won position more solid rather than a lost one less bad.
/// Next, prefer the move CAT's own impact-heat model rates as more
/// significant (the same attention signal that drives move ordering and LMR
/// elsewhere) -- among true ties this reads as "which move actually matters
/// here" rather than falling straight to arbitrary move-generation order.
/// Never overrides an actual score difference -- this only breaks EXACT ties.
#[inline]
#[allow(clippy::too_many_arguments)]
fn better_clean_win_candidate(
    score: i32,
    static_eval: i32,
    is_pawn_move: bool,
    opp_dist_after: i32,
    cat_heat: i32,
    order: usize,
    best_score: i32,
    best_static: i32,
    best_is_pawn_move: bool,
    best_opp_dist_after: i32,
    best_cat_heat: i32,
    best_order: usize,
) -> bool {
    if best_score == i32::MIN {
        return true;
    }
    if score != best_score {
        return score > best_score;
    }
    if is_pawn_move != best_is_pawn_move {
        return is_pawn_move;
    }
    if !is_pawn_move && opp_dist_after != best_opp_dist_after {
        return opp_dist_after > best_opp_dist_after;
    }
    if cat_heat != best_cat_heat {
        return cat_heat > best_cat_heat;
    }
    if static_eval != best_static {
        return static_eval > best_static;
    }
    order < best_order
}

#[derive(Debug, Clone)]
pub struct RootDefenseDiag {
    pub mv: i16,
    pub full_depth_searched: bool,
    pub child_depth_used: i32,
    pub result_class: &'static str,
    pub dtm: Option<i32>,
    pub search_score: i32,
    pub static_eval: i32,
    pub nodes: u64,
    pub selection_key: i32,
    pub own_dist_before: i32,
    pub own_dist_after: i32,
    pub opp_dist_after: i32,
}

pub fn format_root_defense_diag_json(entries: &[RootDefenseDiag]) -> String {
    let mut out = String::from("[");
    for (i, e) in entries.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        let mv = crate::titanium::move_id_to_algebraic(e.mv);
        let dtm = e
            .dtm
            .map(|v| v.to_string())
            .unwrap_or_else(|| "null".to_string());
        out.push_str(&format!(
            "{{\"move\":\"{mv}\",\"fullDepthSearched\":{},\"childDepthUsed\":{},\"resultClass\":\"{}\",\"dtm\":{dtm},\"searchScore\":{},\"staticEval\":{},\"nodes\":{},\"finalSelectionKey\":{},\"ownDistBefore\":{},\"ownDistAfter\":{},\"oppDistAfter\":{}}}",
            e.full_depth_searched,
            e.child_depth_used,
            e.result_class,
            e.search_score,
            e.static_eval,
            e.nodes,
            e.selection_key,
            e.own_dist_before,
            e.own_dist_after,
            e.opp_dist_after,
        ));
    }
    out.push(']');
    out
}

pub fn think_result_progress_json(
    engine_label: &str,
    result: &ThinkResult,
    root_scores: bool,
    multipv: usize,
) -> String {
    let json = ace_progress_json(
        engine_label,
        &result.depth_log,
        result.depth,
        result.nodes,
        result.main_thread_nodes,
        &result.helper_nodes,
        result.total_nodes,
        result.score,
        &result.root_moves,
        result.white_dist,
        result.black_dist,
        result.ms,
        root_scores,
        multipv,
    );
    append_race_result_json(json, result.race)
}

fn append_race_result_json(mut json: String, race: RaceResultInfo) -> String {
    let _ = json.pop();
    let kind = if race.exact_dtm.is_some() {
        "race_dtm"
    } else if race.outcome != 0 {
        "race_bound"
    } else {
        "score"
    };
    let approx = race
        .approximate_plies
        .map_or_else(|| "null".to_owned(), |v| v.to_string());
    let dtm = race
        .exact_dtm
        .map_or_else(|| "null".to_owned(), |v| v.to_string());
    json.push_str(&format!(
        r#","scoreKind":"{kind}","scoreProven":{},"raceOutcome":{},"estimatedPlies":{approx},"estimateTolerancePlies":{},"dtm":{dtm}}}"#,
        race.outcome != 0,
        race.outcome,
        race.approximation_tolerance,
    ));
    json
}

fn format_ranked_root_json(
    ordered: &[(i16, i32)],
    root_scores: bool,
    multipv: usize,
) -> (String, String) {
    let mut root_json = String::new();
    if root_scores {
        for (i, (mv, score)) in ordered.iter().enumerate() {
            if i > 0 {
                root_json.push(',');
            }
            let alg = crate::titanium::move_id_to_algebraic(*mv);
            root_json.push_str(&format!(
                r#"{{"move":"{}","score":{},"rank":{}}}"#,
                alg,
                score,
                i + 1
            ));
        }
    }
    let mut multipv_json = String::new();
    let multipv_count = multipv.min(ordered.len());
    for i in 0..multipv_count {
        if i > 0 {
            multipv_json.push(',');
        }
        let (mv, score) = ordered[i];
        let alg = crate::titanium::move_id_to_algebraic(mv);
        multipv_json.push_str(&format!(
            r#"{{"move":"{}","score":{},"rank":{}}}"#,
            alg,
            score,
            i + 1
        ));
    }
    (root_json, multipv_json)
}

fn ace_progress_json(
    engine_label: &str,
    depth_log: &[AceDepthLogEntry],
    search_depth: i32,
    nodes: u64,
    main_thread_nodes: u64,
    helper_nodes: &[u64],
    total_nodes: u64,
    root_score: i32,
    root_moves: &[(i16, i32)],
    white_dist: u8,
    black_dist: u8,
    elapsed_ms: u64,
    root_scores: bool,
    multipv: usize,
) -> String {
    let mut depth_json = String::new();
    for (i, e) in depth_log.iter().enumerate() {
        if i > 0 {
            depth_json.push(',');
        }
        const ESC_DQ: &str = "\\\"";
        let pv = e.pv.replace('\\', "\\\\").replace('"', ESC_DQ);
        let score_text = score_label(e.score);
        depth_json.push_str(&format!(
            r#"{{"depth":{},"score":{},"scoreText":"{}","nodes":{},"elapsedMs":{},"marginalNodes":{},"pv":"{}"}}"#,
            e.depth, e.score, score_text, e.nodes, e.elapsed_ms, e.marginal_nodes, pv
        ));
    }
    let mut helper_json = String::new();
    for (i, nodes) in helper_nodes.iter().enumerate() {
        if i > 0 {
            helper_json.push(',');
        }
        helper_json.push_str(&nodes.to_string());
    }
    let root_score_text = score_label(root_score);
    let mut ordered_root_moves = root_moves.to_vec();
    ordered_root_moves.sort_by(|a, b| b.1.cmp(&a.1));
    let (root_json, multipv_json) =
        format_ranked_root_json(&ordered_root_moves, root_scores, multipv);
    format!(
        r#"{{"engine":"{engine_label}","stoppedBy":"{engine_label}","searchDepth":{search_depth},"nodes":{nodes},"mainThreadNodes":{main_thread_nodes},"helperNodes":[{helper_json}],"totalNodes":{total_nodes},"totalNodesAcrossWorkers":{total_nodes},"rootScore":{root_score},"rootScoreText":"{root_score_text}","whiteDist":{white_dist},"blackDist":{black_dist},"elapsedMs":{elapsed_ms},"depthLog":[{depth_json}],"rootMoves":[{root_json}],"multiPv":[{multipv_json}]}}"#
    )
}

fn emit_ace_progress(
    engine_label: &str,
    depth_log: &[AceDepthLogEntry],
    search_depth: i32,
    nodes: u64,
    root_score: i32,
    root_moves: &[(i16, i32)],
    white_dist: u8,
    black_dist: u8,
    elapsed_ms: u64,
    root_scores: bool,
    multipv: usize,
    race: RaceResultInfo,
    #[cfg(feature = "wasm")] wasm_progress: Option<&mut Vec<String>>,
    #[cfg(feature = "wasm")] wasm_cb: Option<&js_sys::Function>,
) {
    let json = append_race_result_json(
        ace_progress_json(
            engine_label,
            depth_log,
            search_depth,
            nodes,
            nodes,
            &[],
            nodes,
            root_score,
            root_moves,
            white_dist,
            black_dist,
            elapsed_ms,
            root_scores,
            multipv,
        ),
        race,
    );
    #[cfg(feature = "wasm")]
    {
        // Real live streaming: call straight into JS the moment this depth's
        // progress is ready (same as ace::search::emit_ace_progress's f.call1
        // -- this is what actually makes a card update mid-think instead of
        // only at the end).
        if let Some(f) = wasm_cb {
            let _ = f.call1(
                &wasm_bindgen::JsValue::NULL,
                &wasm_bindgen::JsValue::from_str(&json),
            );
        }
        // Kept as a fallback/replay source for go_threads_json's JS wrapper;
        // harmless if the direct call above already delivered it live.
        if let Some(events) = wasm_progress {
            events.push(json);
        }
    }
    #[cfg(not(feature = "wasm"))]
    {
        eprintln!("info json {json}");
        let _ = std::io::Write::flush(&mut std::io::stderr());
    }
}

/// RaceProof race-table LRU slots (keyed by wall-config zobrist).
const RC_SLOTS: usize = 64;

/// Net-eval intermediates for Python parity tests (does not alter ``evaluate()``).
#[derive(Clone, Copy, Debug)]
pub struct EvalParityTrace {
    pub d_me: f64,
    pub d_opp: f64,
    pub w_me: f64,
    pub w_opp: f64,
    pub pd: f64,
    pub wd: f64,
    pub width_opp: f64,
    pub scalar_out: f64,
    pub route_out: f64,
    pub cat_out: f64,
    pub width_contrib: f64,
    pub wall_acc: [f64; MAX_NET_H],
    pub hidden_pre: [f64; MAX_NET_H],
    pub hidden_clip: [f64; MAX_NET_H],
    pub neural_out: f64,
    pub eval: i32,
}

pub struct TitaniumSearch {
    pub g: GameState,
    tt_key_hi: Vec<u32>,
    tt_key_lo: Vec<u32>,
    /// Packed TT word: `move(10) | flag<<10 | depth<<12` with depth in **8 bits**
    /// (0..=255). Higher bits of `meta` are unused / must be zero on store.
    /// `meta == 0` means empty.
    tt_meta: Vec<i32>,
    tt_score: Vec<i32>,
    // ZeroFence-A: 1 = tainted-zero entry (move-only, never a score cutoff)
    tt_rep: Vec<u8>,
    tt_anc_lo: Vec<u32>,
    tt_anc_hi: Vec<u32>,
    /// Generation counter — wraps; incremented every think(). Stored per TT slot.
    /// Depth-preferred replacement: within the same generation only deeper entries
    /// overwrite; entries from a prior generation are always replaced.
    tt_gen: u8,
    tt_entry_gen: Vec<u8>,
    /// Index mask for the TT vecs (`size - 1`). Runtime so the TT can be resized
    /// (Titanium-style larger table) without recompiling — `1<<TT_BITS` default.
    tt_mask: u32,
    /// Current TT index bits (`tt_mask == (1<<tt_bits)-1`).
    tt_bits: usize,
    /// Occupied slots (meta != 0). Drives overflow-triggered growth.
    tt_filled: usize,
    /// Overflow-driven cache-tier growth targets (Titanium strategy): start in L1,
    /// jump L1→L2→L3→d4(18)→d5(22) on overflow, then +1 per overflow past d5. Each
    /// jump lands on a calibrated size that won't immediately re-overflow. Inactive
    /// unless [`enable_adaptive_tt`](TitaniumSearch::enable_adaptive_tt) is called.
    tt_l2: usize,
    tt_l3: usize,
    tt_d4: usize,
    tt_d5: usize,
    tt_max: usize,
    tt_adaptive: bool,
    // per-ply open-subtree dependency window: min external path-rep target ply
    sub_min: [i32; MAX_PLY],
    sub_anc_lo: [u32; MAX_PLY],
    sub_anc_hi: [u32; MAX_PLY],
    history_tbl: [i32; HIST_SPAN],
    cm: [i16; HIST_SPAN], // countermove table, indexed by dense history code
    killers: [[i16; 2]; MAX_PLY],
    /// Stockfish-style history experiment (A/B flag, default off): side-split
    /// `[stm][move]` full-action history updated with the gravity formula
    /// (`h += bonus − h·|bonus|/MAX`, self-saturating) plus MALUSES — every
    /// action searched before the beta cutoff at a node gets the negative bonus,
    /// so ordering mistakes are demoted instead of keeping stale credit.
    /// When on, replaces `history_tbl` for full-action ordering reads (and wall
    /// pruning reads);
    /// `history_tbl` itself keeps updating unchanged so this stays a pure
    /// read-side experiment (single-axis A/B, same decay policy both modes).
    sf_history: bool,
    /// Conservative reduced-depth fail-high verification. Off by default and
    /// exposed only as an explicit A/B experiment.
    probcut: bool,
    hist_sf: [[i32; HIST_SPAN]; 2],
    /// SF batch 2 (branch build, unconditional): corrected static eval per
    /// ply, for the `improving` flag (i32::MIN = never written).
    eval_stack: [i32; MAX_PLY],
    /// Correction history: `[stm][wall-structure-hash]` → running EMA of
    /// (search score − static eval) in cp, clamped ±256. Teaches the static
    /// eval its systematic bias per wall structure, online, no training.
    /// Applied only in the net-eval band — cert/mate scores never corrected.
    corr_hist: [[i16; CORR_SIZE]; 2],
    /// Continuation history: flat `[prev_hist * HIST_SPAN + move_hist]` gravity table —
    /// scores wall replies to the previous move (the SF conthist analog; our
    /// `cm` table keeps only the single best reply, this keeps a full score
    /// surface). Heap Vec: HIST_SPAN*HIST_SPAN*4B = 256KiB.
    cont_hist: Vec<i32>,
    // Offline-only LMR counterfactual probe. A target ordinal receives exactly
    // one provisional extra reduction; verification always uses native depth.
    reduction_probe_enabled: bool,
    reduction_probe_target: Option<u64>,
    reduction_probe_next: u64,
    reduction_probe_limit: usize,
    reduction_probe_min_depth: i32,
    reduction_probe_events: Vec<ReductionProbeEvent>,
    reduction_sidecar: Option<ReductionSidecar>,
    reduction_shadow_stats: ReductionShadowStats,
    path_lo: [u32; MAX_PLY],
    path_hi: [u32; MAX_PLY],
    d0: [[u8; 81]; MAX_PLY],
    d1: [[u8; 81]; MAX_PLY],
    d0_layers: [[u128; 81]; MAX_PLY],
    d1_layers: [[u128; 81]; MAX_PLY],
    d0_layer_depth: [usize; MAX_PLY],
    /// Wall-topology zobrist (`wall_topology_key`, hi<<32|lo) of the fields last
    /// written into each ply slot. After an unmake the parent's fields are still
    /// in its slot — a key match restores them instead of re-flooding. u64::MAX =
    /// never written.
    d0_key: [u64; MAX_PLY],
    d1_key: [u64; MAX_PLY],
    /// Static-eval cache: the NNUE + scalar + route portion of `evaluate` is a
    /// pure function of (position hash, walls-remaining) for a fixed net, and the
    /// search transposes heavily. Cert/race floors are applied AFTER the cached
    /// value so their budgeted/stateful behavior is untouched. Direct-mapped;
    /// key 0 with meta u16::MAX = empty. wl is verified separately because the
    /// zobrist hash does not encode per-player wall counts.
    eval_cache: Vec<EvalCacheEntry>,
    /// Current eval_cache index bits (`eval_cache.len() == 1 << eval_cache_bits`
    /// except helpers, which use a 1-slot stub with bits = 0).
    eval_cache_bits: usize,
    dist_lru: Vec<DistTopoEntry>,
    /// Current dist_lru index bits (`dist_lru.len() == 1 << dist_lru_bits`).
    dist_lru_bits: usize,
    /// Occupied slots; growth trigger at 50% like `tt_filled`/`tt_grow`.
    dist_lru_filled: usize,
    /// When false, `dist_lru_store` never grows (LazySMP helpers keep a 1-slot stub).
    dist_lru_growable: bool,
    /// Rare deep-layer tails keyed by wall-topology `wkey`. Hot DistTopoEntry
    /// has no spill_id — depth > INLINE is the only deep marker.
    dist_layer_spill: HashMap<u64, DistLayerSpill>,
    d1_layer_depth: [usize; MAX_PLY],
    dist0_idx: usize, // active ply slot in d0 (JS: this.dist0 array ref)
    dist1_idx: usize,
    cached_stamp: i32,
    /// After `cat_path_lmr` refresh at `ply+1`, child `ab(ply+1)` may reuse it.
    pending_cat_child_ply: Option<usize>,
    ab_after_cat_child: bool,
    dir_masks_key_lo: u32,
    dir_masks_key_hi: u32,
    dir_masks_cache: DirMasks,
    // HalfPW accumulator cache
    np_acc0: [f64; MAX_NET_H],
    np_acc1: [f64; MAX_NET_H],
    np_hbits: u64,
    np_vbits: u64,
    np_b0: i32,
    np_b1v: i32,
    net: &'static Net,
    /// Mirrored Titanium board (movegen and/or CAT).
    bridge: Option<Box<TiBridge>>,
    /// Use Titanium `generate_legal_moves_slice` instead of ACE `wall_legal`.
    ti_movegen: bool,
    /// CAT-filter walls at inner nodes (requires `bridge`).
    cat_walls: bool,
    /// Historical field name for the production v17 CAT-graduated LMR.
    cat_lmr_v16: bool,
    cat_lmr_ceiling: u16,
    cat_lmr_fringe_pct: u16,
    /// Experimental: small ordering bonus for walls touching either
    /// player's shortest-route cell set (see ROUTE_TOUCH_ORDER_BONUS).
    /// Off by default; only takes effect alongside cat_lmr_v16.
    route_touch_ordering: bool,
    /// Opt-in CAT path-aware correction for the wall-LMR branch only. The
    /// correction can only reduce v16's existing reduction by one ply.
    cat_path_lmr: bool,
    /// Skip CAT `refresh_dist` when the wall cuts neither shortest-path edge.
    /// Enabled with `cat_path_lmr` on v17 after oracle + parity validation.
    cat_no_edge_skip: bool,
    /// Ka-AB-style horizon quiescence: extend one ply at depth<=0 when a wall
    /// fight or jump race is tactically noisy. Off by default.
    q_search: bool,
    q_max: i32,
    q_swing_cp: i32,
    /// Opt-in Lazy SMP role for the last worker: search only the true top-N
    /// ordered root moves. Off preserves the v17 worker schedule exactly.
    lazy_topn: bool,
    /// Opt-in ACE-style frontier LMP candidate. Off preserves the v17
    /// depth<=2/index-threshold policy exactly.
    ace_lmp: bool,
    /// Predictive iterative-deepening stop before starting a depth that is
    /// unlikely to fit the remaining time budget.
    use_predict_stop: bool,
    /// Opt-in ACE-style reverse futility pruning candidate. When enabled, RFP
    /// is limited to depth <= 3 with a fixed 100 cp/depth margin.
    ace_rfp: bool,
    /// At roots allotted at most 200 ms, allow the ACE RFP rule at depth four
    /// on null-window nodes. Fixed-depth and longer searches remain depth three.
    rfp_tc_adaptive: bool,
    ace_rfp_max_depth: i32,
    /// SOUND dead-zone wall prune at inner nodes (requires `bridge`): drop only
    /// walls in an unreachable void / sealed interior — provably irrelevant (they
    /// change no path and only burn inventory, never the best move). NPS-only;
    /// cannot cost Elo. Distinct from `cat_walls` (heat filter, which can).
    dead_zone_prune: bool,
    /// Grafted-engine flag: in the hands-empty endgame, use Titanium's cheap
    /// path-aware tempo classifier ([`cert_bridge::hands_empty_race`]) instead of
    /// the full recursive `certify`. Same result, a fraction of the nodes — frees
    /// NPS for the rest of the search. Off = faithful gen13 (always `certify`).
    cheap_cert: bool,
    /// When true, recursive certify + k=0 race oracle run only at quiescence
    /// leaves with both hands empty. Inner nodes use the HalfPW net (search
    /// + EME resolve tempo ambiguity). Set in [`Self::grafted_with_weights`].
    cert_eval_leaves_only: bool,
    /// Override for experimental wall-ignorance certificate (`None` = env only).
    wall_ignore_cert_override: Option<bool>,
    /// Lazy per-instance cache of the resolved override-or-env decision (`None`
    /// = not yet resolved). `std::env::var` is a heap-allocating, lock-adjacent
    /// syscall — reading it on every `evaluate_tail` call (once per node where
    /// walls remain) measured as a flat ~400ns/call tax, independent of
    /// position. Env vars don't change mid-process for this flag in any real
    /// caller, so resolving once per instance is safe; `wall_ignore_loss_cert_enabled()`
    /// itself is left untouched (other callers, e.g. the alphabeta.rs test that
    /// toggles this env var at runtime, keep reading it fresh).
    wall_ignore_cert_resolved: Option<bool>,
    /// Cached `TITANIUM_RACE_ONE_WALL` decision.
    one_wall_race_resolved: Option<bool>,
    /// Restrict the one-wall proof to PV/full-window nodes.
    one_wall_race_pv_only: bool,
    /// Cached `TITANIUM_RACE_TWO_WALL` decision. The experiment is deliberately
    /// default-off until its proof audit and strength gate both pass.
    two_wall_race_resolved: Option<bool>,
    /// Restrict the optional two-wall proof to PV/full-window nodes. Race1
    /// remains available throughout the tree.
    two_wall_race_pv_only: bool,
    /// Early Move Extensions on the first ordered wall moves (mirror of graduated LMR).
    eme: bool,
    pub nodes: u64,
    deadline: Instant,
    root_best: i16,
    root_score: i32,
    /// Lague partial-iteration: on time-abort, adopt the best FULLY-searched
    /// root move from the unfinished deepest iteration instead of discarding it.
    use_partial_iter: bool,
    /// Pure-JS-port mode: disables all Rust-side state-retention extras
    /// (gen TT, history aging, dynamic ID startup, accumulator retention).
    /// Use with `ti_movegen=true` as the fair baseline opponent.
    pure_mode: bool,
    /// Ponder mode: suppresses tt_gen advance and history decay so all ponder
    /// chunks share one TT generation and history accumulates uninterrupted.
    /// Set true before the ponder loop, false before the real think() call.
    is_pondering: bool,
    // ---------- pathfix feature flags (gen11 shipping config) ----------
    /// Exact k=0 race endgame + last-wall gate (JS `raceProof`, ships true).
    race_proof: bool,
    // RaceProof: race-table LRU (keyed by wall-config zobrist = hash sans pawn/turn)
    rc_key_lo: [u32; RC_SLOTS],
    rc_key_hi: [u32; RC_SLOTS],
    rc_tbl: Vec<Option<Box<[i16]>>>,
    rc_use: [u64; RC_SLOTS],
    rc_tick: u64,
    rc_last: i32,
    rc_build_ms: u64,
    rc_hits: u64,
    rc_solves: u64,
    rc_think_solve_ms: u64,
    rc_solve_cap: f64,
    rc_blocked: bool,
    rc_miss_lo: u32,
    rc_miss_hi: u32,
    rc_think_solves: u32,
    /// deterministic per-think in-tree solve cap (LRU holds 64: stops config-thrash)
    rc_count_cap: u32,
    rp_build_ok: bool,
    rp_root_empty: bool,
    pub rp_demotions: u64,
    pub rp_root_solves: u64,
    /// -1 sentinel: cell 0 (a1) is a legal pawn-move id
    root_pawn_best: i16,
    root_pawn_score: i32,
    /// Lost-position root defense diagnostics from the latest verification pass.
    root_defense_diag: Vec<RootDefenseDiag>,
    race_scratch: Option<Box<RaceScratch>>,
    race_outcome_stats: RaceOutcomeStats,
    // RaceProof(c) certificate memo. Key = (lo, hi, side, wl0, wl1);
    // value = 1 proven (permanent, sound) / -work for a failure (richer retries
    // re-run; weaker-or-equal retries inherit the false).
    cw_cache: std::collections::HashMap<(u32, u32, usize, i32, i32), i32, FxBuildHasher>,
    cw_think_calls: u32,
    cw_cap: u32,
    /// Live `info json` during `think(..., log=true)` — cleared when search ends.
    stream_log: bool,
    stream_label: String,
    stream_t0: Instant,
    stream_root_score: i32,
    stream_root_moves: Vec<(i16, i32)>,
    /// Top-N root lines to expose in progress JSON (`multiPv`). Open-window root
    /// search when `> 1` so scores are comparable.
    multipv: usize,
    /// When true, emit ranked `rootMoves` in progress JSON (not full MultiPV PV lines).
    root_scores: bool,
    stream_search_depth: i32,
    stream_depth_log: Vec<AceDepthLogEntry>,
    stream_last_emit_nodes: u64,
    stream_last_emit_ms: u64,
    stream_last_best: i16,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    shared_tt: Option<Arc<SharedTitaniumTt>>,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    lazy_runtime: Option<Arc<LazySmpRuntime>>,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    lazy_root_moves: Option<Arc<Vec<i16>>>,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    lazy_root_visit_map: Option<Arc<Vec<usize>>>,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    lazy_root_allowed: usize,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    lazy_worker_id: usize,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    lazy_skip_setup: bool,
    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    lazy_root_visits: Vec<usize>,
    opening_book: Option<std::sync::Arc<crate::titanium::opening_book::OpeningBook>>,
    opening_book_mode: crate::titanium::opening_book::OpeningBookMode,
    opening_book_order: Option<Vec<i16>>,
    opening_book_attention: Option<Vec<i32>>,
    pending_opening_book_diag: Option<crate::titanium::opening_book::OpeningBookDiagnostics>,
    /// GitHub Pages: progress payloads buffered until wasm-bindgen releases the
    /// exported `&mut self` borrow. Retained as a fallback/replay path; real
    /// live streaming now goes through `wasm_progress_cb` below (see it for
    /// why the buffer-then-replay-after-think() approach alone left the site's
    /// info cards frozen for the whole think, updating only once at the end).
    #[cfg(feature = "wasm")]
    wasm_progress: Vec<String>,
    /// Direct JS callback invoked from `emit_stream_progress` during the
    /// search itself (same pattern as `ace::search::AceSearch::wasm_progress`,
    /// which is what makes ACE v13's card update live while Titanium's sat
    /// frozen on "Thinking..." for the entire think -- Titanium had the
    /// periodic emit hook (`check_time` -> `emit_stream_progress`, every
    /// ~64K nodes / 100ms) but never actually invoked a JS function from it,
    /// only buffered JSON strings for the `go_threads_json` JS wrapper to
    /// replay in a burst after the whole blocking search call returned.
    #[cfg(feature = "wasm")]
    wasm_progress_cb: Option<js_sys::Function>,
}

#[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
unsafe impl Send for TitaniumSearch {}

/// Periodic progress cadence: every 64K nodes AND ≥ 100ms apart — stdout/stderr
/// writes are expensive; spamming them steals think time from the search.
const STREAM_EMIT_NODE_MASK: u64 = 65535;
const STREAM_EMIT_MIN_INTERVAL_MS: u64 = 100;

enum HandsEmptyPipelineOutcome {
    Score(i32),
}

const RACE_APPROX_TOLERANCE_PLIES: u16 = 1;

#[derive(Clone, Copy)]
struct RaceRootCandidate {
    mv: i16,
    root_wins: bool,
    approximate_plies: Option<u16>,
    exact_dtm: Option<u16>,
}

struct RaceRootSolution {
    mv: i16,
    score: i32,
    info: RaceResultInfo,
    exact: bool,
    legal_moves: usize,
}

#[inline]
fn race_estimate_interval(plies: u16) -> (u16, u16) {
    (
        plies.saturating_sub(RACE_APPROX_TOLERANCE_PLIES),
        plies.saturating_add(RACE_APPROX_TOLERANCE_PLIES),
    )
}

fn race_candidate_definitely_best(
    candidate: RaceRootCandidate,
    peers: &[RaceRootCandidate],
) -> bool {
    let Some(estimate) = candidate.approximate_plies else {
        return false;
    };
    let (lo, hi) = race_estimate_interval(estimate);
    peers.iter().all(|peer| {
        if peer.mv == candidate.mv {
            return true;
        }
        let Some(peer_estimate) = peer.approximate_plies else {
            return false;
        };
        let (peer_lo, peer_hi) = race_estimate_interval(peer_estimate);
        if candidate.root_wins {
            hi < peer_lo
        } else {
            lo > peer_hi
        }
    })
}

impl TitaniumSearch {
    pub fn new(g: GameState) -> Box<Self> {
        Box::new(Self {
            g,
            tt_key_hi: vec![0; TT_SIZE],
            tt_key_lo: vec![0; TT_SIZE],
            tt_meta: vec![0; TT_SIZE],
            tt_score: vec![0; TT_SIZE],
            tt_rep: vec![0; TT_SIZE],
            tt_anc_lo: vec![0; TT_SIZE],
            tt_anc_hi: vec![0; TT_SIZE],
            tt_gen: 0,
            tt_entry_gen: vec![0; TT_SIZE],
            tt_mask: TT_MASK,
            tt_bits: TT_BITS,
            tt_filled: 0,
            // Defaults overwritten by enable_adaptive_tt(); harmless when inactive.
            tt_l2: TT_BITS,
            tt_l3: TT_BITS,
            tt_d4: 18,
            tt_d5: 22,
            tt_max: 25,
            tt_adaptive: false,
            sub_min: [MAX_PLY as i32; MAX_PLY],
            sub_anc_lo: [0; MAX_PLY],
            sub_anc_hi: [0; MAX_PLY],
            history_tbl: [0; HIST_SPAN],
            cm: [0; HIST_SPAN],
            // Stockfish-style history is an explicit A/B experiment. Production
            // Titanium stays on the legacy history table unless a caller enables
            // the experiment with `set_sf_history(true)`.
            sf_history: false,
            probcut: false,
            hist_sf: [[0; HIST_SPAN]; 2],
            eval_stack: [i32::MIN; MAX_PLY],
            corr_hist: [[0; CORR_SIZE]; 2],
            cont_hist: vec![0; HIST_SPAN * HIST_SPAN],
            killers: [[0; 2]; MAX_PLY],
            reduction_probe_enabled: false,
            reduction_probe_target: None,
            reduction_probe_next: 0,
            reduction_probe_limit: 0,
            reduction_probe_min_depth: 0,
            reduction_probe_events: Vec::new(),
            reduction_sidecar: None,
            reduction_shadow_stats: ReductionShadowStats::default(),
            path_lo: [0; MAX_PLY],
            path_hi: [0; MAX_PLY],
            d0: [[0; 81]; MAX_PLY],
            d1: [[0; 81]; MAX_PLY],
            d0_layers: [[0; 81]; MAX_PLY],
            d1_layers: [[0; 81]; MAX_PLY],
            d0_layer_depth: [0; MAX_PLY],
            d0_key: [u64::MAX; MAX_PLY],
            d1_key: [u64::MAX; MAX_PLY],
            eval_cache: vec![EvalCacheEntry::default(); EVAL_CACHE_SIZE],
            eval_cache_bits: EVAL_CACHE_BITS,
            dist_lru: vec![DistTopoEntry::default(); 1 << DIST_LRU_MIN_BITS],
            dist_lru_bits: DIST_LRU_MIN_BITS,
            dist_lru_filled: 0,
            dist_lru_growable: true,
            dist_layer_spill: HashMap::new(),
            d1_layer_depth: [0; MAX_PLY],
            dist0_idx: 0,
            dist1_idx: 0,
            cached_stamp: -1,
            pending_cat_child_ply: None,
            ab_after_cat_child: false,
            dir_masks_key_lo: u32::MAX,
            dir_masks_key_hi: u32::MAX,
            dir_masks_cache: DirMasks::default(),
            np_acc0: [0.0; MAX_NET_H],
            np_acc1: [0.0; MAX_NET_H],
            np_hbits: 0,
            np_vbits: 0,
            np_b0: -1,
            np_b1v: -1,
            net: net(),
            bridge: None,
            ti_movegen: false,
            cat_walls: false,
            cat_lmr_v16: false,
            cat_lmr_ceiling: crate::cat::CAT_V16_LMR_CEILING_DEFAULT,
            cat_lmr_fringe_pct: crate::cat::CAT_V16_FRINGE_PCT_DEFAULT,
            route_touch_ordering: false,
            cat_path_lmr: false,
            cat_no_edge_skip: false,
            q_search: false,
            q_max: Q_SEARCH_MAX_DEFAULT,
            q_swing_cp: Q_SWING_CP_DEFAULT,
            lazy_topn: false,
            ace_lmp: false,
            use_predict_stop: true,
            ace_rfp: false,
            rfp_tc_adaptive: false,
            ace_rfp_max_depth: 3,
            dead_zone_prune: false,
            cheap_cert: false,
            cert_eval_leaves_only: false,
            wall_ignore_cert_override: None,
            wall_ignore_cert_resolved: None,
            one_wall_race_resolved: None,
            one_wall_race_pv_only: false,
            two_wall_race_resolved: None,
            two_wall_race_pv_only: false,
            eme: false,
            nodes: 0,
            deadline: Instant::now(),
            root_best: crate::titanium::TITANIUM_NO_MOVE,
            root_score: 0,
            use_partial_iter: true,
            pure_mode: false,
            is_pondering: false,
            race_proof: true,
            rc_key_lo: [0; RC_SLOTS],
            rc_key_hi: [0; RC_SLOTS],
            rc_tbl: (0..RC_SLOTS).map(|_| None).collect(),
            rc_use: [0; RC_SLOTS],
            rc_tick: 0,
            rc_last: -1,
            rc_build_ms: 6,
            rc_hits: 0,
            rc_solves: 0,
            rc_think_solve_ms: 0,
            rc_solve_cap: f64::INFINITY,
            rc_blocked: false,
            rc_miss_lo: 0,
            rc_miss_hi: 0,
            rc_think_solves: 0,
            rc_count_cap: 48,
            rp_build_ok: false,
            rp_root_empty: false,
            rp_demotions: 0,
            rp_root_solves: 0,
            root_pawn_best: -1,
            root_pawn_score: i32::MIN,
            root_defense_diag: Vec::new(),
            race_scratch: None,
            race_outcome_stats: RaceOutcomeStats::default(),
            cw_cache: std::collections::HashMap::default(),
            cw_think_calls: 0,
            cw_cap: 24,
            stream_log: false,
            stream_label: String::new(),
            stream_t0: Instant::now(),
            stream_root_score: 0,
            stream_root_moves: Vec::new(),
            multipv: 1,
            root_scores: true,
            stream_search_depth: 0,
            stream_depth_log: Vec::new(),
            stream_last_emit_nodes: 0,
            stream_last_emit_ms: 0,
            stream_last_best: 0,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            shared_tt: None,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            lazy_runtime: None,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            lazy_root_moves: None,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            lazy_root_visit_map: None,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            lazy_root_allowed: usize::MAX,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            lazy_worker_id: 0,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            lazy_skip_setup: false,
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            lazy_root_visits: Vec::new(),
            opening_book: None,
            opening_book_mode: crate::titanium::opening_book::OpeningBookMode::Off,
            opening_book_order: None,
            opening_book_attention: None,
            pending_opening_book_diag: None,
            #[cfg(feature = "wasm")]
            wasm_progress: Vec::new(),
            #[cfg(feature = "wasm")]
            wasm_progress_cb: None,
        })
    }

    #[cfg(feature = "wasm")]
    pub fn set_wasm_progress(&mut self, cb: Option<js_sys::Function>) {
        self.wasm_progress_cb = cb;
    }

    /// Clear browser progress events before a new exported WASM search starts.
    #[cfg(feature = "wasm")]
    pub fn clear_wasm_progress(&mut self) {
        self.wasm_progress.clear();
    }

    /// Queue one browser progress event while the exported WASM method still
    /// owns `&mut self`; JS drains the buffer after the method returns.
    #[cfg(feature = "wasm")]
    pub fn queue_wasm_progress(&mut self, json: String) {
        self.wasm_progress.push(json);
    }

    /// Take buffered browser progress events after a WASM search returns.
    #[cfg(feature = "wasm")]
    pub fn take_wasm_progress(&mut self) -> Vec<String> {
        std::mem::take(&mut self.wasm_progress)
    }

    /// Enable Early Move Extensions — same gates/tuning as graduated LMR, early indices.
    pub fn enable_eme(&mut self) {
        self.eme = true;
    }

    pub fn enable_route_touch_ordering(&mut self) {
        self.route_touch_ordering = true;
    }

    pub fn enable_cat_path_lmr(&mut self) {
        self.cat_path_lmr = true;
    }

    pub fn enable_cat_no_edge_skip(&mut self) {
        self.cat_no_edge_skip = true;
    }

    pub fn cat_no_edge_skip_enabled(&self) -> bool {
        self.cat_no_edge_skip
    }

    pub fn cat_path_lmr_enabled(&self) -> bool {
        self.cat_path_lmr
    }

    /// Enable bounded horizon quiescence (wall-fight / jump-race extensions).
    pub fn enable_q_search(&mut self) {
        self.q_search = true;
        self.q_max = Q_SEARCH_MAX_DEFAULT;
        self.q_swing_cp = Q_SWING_CP_DEFAULT;
    }

    pub fn route_touch_ordering_enabled(&self) -> bool {
        self.route_touch_ordering
    }

    pub fn q_search_enabled(&self) -> bool {
        self.q_search
    }

    pub fn enable_lazy_topn(&mut self) {
        self.lazy_topn = true;
    }

    pub fn lazy_topn_enabled(&self) -> bool {
        self.lazy_topn
    }

    pub fn set_ace_lmp(&mut self, on: bool) {
        self.ace_lmp = on;
    }

    /// Report top-N root moves by score in progress JSON. When `> 1`, root moves
    /// are searched with an open window so scores are comparable.
    pub fn set_multipv(&mut self, n: u32) {
        self.multipv = n.max(1) as usize;
    }

    pub fn multipv(&self) -> usize {
        self.multipv
    }

    /// Dump searched root moves with score+rank in progress JSON; not MultiPV PV lines.
    pub fn set_root_scores(&mut self, enabled: bool) {
        self.root_scores = enabled;
    }

    pub fn root_scores_enabled(&self) -> bool {
        self.root_scores
    }

    pub fn ace_lmp_enabled(&self) -> bool {
        self.ace_lmp
    }

    pub fn set_ace_rfp(&mut self, on: bool) {
        self.ace_rfp = on;
    }

    pub fn set_rfp_tc_adaptive(&mut self, on: bool) {
        self.rfp_tc_adaptive = on;
    }

    /// Select the proven remaining-wall race layers used by an engine variant.
    pub fn set_remaining_wall_race_layers(&mut self, one_wall: bool, two_wall: bool) {
        self.one_wall_race_resolved = Some(one_wall);
        self.two_wall_race_resolved = Some(two_wall);
    }

    pub fn set_two_wall_race_pv_only(&mut self, on: bool) {
        self.two_wall_race_pv_only = on;
    }

    pub fn set_one_wall_race_pv_only(&mut self, on: bool) {
        self.one_wall_race_pv_only = on;
    }

    #[cfg(test)]
    pub fn remaining_wall_race_layers(&self) -> (bool, bool) {
        (
            self.one_wall_race_resolved.unwrap_or(false),
            self.two_wall_race_resolved.unwrap_or(false),
        )
    }

    #[cfg(test)]
    pub fn two_wall_race_pv_only(&self) -> bool {
        self.two_wall_race_pv_only
    }

    #[cfg(test)]
    pub fn one_wall_race_pv_only(&self) -> bool {
        self.one_wall_race_pv_only
    }

    pub fn ace_rfp_enabled(&self) -> bool {
        self.ace_rfp
    }

    #[cfg(test)]
    pub fn rfp_tc_adaptive_enabled(&self) -> bool {
        self.rfp_tc_adaptive
    }

    pub fn set_opening_book(
        &mut self,
        mode: crate::titanium::opening_book::OpeningBookMode,
        db_path: Option<std::path::PathBuf>,
    ) {
        use crate::titanium::opening_book::{OpeningBook, OpeningBookMode};
        self.opening_book_mode = mode;
        self.opening_book_order = None;
        self.opening_book_attention = None;
        self.pending_opening_book_diag = None;
        if mode == OpeningBookMode::Off {
            self.opening_book = None;
            return;
        }
        self.opening_book = OpeningBook::open(db_path.as_deref()).ok();
    }

    fn prepare_opening_book_at_root(&mut self) -> Option<i16> {
        use crate::titanium::opening_book::OpeningBookMode;
        self.opening_book_order = None;
        self.opening_book_attention = None;
        self.pending_opening_book_diag = None;
        if self.opening_book_mode == OpeningBookMode::Off {
            return None;
        }
        let Some(book) = self.opening_book.clone() else {
            return None;
        };
        let mut legal = [0i16; 160];
        let n = self.gen_moves(0, 1, 0, &mut legal);
        let n = crate::titanium::opening_book::filter_denied_opening_legal_moves(
            &self.g, &mut legal, n,
        );
        let consult = book.consult(&self.g, self.opening_book_mode, &legal[..n]);
        self.pending_opening_book_diag = Some(consult.diagnostics);
        if !consult.order.is_empty() {
            self.opening_book_order = Some(consult.order);
            self.opening_book_attention = Some(consult.order_attention);
        }
        consult.direct_play
    }

    /// Enable offline observation of complete native LMR move pipelines.
    /// `target=None` records baseline events; `Some(n)` applies +1 only to event n.
    /// `min_depth` skips events at local depth < min_depth so shallow-tree events
    /// (which dominate post-order traversal) do not fill the limit before useful ones.
    pub fn enable_reduction_probe(&mut self, target: Option<u64>, limit: usize, min_depth: i32) {
        self.reduction_probe_enabled = true;
        self.reduction_probe_target = target;
        self.reduction_probe_next = 0;
        self.reduction_probe_limit = limit;
        self.reduction_probe_min_depth = min_depth;
        self.reduction_probe_events.clear();
    }

    pub fn reduction_probe_events(&self) -> &[ReductionProbeEvent] {
        &self.reduction_probe_events
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn enable_reduction_shadow(&mut self, path: &std::path::Path) -> Result<(), String> {
        self.reduction_sidecar = Some(ReductionSidecar::load(path)?);
        self.reduction_shadow_stats = ReductionShadowStats::default();
        Ok(())
    }

    pub fn reduction_shadow_stats(&self) -> ReductionShadowStats {
        self.reduction_shadow_stats
    }

    /// Titanium movegen on a mirrored board — same legal set, much faster than `wall_legal`.
    pub fn with_ti_movegen(g: GameState) -> Box<Self> {
        let mut search = Self::new(g);
        search.bridge = Some(TiBridge::from_game(&search.g));
        search.ti_movegen = true;
        search
    }

    /// ACE v13 reference tier with Titanium movegen acceleration, pinned to the
    /// frozen HalfPW blob used by the JS reference instead of live training weights.
    pub fn with_ti_movegen_frozen(g: GameState) -> Box<Self> {
        let mut search = Self::with_ti_movegen(g);
        search.net = net_frozen();
        search
    }

    /// Pure JS-port baseline + O1 movegen only. Uses **frozen** v13 HalfPW weights
    /// (`net_weights_frozen.bin`) — never picks up live training/deploy updates.
    pub fn with_ti_movegen_pure(g: GameState) -> Box<Self> {
        let mut search = Self::with_ti_movegen_frozen(g);
        search.pure_mode = true;
        search
    }

    /// CAT hybrid: walls at inner nodes must pass `wall_should_search`.
    pub fn with_cat(g: GameState) -> Box<Self> {
        let mut search = Self::new(g);
        search.bridge = Some(TiBridge::from_game(&search.g));
        search.cat_walls = true;
        search
    }

    /// Fast Titanium movegen + CAT wall filter.
    pub fn with_ti_movegen_and_cat(g: GameState) -> Box<Self> {
        let mut search = Self::with_ti_movegen(g);
        search.cat_walls = true;
        search
    }

    /// **Grafted engine** — gen13 net search + Titanium's *logically-safe* extras:
    ///   - cheap hands-empty cert: replaces the recursive `certify` with the exact
    ///     race classifier when no walls remain — IDENTICAL verdict, fewer nodes.
    ///     A strict non-regression (can't produce a worse move; frees NPS).
    ///   - adaptive cache-tier TT: identical TT semantics, better cache locality
    ///     and safe growth. Also can't hurt.
    ///
    /// EXCLUDED:
    ///   - CAT heat-prune: removes wall candidates the net wants (drops Elo).
    ///   - dead-zone prune: unsound (block-a-blocker) AND its apparent gain was
    ///     measurement noise — a single-seed +76 became −25 on another seed.
    ///
    /// NOTE on measurement: 112-game runs carry a ±~64 Elo 95% CI, so per-run Elo
    /// deltas are not individually trustworthy. These two extras are kept because
    /// they are *provably* non-harmful, not because a single match "won".
    /// `tt_bits = Some(n)` pins a fixed TT instead of the adaptive one.
    pub fn grafted(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        Self::grafted_with_weights(g, tt_bits, net())
    }

    /// Same as [`grafted_frozen`] but uses the frozen v13 HalfPW blob (training A/B control).
    pub fn grafted_frozen(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        Self::grafted_with_weights(g, tt_bits, net_frozen())
    }

    /// Medium tier — runtime-installed weights (`net_weights_medium.bin`).
    pub fn grafted_medium(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        let weights = crate::titanium::net::net_medium()
            .expect("medium NNUE weights not installed — fetch net_weights_medium.bin first");
        Self::grafted_with_weights(g, tt_bits, weights)
    }

    /// Production graft minus RaceProof/cert gates. Experimental only: useful for
    /// measuring whether search can replace the proof layer before removing it.
    pub fn grafted_no_raceproof(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        let mut search = Self::grafted(g, tt_bits);
        search.race_proof = false;
        search
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn grafted_lazy_walls_for_bench(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        let mut search = Self::grafted(g, tt_bits);
        search.ti_movegen = false;
        search
    }

    /// Titanium v15 experimental — wall-ignorance loss certificate (frozen net).
    pub fn grafted_wall_ignore_experimental(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        let mut search = Self::grafted_frozen(g, tt_bits);
        search.wall_ignore_cert_override = Some(true);
        search
    }

    pub fn grafted_with_weights(
        g: GameState,
        tt_bits: Option<usize>,
        weights: &'static Net,
    ) -> Box<Self> {
        let mut search = Self::with_ti_movegen(g);
        search.net = weights;
        search.cheap_cert = true;
        search.cert_eval_leaves_only = true;
        match tt_bits {
            Some(bits) => search.resize_tt(bits),
            None => search.enable_adaptive_tt(),
        }
        search
    }

    /// **Titanium v17** — v15 graft + ACE v13 graduated CAT LMR with two hard
    /// overrides: dead-tail walls (attention ≤ 10%) and backward moves search
    /// at child depth 1. The internal `cat_lmr_v16` name is historical.
    pub fn grafted_v17(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        #[cfg(not(target_arch = "wasm32"))]
        let ceiling = crate::cat::cat_v16_lmr_ceiling_from_env();
        #[cfg(target_arch = "wasm32")]
        let ceiling = crate::cat::CAT_V16_LMR_CEILING_DEFAULT;
        Self::grafted_v17_with_ceiling(g, tt_bits, ceiling)
    }

    pub fn grafted_v17_with_ceiling(
        g: GameState,
        tt_bits: Option<usize>,
        ceiling: u16,
    ) -> Box<Self> {
        Self::grafted_v17_with_ceiling_and_weights(g, tt_bits, ceiling, net())
    }

    pub fn grafted_v17_with_ceiling_and_weights(
        g: GameState,
        tt_bits: Option<usize>,
        ceiling: u16,
        weights: &'static Net,
    ) -> Box<Self> {
        let mut search = Self::grafted_with_weights(g, tt_bits, weights);
        search.cat_lmr_v16 = true;
        search.cat_lmr_ceiling = if crate::cat::CAT_V16_LMR_CEILINGS.contains(&ceiling) {
            ceiling
        } else {
            crate::cat::CAT_V16_LMR_CEILING_DEFAULT
        };
        search
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn grafted_v17_lazy_walls_for_bench(
        g: GameState,
        tt_bits: Option<usize>,
        ceiling: u16,
    ) -> Box<Self> {
        let mut search = Self::grafted_v17_with_ceiling(g, tt_bits, ceiling);
        search.ti_movegen = false;
        search
    }

    /// Compatibility alias for the retired Titanium v16 product label.
    #[inline]
    pub fn grafted_v16(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        Self::grafted_v17(g, tt_bits)
    }

    /// Compatibility alias for the retired Titanium v16 product label.
    #[inline]
    pub fn grafted_v16_with_ceiling(
        g: GameState,
        tt_bits: Option<usize>,
        ceiling: u16,
    ) -> Box<Self> {
        Self::grafted_v17_with_ceiling(g, tt_bits, ceiling)
    }

    /// Compatibility alias for the retired Titanium v16 product label.
    #[cfg(not(target_arch = "wasm32"))]
    #[inline]
    pub fn grafted_v16_lazy_walls_for_bench(
        g: GameState,
        tt_bits: Option<usize>,
        ceiling: u16,
    ) -> Box<Self> {
        Self::grafted_v17_lazy_walls_for_bench(g, tt_bits, ceiling)
    }

    /// gen13 net search + O1 movegen + cheap hands-empty cert, but **no CAT**.
    /// Isolates the certificate contribution from CAT wall-pruning.
    pub fn with_ti_movegen_cheap_cert(g: GameState, tt_bits: Option<usize>) -> Box<Self> {
        let mut search = Self::with_ti_movegen(g);
        search.cheap_cert = true;
        if let Some(bits) = tt_bits {
            search.resize_tt(bits);
        }
        search
    }

    /// gen13 net search + O1 movegen + adaptive cache-tier TT (no CAT, no cert).
    /// Isolates the TT-growth contribution.
    pub fn with_ti_movegen_adaptive_tt(g: GameState) -> Box<Self> {
        let mut search = Self::with_ti_movegen(g);
        search.enable_adaptive_tt();
        search
    }

    /// gen13 net search + O1 movegen + SOUND dead-zone wall prune (no CAT heat).
    /// Isolates the dead-zone pruner's contribution (NPS-only, can't cost Elo).
    pub fn with_ti_movegen_deadzone(g: GameState) -> Box<Self> {
        let mut search = Self::with_ti_movegen(g);
        search.dead_zone_prune = true;
        search
    }

    /// Reallocate the transposition table to `1 << bits` entries. Clears all TT
    /// state — call before search starts, not mid-think.
    pub fn resize_tt(&mut self, bits: usize) {
        let size = 1usize << bits;
        self.tt_key_hi = vec![0; size];
        self.tt_key_lo = vec![0; size];
        self.tt_meta = vec![0; size];
        self.tt_score = vec![0; size];
        self.tt_rep = vec![0; size];
        self.tt_anc_lo = vec![0; size];
        self.tt_anc_hi = vec![0; size];
        self.tt_entry_gen = vec![0; size];
        self.tt_mask = (size - 1) as u32;
        self.tt_bits = bits;
        self.tt_filled = 0;
    }

    /// Enable overflow-driven cache-tier TT growth (Titanium strategy). Starts the
    /// table small enough to live in L1, then jumps L1→L2→L3→18→22 as it fills, so
    /// at low node counts the whole TT stays hot in cache (the big win at short TC)
    /// and only grows when it genuinely overflows. `entry_bytes` = 25 (the 7 SoA
    /// arrays: 3×u32 key/anc + 2×i32 meta/score + 1×u8 rep ≈ 25 B/logical entry).
    pub fn enable_adaptive_tt(&mut self) {
        const ENTRY_BYTES: usize = 25;
        let (start, l2, l3) = super::tt_sizing::cache_tier_bits(ENTRY_BYTES);
        self.tt_l2 = l2.max(start + 1);
        self.tt_l3 = l3.max(self.tt_l2 + 1);
        self.tt_d4 = 18.max(self.tt_l3);
        self.tt_d5 = 22.max(self.tt_d4);
        self.tt_max = 25.max(self.tt_d5);
        self.tt_adaptive = true;
        self.resize_tt(start); // start small — grows on overflow
    }

    /// Grow the TT to the next calibrated cache tier and rehash live entries.
    /// Always-replace on collision (matches the live store policy). Called from the
    /// store path when occupancy crosses 50%.
    fn tt_grow(&mut self) {
        let nb = if self.tt_bits < self.tt_l2 {
            self.tt_l2
        } else if self.tt_bits < self.tt_l3 {
            self.tt_l3
        } else if self.tt_bits < self.tt_d4 {
            self.tt_d4
        } else if self.tt_bits < self.tt_d5 {
            self.tt_d5
        } else {
            self.tt_bits + 1
        }
        .min(self.tt_max);
        if nb <= self.tt_bits {
            return;
        }
        let new_size = 1usize << nb;
        let new_mask = (new_size - 1) as u32;
        let mut k_hi = vec![0u32; new_size];
        let mut k_lo = vec![0u32; new_size];
        let mut meta = vec![0i32; new_size];
        let mut score = vec![0i32; new_size];
        let mut rep = vec![0u8; new_size];
        let mut a_lo = vec![0u32; new_size];
        let mut a_hi = vec![0u32; new_size];
        let mut e_gen = vec![0u8; new_size];
        let mut filled = 0usize;
        for i in 0..self.tt_meta.len() {
            if self.tt_meta[i] == 0 {
                continue;
            }
            let ni = (self.tt_key_lo[i] & new_mask) as usize;
            if meta[ni] == 0 {
                filled += 1;
            }
            k_hi[ni] = self.tt_key_hi[i];
            k_lo[ni] = self.tt_key_lo[i];
            meta[ni] = self.tt_meta[i];
            score[ni] = self.tt_score[i];
            rep[ni] = self.tt_rep[i];
            a_lo[ni] = self.tt_anc_lo[i];
            a_hi[ni] = self.tt_anc_hi[i];
            e_gen[ni] = self.tt_entry_gen[i];
        }
        self.tt_key_hi = k_hi;
        self.tt_key_lo = k_lo;
        self.tt_meta = meta;
        self.tt_score = score;
        self.tt_rep = rep;
        self.tt_anc_lo = a_lo;
        self.tt_anc_hi = a_hi;
        self.tt_entry_gen = e_gen;
        self.tt_mask = new_mask;
        self.tt_bits = nb;
        self.tt_filled = filled;
    }

    /// Advance the live game one ply, keeping TT/killers/history warm.
    /// Long-lived session path — the next `think` reuses prior analysis.
    pub fn apply_move(&mut self, m: i16) {
        self.g.make_move(m);
        if is_wall_move(m) {
            self.cached_stamp = -1;
        }
        if self.pure_mode {
            // Faithful JS baseline: reset accumulator every move (no retention).
            self.np_b0 = -1;
        }
        // non-pure: do NOT reset np_b0/np_b1v — evaluate()'s bucket-aware diff handles any
        // accumulator transition (wall diff or full rebuild on bucket cross).
    }

    /// Replace the position outright (undo, new game) without clearing the
    /// TT — entries are hash-keyed, stale ones simply never match.
    pub fn set_position(&mut self, g: GameState) {
        self.g = g;
        self.position_changed();
    }

    /// Scale history table by a surprise-proportional factor.
    /// Called when the opponent played an unexpected move so stale tactical
    /// patterns from the abandoned search don't dominate the new root.
    /// For a correct prediction (|prior - current| ≈ 0) decay ≈ 1.0 (no-op).
    pub fn decay_history_by_surprise(&mut self, prior_score: i32) {
        let surprise = (prior_score - self.root_score).abs() as f32;
        let decay = 1.0 / (1.0 + surprise / 200.0);
        for h in self.history_tbl.iter_mut() {
            *h = (*h as f32 * decay) as i32;
        }
        for side in self.hist_sf.iter_mut() {
            for h in side.iter_mut() {
                *h = (*h as f32 * decay) as i32;
            }
        }
        for h in self.cont_hist.iter_mut() {
            *h = (*h as f32 * decay) as i32;
        }
    }

    /// Enable the Stockfish-style history experiment (side-split + gravity +
    /// maluses). Off by default; A/B-gated via the match harness before any
    /// default flip — see the `hist_sf` field docs.
    pub fn set_sf_history(&mut self, on: bool) {
        self.sf_history = on;
    }

    pub fn sf_history_enabled(&self) -> bool {
        self.sf_history
    }

    /// Enable the conservative ProbCut experiment. Disabled by default: it
    /// trades a shallow verification search for a speculative beta cutoff and
    /// must be measured separately before any broader use.
    pub fn set_probcut(&mut self, on: bool) {
        self.probcut = on;
    }

    /// Full-action history as seen by ordering/pruning: the side-split gravity
    /// table when the SF-history experiment is on, the legacy shared counter
    /// otherwise. Pawn destinations already occupy valid move IDs.
    #[inline]
    fn move_hist(&self, stm: usize, m: i16) -> i32 {
        if self.sf_history {
            self.hist_sf[stm][m as usize]
        } else {
            self.history_tbl[m as usize]
        }
    }

    /// Gravity update: `h += bonus − h·|bonus|/MAX`. Saturates at ±MAX with no
    /// overflow guard needed, and pulls stale scores toward the fresh signal
    /// (a saturated entry that stops earning bonuses decays on every malus).
    #[inline]
    fn sf_hist_apply(&mut self, stm: usize, m: i16, bonus: i32) {
        let h = &mut self.hist_sf[stm][m as usize];
        *h += bonus - ((*h as i64 * bonus.unsigned_abs() as i64) / SF_HIST_MAX as i64) as i32;
    }

    /// Same gravity formula on the continuation-history surface.
    #[inline]
    fn cont_hist_apply(&mut self, prev_move: i16, m: i16, bonus: i32) {
        let h = &mut self.cont_hist[prev_move as usize * HIST_SPAN + m as usize];
        *h += bonus - ((*h as i64 * bonus.unsigned_abs() as i64) / SF_HIST_MAX as i64) as i32;
    }

    #[inline]
    fn cont_hist_read(&self, prev_move: i16, m: i16) -> i32 {
        if prev_move < 0 {
            return 0;
        }
        self.cont_hist[prev_move as usize * HIST_SPAN + m as usize]
    }

    /// Wall-structure bucket for the correction history: the incremental
    /// position zobrist with both pawn components (and the turn component)
    /// XORed back out — an O(1) walls-only hash, no game.rs changes needed.
    #[inline]
    fn wall_corr_index(&self) -> usize {
        let z = &ZOBRIST;
        let mut h = self.g.hash_lo ^ z.pawn_lo[0][self.g.pawn[0]] ^ z.pawn_lo[1][self.g.pawn[1]];
        if self.g.turn == 1 {
            h ^= z.turn_lo;
        }
        (h as usize) & (CORR_SIZE - 1)
    }

    /// Advance the root by one ply (predicted opponent move) and adjust state
    /// for seamless continuation. For use after `go infinite` + `ponderhit`.
    pub fn migrate_root(&mut self, m: i16, prior_score: i32) {
        self.apply_move(m);
        self.decay_history_by_surprise(prior_score);
        if !self.pure_mode {
            self.tt_gen = self.tt_gen.wrapping_add(1);
        }
    }

    /// Static evaluation of the current position (no search) — primes the distance
    /// cache and forces an accumulator rebuild, then runs `evaluate()`. On mid-game
    /// positions this returns the pure HalfPW net output; used by the NNUE trainer
    /// parity harness to confirm the Python forward pass matches the engine.
    pub fn eval_position(&mut self) -> i32 {
        self.position_changed();
        self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_EVAL_POSITION);
        self.evaluate(0)
    }

    /// Enable Lague partial-iteration (keep the best fully-searched move from a
    /// time-aborted deepest iteration). Off by default; A/B-measured before adoption.
    pub fn set_partial_iter(&mut self, on: bool) {
        self.use_partial_iter = on;
    }

    pub fn partial_iter_enabled(&self) -> bool {
        self.use_partial_iter
    }

    /// Enable or disable the predictive iterative-deepening stop.
    pub fn set_predict_stop(&mut self, on: bool) {
        self.use_predict_stop = on;
    }

    pub fn predict_stop_enabled(&self) -> bool {
        self.use_predict_stop
    }

    /// Enter/exit ponder mode. While pondering, `think()` skips the tt_gen
    /// advance and history decay so all ponder chunks build on each other
    /// rather than aging their own work.  Call with `false` before the real
    /// think so it does the normal one-time decay and advances the generation.
    pub fn set_pondering(&mut self, on: bool) {
        self.is_pondering = on;
    }

    pub fn set_cat_lmr_fringe_pct(&mut self, pct: u16) {
        self.cat_lmr_fringe_pct = pct.min(crate::cat::CAT_V16_FRINGE_PCT_MAX);
    }

    pub fn set_cat_lmr_worker_profile(&mut self, worker_id: usize) {
        self.set_cat_lmr_fringe_pct(cat_v16_lmr_fringe_pct_for_worker(worker_id));
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn lazy_smp_width_percent(worker_id: usize) -> usize {
        LAZY_SMP_WIDTHS
            .get(worker_id)
            .copied()
            .unwrap_or(*LAZY_SMP_WIDTHS.last().expect("width schedule is non-empty"))
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn apply_think_start_state(&mut self) {
        if !self.pure_mode && !self.is_pondering {
            self.tt_gen = self.tt_gen.wrapping_add(1);
            for h in self.history_tbl.iter_mut() {
                *h >>= 1;
            }
            // Same aging policy for the SF-history experiment tables (`/ 2`,
            // not `>>`: arithmetic shift would pin negative entries at -1
            // forever instead of converging to 0). Correction history is NOT
            // aged — eval bias per wall structure is slow-moving knowledge,
            // not a tactical pattern (SF also persists it across moves).
            for side in self.hist_sf.iter_mut() {
                for h in side.iter_mut() {
                    *h /= 2;
                }
            }
            for h in self.cont_hist.iter_mut() {
                *h /= 2;
            }
        }
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn ordered_root_moves_snapshot(&mut self, depth: i32) -> Vec<i16> {
        self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_LAZY_ROOT);
        if self.bridge.is_some() {
            self.bridge = Some(TiBridge::from_game(&self.g));
        }
        let mut moves = [0i16; 160];
        let root_entry = self
            .shared_tt
            .as_ref()
            .and_then(|tt| tt.probe(self.g.hash_lo, self.g.hash_hi));
        let tt_move = root_entry
            .map(|entry| (entry.meta & 1023) as i16)
            .unwrap_or_else(|| {
                let idx = (self.g.hash_lo & self.tt_mask) as usize;
                let meta = self.tt_meta[idx];
                if meta != 0
                    && self.tt_key_hi[idx] == self.g.hash_hi
                    && self.tt_key_lo[idx] == self.g.hash_lo
                {
                    (meta & 1023) as i16
                } else {
                    0
                }
            });
        let n = self.gen_moves(0, depth.max(1), tt_move, &mut moves);
        self.order_moves(0, &mut moves[..n], tt_move, 0);
        let n = crate::titanium::opening_book::filter_denied_opening_legal_moves(
            &self.g, &mut moves, n,
        );
        moves[..n].to_vec()
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn gcd_usize(mut a: usize, mut b: usize) -> usize {
        while b != 0 {
            let r = a % b;
            a = b;
            b = r;
        }
        a
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn lazy_smp_profile_root_moves(
        root_moves: &[i16],
        worker_id: usize,
        allowed: usize,
        force_top_k: bool,
    ) -> (Vec<i16>, Vec<usize>) {
        let len = root_moves.len();
        let allowed = allowed.min(len);
        if allowed == 0 {
            return (Vec::new(), Vec::new());
        }
        // Main (worker 0) always gets the clean best-ordered slice. A worker
        // pinned to `top_n_override` (e.g. the top-3 depth specialist) also
        // needs the true top-K by move ordering, not the strided/offset
        // diversification sample below -- diversifying a 3-move slice would
        // silently swap out the actual best candidates for scattered ones,
        // defeating the entire point of a "go deep on the best moves" worker.
        if worker_id == 0 || force_top_k || len <= 1 {
            return (
                root_moves[..allowed].to_vec(),
                (0..allowed).collect::<Vec<_>>(),
            );
        }

        let mut stride = worker_id.saturating_mul(2).saturating_add(1).max(3);
        while Self::gcd_usize(stride, len) != 1 {
            stride = stride.saturating_add(2);
        }
        let offset = worker_id.saturating_mul(37) % len;
        let mut seen = vec![false; len];
        let mut profiled = Vec::with_capacity(allowed);
        let mut original_indices = Vec::with_capacity(allowed);
        let mut cursor = offset;
        while profiled.len() < allowed {
            if !seen[cursor] {
                seen[cursor] = true;
                profiled.push(root_moves[cursor]);
                original_indices.push(cursor);
            }
            cursor = (cursor + stride) % len;
        }
        (profiled, original_indices)
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn fork_lazy_worker(&self, root: &GameState) -> Box<Self> {
        let mut worker = Self::new(root.clone());
        worker.history_tbl = self.history_tbl;
        worker.sf_history = self.sf_history;
        worker.probcut = self.probcut;
        worker.hist_sf = self.hist_sf;
        worker.corr_hist = self.corr_hist;
        worker.cont_hist = self.cont_hist.clone();
        worker.cm = self.cm;
        worker.killers = self.killers;
        worker.net = self.net;
        worker.ti_movegen = self.ti_movegen;
        worker.cat_walls = self.cat_walls;
        worker.cat_lmr_v16 = self.cat_lmr_v16;
        worker.cat_lmr_ceiling = self.cat_lmr_ceiling;
        worker.cat_lmr_fringe_pct = self.cat_lmr_fringe_pct;
        worker.route_touch_ordering = self.route_touch_ordering;
        worker.cat_path_lmr = self.cat_path_lmr;
        worker.cat_no_edge_skip = self.cat_no_edge_skip;
        worker.q_search = self.q_search;
        worker.q_max = self.q_max;
        worker.q_swing_cp = self.q_swing_cp;
        worker.lazy_topn = self.lazy_topn;
        worker.ace_lmp = self.ace_lmp;
        worker.use_predict_stop = self.use_predict_stop;
        worker.ace_rfp = self.ace_rfp;
        worker.rfp_tc_adaptive = self.rfp_tc_adaptive;
        worker.ace_rfp_max_depth = self.ace_rfp_max_depth;
        worker.dead_zone_prune = self.dead_zone_prune;
        worker.cheap_cert = self.cheap_cert;
        worker.cert_eval_leaves_only = self.cert_eval_leaves_only;
        worker.wall_ignore_cert_override = self.wall_ignore_cert_override;
        worker.one_wall_race_resolved = self.one_wall_race_resolved;
        worker.one_wall_race_pv_only = self.one_wall_race_pv_only;
        worker.two_wall_race_resolved = self.two_wall_race_resolved;
        worker.two_wall_race_pv_only = self.two_wall_race_pv_only;
        worker.eme = self.eme;
        worker.use_partial_iter = self.use_partial_iter;
        worker.pure_mode = self.pure_mode;
        worker.race_proof = self.race_proof;
        worker.opening_book_mode = self.opening_book_mode;
        worker.opening_book_order = self.opening_book_order.clone();
        worker.opening_book_attention = self.opening_book_attention.clone();
        worker.opening_book = self.opening_book.clone();
        worker.tt_gen = self.tt_gen;
        worker.tt_mask = self.tt_mask;
        worker.tt_bits = self.tt_bits;
        worker.tt_adaptive = false;
        // Helpers search against the shared lazy-SMP TT only, so drop the full
        // local TT that Self::new allocated. At TT_BITS=20 that is ~26MB per
        // worker; 7 helpers previously allocated ~182MB of dead tables and
        // overflowed the wasm memory cap (the first threaded search aborted in
        // handle_alloc_error → bare `unreachable`). Local TT probes/stores are
        // gated on `shared_tt.is_none()`, so these 1-element vecs are never
        // indexed once install_lazy_smp_context() runs.
        worker.tt_key_hi = vec![0; 1];
        worker.tt_key_lo = vec![0; 1];
        worker.tt_meta = vec![0; 1];
        worker.tt_score = vec![0; 1];
        worker.tt_rep = vec![0; 1];
        worker.tt_anc_lo = vec![0; 1];
        worker.tt_anc_hi = vec![0; 1];
        worker.tt_entry_gen = vec![0; 1];
        // Helpers search as temporary workers. Private giant eval/dist caches
        // duplicate ~70MB each and thrash LLC; same rationale as dropping local TT.
        // Keep 1-slot stubs so evaluate/dist_lru paths stay valid (heavy thrash = nearly no cache).
        worker.eval_cache = vec![EvalCacheEntry::default(); 1];
        worker.eval_cache_bits = 0;
        worker.dist_lru = vec![DistTopoEntry::default(); 1];
        worker.dist_lru_bits = 0;
        worker.dist_lru_filled = 0;
        worker.dist_lru_growable = false;
        // Helpers rarely need spill; keep pools empty (1-slot stub thrash).
        worker.dist_layer_spill.clear();
        if self.bridge.is_some() {
            worker.bridge = Some(TiBridge::from_game(root));
        }
        worker
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn install_lazy_smp_context(
        &mut self,
        worker_id: usize,
        shared_tt: Arc<SharedTitaniumTt>,
        runtime: Arc<LazySmpRuntime>,
        root_moves: Arc<Vec<i16>>,
        root_visit_map: Arc<Vec<usize>>,
        allowed: usize,
    ) {
        self.shared_tt = Some(shared_tt.clone());
        self.tt_mask = shared_tt.mask;
        self.tt_bits = shared_tt.bits;
        self.tt_adaptive = false;
        self.lazy_runtime = Some(runtime.clone());
        self.deadline = runtime.deadline;
        self.lazy_root_moves = Some(root_moves);
        self.lazy_root_visit_map = Some(root_visit_map);
        self.lazy_root_allowed = allowed;
        self.lazy_worker_id = worker_id;
        self.set_cat_lmr_worker_profile(worker_id);
        self.lazy_skip_setup = true;
        self.lazy_root_visits.clear();
    }

    /// Wall-cache profiling counters (TiBridge path only).
    pub fn wall_cache_stats(&self) -> Option<GeometricWallCacheStats> {
        self.bridge.as_ref().map(|b| b.wall_cache_stats)
    }

    /// Dump the raw net inputs + the resulting eval as JSON. Lets the Python NNUE
    /// trainer verify its forward pass against the engine on the *inputs alone*,
    /// without reimplementing Quoridor rules/BFS in Python — and is the record
    /// format for training-data generation.
    ///
    /// `d0`/`d1` are the pawn shortest-path distances (scalars).
    /// Canonical field keys: `goal_inv_p0_field`, `pawn_fwd_p0_field`, `corridor_delta_p0_field`,
    /// `path_cross_p0_field` (and `_p1` variants). Legacy aliases `d0_field`, `player0_field`, …
    /// are duplicated in the JSON for old JSONL; trainer reads either via `rec_field()`.
    /// Same JSON as [`Self::eval_dump_json`] with packed-batch metadata prefix fields.
    pub fn eval_dump_json_packed(&mut self, row: u32) -> String {
        let body = self.eval_dump_json();
        format!(
            "{{\"row\":{row},\"ok\":true,\"feature_schema\":\"{FEATURE_SCHEMA}\",\"protocol\":\"eval-packed-v1\",{}",
            &body[1..]
        )
    }

    /// Emit CATv5's four exact paths per player, ranked raw witnesses, and
    /// per-player/combined propagated fields. Normalization belongs to the
    /// learning boundary; this diagnostic protocol preserves exact integers.
    pub fn cat_dump_json_packed(&self, row: u32) -> String {
        let bridge = TiBridge::from_game(&self.g);
        let maps = crate::cat::build::build_catv5_heatmaps(&bridge.board);
        let field8 = |arr: &[u8; 81]| arr.iter().map(u8::to_string).collect::<Vec<_>>().join(",");
        let field16 =
            |arr: &[u16; 81]| arr.iter().map(u16::to_string).collect::<Vec<_>>().join(",");
        format!(
            "{{\"row\":{row},\"ok\":true,\"protocol\":\"catv5-precise-packed-v2\",\"cat_witness_p0_field\":[{}],\"cat_witness_p1_field\":[{}],\"cat_propagated_p0_field\":[{}],\"cat_propagated_p1_field\":[{}],\"cat_propagated_field\":[{}]}}",
            field8(&maps.witness_p0),
            field8(&maps.witness_p1),
            field16(&maps.propagated_p0),
            field16(&maps.propagated_p1),
            field16(&maps.propagated),
        )
    }

    pub fn eval_dump_json(&mut self) -> String {
        self.position_changed();
        self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_EVAL_DUMP);
        let net_eval = self.compute_net_eval_trace().eval;
        let eval = self.evaluate(0);
        let d0_scalar = self.d0[self.dist0_idx][self.g.pawn[0]];
        let d1_scalar = self.d1[self.dist1_idx][self.g.pawn[1]];
        let bits = |arr: &[u8; 64]| {
            let mut s = String::new();
            for (i, b) in arr.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push(if *b != 0 { '1' } else { '0' });
            }
            s
        };
        let field = |arr: &[u8; 81]| {
            let mut s = String::new();
            for (i, &v) in arr.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&v.to_string());
            }
            s
        };
        let field16 = |arr: &[u16; 81]| {
            let mut s = String::new();
            for (i, &v) in arr.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&v.to_string());
            }
            s
        };
        let d0f = self.d0[self.dist0_idx];
        let d1f = self.d1[self.dist1_idx];
        let mut p0_steps = [255u8; 81];
        let mut p1_steps = [255u8; 81];
        let mut delta0 = [255u8; 81];
        let mut delta1 = [255u8; 81];
        fill_ace_dist_from_pawn(&self.g, self.g.pawn[0], &mut p0_steps);
        fill_ace_dist_from_pawn(&self.g, self.g.pawn[1], &mut p1_steps);
        fill_corridor_delta(&p0_steps, &d0f, d0_scalar, &mut delta0);
        fill_corridor_delta(&p1_steps, &d1f, d1_scalar, &mut delta1);
        let cross0 = [0u8; 81];
        let cross1 = [0u8; 81];
        let mut choke0 = [0u8; 81];
        let mut choke1 = [0u8; 81];
        fill_choke_points(&self.g, &p0_steps, &d0f, d0_scalar, &mut choke0);
        fill_choke_points(&self.g, &p1_steps, &d1f, d1_scalar, &mut choke1);
        let mut contested = [0u8; 81];
        fill_contested(&delta0, &delta1, &mut contested);
        let mut route0 = [0u8; 81];
        let mut route1 = [0u8; 81];
        let mut flank0 = [0u8; 81];
        let mut flank1 = [0u8; 81];
        fill_sparse_route_masks(&self.g, self.g.pawn[0], &d0f, &mut route0, &mut flank0);
        fill_sparse_route_masks(&self.g, self.g.pawn[1], &d1f, &mut route1, &mut flank1);
        let (cat_best_p0, cat_best_p1, cat_maps) = {
            let mut bridge = TiBridge::from_game(&self.g);
            let maps = crate::cat::build::build_catv5_heatmaps(&bridge.board);
            let mut cat = crate::cat::attention::CorridorAttention::default();
            cat.square_heat = maps.propagated;
            let (best0, best1) =
                crate::cat::best_pawn_cat_heats(&bridge.board, &cat, &mut bridge.bfs);
            (best0, best1, maps)
        };
        let legal_walls = 0;
        let (cross_p0, cross_p1) = (0, 0);
        let width_me = self.d0[self.dist0_idx]
            .iter()
            .filter(|&&d| d as i32 == d0_scalar as i32)
            .count();
        let width_opp = self.d1[self.dist1_idx]
            .iter()
            .filter(|&&d| d as i32 == d1_scalar as i32)
            .count();
        format!(
            "{{\"turn\":{},\"pawn0\":{},\"pawn1\":{},\"wl0\":{},\"wl1\":{},\
             \"d0\":{},\"d1\":{},\"legal_wall_count\":{},\"legal_path_cross_p0\":{},\"legal_path_cross_p1\":{},\
             \"cat_best_p0\":{},\"cat_best_p1\":{},\"cat_witness_p0_field\":[{}],\"cat_witness_p1_field\":[{}],\"cat_propagated_p0_field\":[{}],\"cat_propagated_p1_field\":[{}],\"cat_propagated_field\":[{}],\
             \"corridor_width0\":{},\"corridor_width1\":{},\
             \"goal_inv_p0_field\":[{}],\"goal_inv_p1_field\":[{}],\
             \"pawn_fwd_p0_field\":[{}],\"pawn_fwd_p1_field\":[{}],\
             \"corridor_delta_p0_field\":[{}],\"corridor_delta_p1_field\":[{}],\
             \"path_cross_p0_field\":[{}],\"path_cross_p1_field\":[{}],\
             \"choke_p0_field\":[{}],\"choke_p1_field\":[{}],\
             \"contested_field\":[{}],\
             \"route_p0_field\":[{}],\"route_p1_field\":[{}],\
             \"route_flank_p0_field\":[{}],\"route_flank_p1_field\":[{}],\
             \"d0_field\":[{}],\"d1_field\":[{}],\
             \"player0_field\":[{}],\"player1_field\":[{}],\
             \"delta0_field\":[{}],\"delta1_field\":[{}],\
             \"cross0_field\":[{}],\"cross1_field\":[{}],\
             \"hw\":[{}],\"vw\":[{}],\"net_eval\":{},\"eval\":{}}}",
            self.g.turn,
            self.g.pawn[0],
            self.g.pawn[1],
            self.g.wl[0],
            self.g.wl[1],
            d0_scalar,
            d1_scalar,
            legal_walls,
            cross_p0,
            cross_p1,
            cat_best_p0,
            cat_best_p1,
            field(&cat_maps.witness_p0),
            field(&cat_maps.witness_p1),
            field16(&cat_maps.propagated_p0),
            field16(&cat_maps.propagated_p1),
            field16(&cat_maps.propagated),
            width_me,
            width_opp,
            field(&d0f),
            field(&d1f),
            field(&p0_steps),
            field(&p1_steps),
            field(&delta0),
            field(&delta1),
            field(&cross0),
            field(&cross1),
            field(&choke0),
            field(&choke1),
            field(&contested),
            field(&route0),
            field(&route1),
            field(&flank0),
            field(&flank1),
            field(&d0f),
            field(&d1f),
            field(&p0_steps),
            field(&p1_steps),
            field(&delta0),
            field(&delta1),
            field(&cross0),
            field(&cross1),
            bits(&self.g.hw),
            bits(&self.g.vw),
            net_eval,
            eval
        )
    }

    /// Parity harness only: net eval intermediates without changing ``evaluate()``.
    pub fn eval_parity_trace_json(&mut self) -> String {
        self.position_changed();
        self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_EVAL_PARITY);
        let trace = self.compute_net_eval_trace();
        let f64s = |arr: &[f64]| {
            let mut s = String::new();
            for (i, v) in arr.iter().enumerate() {
                if i > 0 {
                    s.push(',');
                }
                s.push_str(&format!("{v:.17}"));
            }
            s
        };
        format!(
            "{{\"scalar_inputs\":{{\"d_me\":{dm},\"d_opp\":{do_},\"w_me\":{wm},\"w_opp\":{wo},\"pd\":{pd},\"wd\":{wd},\"width_opp\":{wo_}}},\
             \"scalar_out\":{so},\"route_out\":{ro},\"cat_out\":{co},\"width_contrib\":{wc},\
             \"wall_acc\":[{wa}],\"hidden_pre\":[{hp}],\"hidden_clip\":[{hc}],\"neural_out\":{no},\"eval\":{ev}}}",
            dm = trace.d_me,
            do_ = trace.d_opp,
            wm = trace.w_me,
            wo = trace.w_opp,
            pd = trace.pd,
            wd = trace.wd,
            wo_ = trace.width_opp,
            so = trace.scalar_out,
            ro = trace.route_out,
            co = trace.cat_out,
            wc = trace.width_contrib,
            wa = f64s(&trace.wall_acc),
            hp = f64s(&trace.hidden_pre),
            hc = f64s(&trace.hidden_clip),
            no = trace.neural_out,
            ev = trace.eval,
        )
    }

    fn compute_net_eval_trace(&mut self) -> EvalParityTrace {
        let me = self.g.turn;
        let opp = 1 - me;
        let d_me_u = if me == 0 {
            self.d0[self.dist0_idx][self.g.pawn[0]]
        } else {
            self.d1[self.dist1_idx][self.g.pawn[1]]
        };
        let d_opp_u = if opp == 0 {
            self.d0[self.dist0_idx][self.g.pawn[0]]
        } else {
            self.d1[self.dist1_idx][self.g.pawn[1]]
        };
        let w_me_i = self.g.wl[me];
        let w_opp_i = self.g.wl[opp];
        let d_me_i = d_me_u as i32;
        let d_opp_i = d_opp_u as i32;
        let d_me = d_me_i as f64;
        let d_opp = d_opp_i as f64;
        let w_me = w_me_i as f64;
        let w_opp = w_opp_i as f64;
        let nw = self.net;
        let ws = &nw.ws;
        let pd = d_opp - d_me;
        let wd = w_me - w_opp;
        let mut scalar_out = ws[0]
            + ws[1] * pd
            + ws[2] * wd
            + ws[3] * d_me
            + ws[4] * d_opp
            + ws[9] * pd * (w_me + w_opp) / 20.0
            + ws[10] * wd * (d_me + d_opp) / 16.0;
        if w_opp_i == 0 {
            scalar_out += ws[6];
            if d_me <= d_opp {
                scalar_out += ws[5];
            }
        } else if w_me_i == 0 {
            scalar_out += ws[8];
            if d_opp <= d_me - 1.0 {
                scalar_out += ws[7];
            }
        }
        if d_opp <= 4.0 {
            scalar_out += ws[11] * if w_me < 3.0 { w_me } else { 3.0 };
        }
        if d_me <= 4.0 {
            scalar_out += ws[12] * if w_opp < 3.0 { w_opp } else { 3.0 };
        }
        scalar_out += ws[13] * pd * w_opp / 10.0;
        let (route_out, _, _) = self.route_feature_score(nw);
        let mut cat_out = 0.0;
        if nw.cat_active {
            if let Some(bridge) = self.bridge.as_ref() {
                let cat = crate::cat::build::build_catv5_heatmaps(&bridge.board);
                let (raw_me, raw_opp, prop_me, prop_opp) = if me == 0 {
                    (
                        &cat.witness_p0,
                        &cat.witness_p1,
                        &cat.propagated_p0,
                        &cat.propagated_p1,
                    )
                } else {
                    (
                        &cat.witness_p1,
                        &cat.witness_p0,
                        &cat.propagated_p1,
                        &cat.propagated_p0,
                    )
                };
                for sq in 0..81usize {
                    let canon = if me == 0 { sq } else { NET_MIRC[sq] };
                    cat_out += nw.cat_raw_me[canon] * (f64::from(raw_me[sq]) / 4.0)
                        + nw.cat_raw_opp[canon] * (f64::from(raw_opp[sq]) / 4.0)
                        + nw.cat_propagated_me[canon] * (f64::from(prop_me[sq]) / 200.0)
                        + nw.cat_propagated_opp[canon] * (f64::from(prop_opp[sq]) / 200.0)
                        + nw.cat_propagated_combined[canon]
                            * (f64::from(cat.propagated[sq]) / 400.0);
                }
            }
        }
        let width_opp = if self.net.route_active {
            (if me == 0 {
                width_in_layers(
                    &self.d1_layers[self.dist1_idx],
                    self.d1_layer_depth[self.dist1_idx],
                    d_opp_u,
                )
            } else {
                width_in_layers(
                    &self.d0_layers[self.dist0_idx],
                    self.d0_layer_depth[self.dist0_idx],
                    d_opp_u,
                )
            }) as f64
        } else if me == 0 {
            self.d1[self.dist1_idx]
                .iter()
                .filter(|&&d| d as i32 == d_opp_i)
                .count() as f64
        } else {
            self.d0[self.dist0_idx]
                .iter()
                .filter(|&&d| d as i32 == d_opp_i)
                .count() as f64
        };
        let width_contrib = ws[15] * width_opp;
        let b0 = NET_BKT[self.g.pawn[0]] as i32;
        let b1 = NET_BKT[NET_MIRC[self.g.pawn[1]]] as i32;
        self.ensure_nnue_wall_accumulators(nw, b0, b1);
        let mut wall_acc = [0.0f64; MAX_NET_H];
        let mut hidden_pre = [0.0f64; MAX_NET_H];
        let mut hidden_clip = [0.0f64; MAX_NET_H];
        let mut neural_out = 0.0f64;
        if me == 0 {
            wall_acc = self.np_acc0;
            let po = self.g.pawn[0] * nw.h;
            let px = self.g.pawn[1] * nw.h;
            for j in 0..nw.h {
                let h = nw.b1[j] + self.np_acc0[j] + nw.po[po + j] + nw.px[px + j];
                hidden_pre[j] = h;
                hidden_clip[j] = h.clamp(0.0, 1.0);
                neural_out += nw.w2[j] * hidden_clip[j] * 200.0;
            }
        } else {
            wall_acc = self.np_acc1;
            let po = NET_MIRC[self.g.pawn[1]] * nw.h;
            let px = NET_MIRC[self.g.pawn[0]] * nw.h;
            for j in 0..nw.h {
                let h = nw.b1[j] + self.np_acc1[j] + nw.po[po + j] + nw.px[px + j];
                hidden_pre[j] = h;
                hidden_clip[j] = h.clamp(0.0, 1.0);
                neural_out += nw.w2[j] * hidden_clip[j] * 200.0;
            }
        }
        let total = scalar_out + route_out + cat_out + width_contrib + neural_out;
        EvalParityTrace {
            d_me,
            d_opp,
            w_me,
            w_opp,
            pd,
            wd,
            width_opp,
            scalar_out,
            route_out,
            cat_out,
            width_contrib,
            wall_acc,
            hidden_pre,
            hidden_clip,
            neural_out,
            eval: total as i32,
        }
    }

    fn position_changed(&mut self) {
        if self.bridge.is_some() {
            self.bridge = Some(TiBridge::from_game(&self.g));
        }
        self.cached_stamp = -1;
        self.dir_masks_key_lo = u32::MAX;
        self.dir_masks_key_hi = u32::MAX;
        self.np_b0 = -1; // force full accumulator rebuild (v10: no stamp gate)
        self.np_b1v = -1;
    }

    fn sync_stream_meta(
        &mut self,
        depth_log: &[AceDepthLogEntry],
        search_depth: i32,
        root_score: i32,
    ) {
        self.stream_depth_log.clear();
        self.stream_depth_log.extend_from_slice(depth_log);
        self.stream_search_depth = search_depth;
        self.stream_root_score = root_score;
    }

    /// Periodic + forced progress for website SSE (matches JS cumulative `search.nodes`).
    /// Periodic emits are throttled by node count AND wall time; forced emits
    /// (depth complete, root best-move change, deadline) always go out.
    fn emit_stream_progress(&mut self, force: bool) {
        if !self.stream_log {
            return;
        }
        let elapsed_ms = self.stream_t0.elapsed().as_millis() as u64;
        if !force {
            if self.nodes == 0 || self.nodes == self.stream_last_emit_nodes {
                return;
            }
            if (self.nodes & STREAM_EMIT_NODE_MASK) != 0 {
                return;
            }
            if elapsed_ms.saturating_sub(self.stream_last_emit_ms) < STREAM_EMIT_MIN_INTERVAL_MS {
                return;
            }
        }
        self.stream_last_emit_ms = elapsed_ms;
        self.stream_last_emit_nodes = self.nodes;
        self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_PROGRESS);
        let white_dist = self.d0[self.dist0_idx][self.g.pawn[0]];
        let black_dist = self.d1[self.dist1_idx][self.g.pawn[1]];
        let elapsed_ms = self.stream_t0.elapsed().as_millis() as u64;
        emit_ace_progress(
            &self.stream_label,
            &self.stream_depth_log,
            self.stream_search_depth,
            self.nodes,
            self.stream_root_score,
            &self.stream_root_moves,
            white_dist,
            black_dist,
            elapsed_ms,
            self.root_scores,
            self.multipv,
            RaceResultInfo::from_score(self.stream_root_score),
            #[cfg(feature = "wasm")]
            Some(&mut self.wasm_progress),
            #[cfg(feature = "wasm")]
            self.wasm_progress_cb.as_ref(),
        );
    }

    #[inline(always)]
    fn check_time(&mut self) -> Result<(), TimeUp> {
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        if let Some(runtime) = self.lazy_runtime.as_ref() {
            if runtime.stop.load(Ordering::Relaxed) {
                return Err(TimeUp);
            }
        }
        // Sampling the wall clock every 1024 nodes assumes nodes are cheap and
        // roughly uniform cost. Under single-threaded browser WASM (no SIMD,
        // heavier NNUE/CAT eval per node than native), a 1024-node batch can
        // itself take multiple seconds — so a requested 1s/depth-N budget
        // could overrun by however long that whole batch takes before the
        // next check. Checking every 63 nodes instead keeps the worst-case
        // overrun small regardless of per-node cost.
        if (self.nodes & 63) == 0 {
            if Instant::now() > self.deadline {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                if let Some(runtime) = self.lazy_runtime.as_ref() {
                    runtime.stop.store(true, Ordering::Relaxed);
                }
                self.emit_stream_progress(true);
                return Err(TimeUp);
            }
            self.emit_stream_progress(false);
        }
        Ok(())
    }

    fn ace_time_fraction(last_score: i32) -> f64 {
        if last_score < -80 {
            0.92
        } else {
            0.85
        }
    }

    /// Stockfish-style soft budget as a fraction of hard `time_ms`.
    ///
    /// - Base: [`ace_time_fraction`] (0.85 / 0.92 when losing)
    /// - Unstable (best-move flips, score drops): extend toward hard
    /// - Stable (same best, quiet score): shorten to save clock for later
    ///
    /// Hard deadline stays `time_ms`; this only moves the soft stop.
    pub(crate) fn stability_soft_fraction(
        last_score: i32,
        best_move_changes: u32,
        score_delta: i32,
        stable_iters: u32,
    ) -> f64 {
        let mut frac = Self::ace_time_fraction(last_score);
        // Best-move instability → spend more (up to +0.15).
        if best_move_changes > 0 {
            frac += 0.05 * (best_move_changes.min(3) as f64);
        }
        // Eval falling from STM POV (score dropped) → extend.
        if score_delta <= -40 {
            frac += 0.08;
        } else if score_delta >= 80 {
            // Big jump up can also be unstable aspiration; slight extend.
            frac += 0.03;
        }
        // Quiet + stable PV → save time for later stages.
        if stable_iters >= 3 && best_move_changes == 0 && score_delta.abs() < 30 {
            frac -= 0.10;
        } else if stable_iters >= 2 && best_move_changes == 0 && score_delta.abs() < 20 {
            frac -= 0.05;
        }
        frac.clamp(0.55, 1.0)
    }

    fn stability_soft_ms(
        time_ms: u64,
        last_score: i32,
        best_move_changes: u32,
        score_delta: i32,
        stable_iters: u32,
    ) -> u64 {
        let frac =
            Self::stability_soft_fraction(last_score, best_move_changes, score_delta, stable_iters);
        ((time_ms as f64) * frac).round() as u64
    }

    #[allow(dead_code)] // kept as fixed-fraction reference; live path uses soft_over_time_budget
    fn ace_over_time_budget(t0: Instant, time_ms: u64, last_score: i32) -> bool {
        let budget = time_ms as f64 * Self::ace_time_fraction(last_score);
        t0.elapsed().as_millis() as f64 > budget
    }

    fn soft_over_time_budget(t0: Instant, soft_ms: u64) -> bool {
        t0.elapsed().as_millis() as u64 > soft_ms
    }

    /// Projects the next ID iteration's cost as `max(last two iteration
    /// durations)` and refuses to start it if that projection can't fit the
    /// remaining budget. Iteration costs are strongly non-monotone (a cheap
    /// TT/cert-warmed depth can be followed by an expensive one), so a
    /// growth-multiplier model both over- and under-shoots; the max-of-2 floor
    /// is a cheap, measured-good alternative to the fixed-fraction
    /// `ace_over_time_budget` softStop rule above (ported from the ka_ab.js
    /// "predictStop" experiment in reference/ace.html — kept alongside the
    /// fraction check as a belt-and-suspenders bound, not a replacement).
    fn predicted_over_time_budget(
        t0: Instant,
        time_ms: u64,
        depth_log: &[AceDepthLogEntry],
    ) -> bool {
        let n = depth_log.len();
        if n == 0 {
            return false;
        }
        let last_ms = depth_log[n - 1].elapsed_ms as f64
            - if n >= 2 {
                depth_log[n - 2].elapsed_ms as f64
            } else {
                0.0
            };
        let prev_ms = if n >= 2 {
            depth_log[n - 2].elapsed_ms as f64
                - if n >= 3 {
                    depth_log[n - 3].elapsed_ms as f64
                } else {
                    0.0
                }
        } else {
            0.0
        };
        let projected = last_ms.max(prev_ms).max(20.0);
        let elapsed = t0.elapsed().as_millis() as f64;
        elapsed + projected > time_ms as f64
    }

    /// Returns (score, route0_bits, route1_bits) so callers can reuse the route bitsets.
    fn route_feature_score(&mut self, nw: &Net) -> (f64, u128, u128) {
        crate::bench_instr::record(
            |b| &mut b.eval_route_features,
            || self.route_feature_score_inner(nw),
        )
    }

    fn route_feature_score_inner(&mut self, nw: &Net) -> (f64, u128, u128) {
        if !nw.route_active {
            return (0.0, 0, 0);
        }
        let masks = self.current_dir_masks();
        let d0f = &self.d0[self.dist0_idx];
        let d1f = &self.d1[self.dist1_idx];
        let route0 = shortest_route_bits(
            self.g.pawn[0],
            d0f[self.g.pawn[0]],
            &self.d0_layers[self.dist0_idx],
            masks,
        );
        let route1 = shortest_route_bits(
            self.g.pawn[1],
            d1f[self.g.pawn[1]],
            &self.d1_layers[self.dist1_idx],
            masks,
        );
        let near0 = expand_frontier(route0, masks) & !route0 & FLOOD_PLAYABLE;
        let near1 = expand_frontier(route1, masks) & !route1 & FLOOD_PLAYABLE;
        let (me_route, opp_route, me_near, opp_near) = if self.g.turn == 0 {
            (route0, route1, near0, near1)
        } else {
            (route1, route0, near1, near0)
        };
        let contested = (me_route | me_near) & (opp_route | opp_near);
        let bybit = &nw.route_bybit[self.g.turn];
        let sum_bits = |mut bits: u128, tbl: &[f64; 128]| {
            let mut sum = 0.0;
            while bits != 0 {
                let bit = bits.trailing_zeros();
                bits &= bits - 1;
                sum += tbl[bit as usize];
            }
            sum
        };
        let score = sum_bits(me_route, &bybit[0])
            + sum_bits(opp_route, &bybit[1])
            + sum_bits(me_near, &bybit[2])
            + sum_bits(opp_near, &bybit[3])
            + sum_bits(contested, &bybit[4]);
        (score, route0, route1)
    }

    fn wall_topology_key(&self) -> (u32, u32) {
        let z = &ZOBRIST;
        let mut k_lo = self.g.hash_lo ^ z.pawn_lo[0][self.g.pawn[0]] ^ z.pawn_lo[1][self.g.pawn[1]];
        let mut k_hi = self.g.hash_hi ^ z.pawn_hi[0][self.g.pawn[0]] ^ z.pawn_hi[1][self.g.pawn[1]];
        if self.g.turn == 1 {
            k_lo ^= z.turn_lo;
            k_hi ^= z.turn_hi;
        }
        (k_lo, k_hi)
    }

    fn current_dir_masks(&mut self) -> DirMasks {
        let (k_lo, k_hi) = self.wall_topology_key();
        if self.dir_masks_key_lo != k_lo || self.dir_masks_key_hi != k_hi {
            if self.cached_stamp == self.g.wall_stamp - 1 && self.g.hist_len > 0 {
                let m = self.g.hist_m[self.g.hist_len - 1];
                if is_wall_move(m) {
                    let slot = wall_slot(m);
                    let z = &ZOBRIST;
                    let (parent_lo, parent_hi, wall_type) = if is_hwall_move(m) {
                        (k_lo ^ z.hw_lo[slot], k_hi ^ z.hw_hi[slot], 0)
                    } else {
                        (k_lo ^ z.vw_lo[slot], k_hi ^ z.vw_hi[slot], 1)
                    };
                    if self.dir_masks_key_lo == parent_lo && self.dir_masks_key_hi == parent_hi {
                        self.dir_masks_cache = self.dir_masks_cache.with_ace_wall(wall_type, slot);
                        self.dir_masks_key_lo = k_lo;
                        self.dir_masks_key_hi = k_hi;
                        return self.dir_masks_cache;
                    }
                }
            }
            self.dir_masks_cache = DirMasks::from_ace_game(&self.g);
            self.dir_masks_key_lo = k_lo;
            self.dir_masks_key_hi = k_hi;
        }
        self.dir_masks_cache
    }

    fn refresh_dist(&mut self, ply: usize) {
        self.refresh_dist_site(ply, crate::bench_instr::REFRESH_SITE_UNKNOWN);
    }

    fn refresh_dist_site(&mut self, ply: usize, site: u8) {
        #[cfg(feature = "bench-instrument")]
        {
            crate::bench_instr::refresh_site_call_start(site);
            let t0 = std::time::Instant::now();
            self.refresh_dist_inner(ply);
            crate::bench_instr::refresh_site_call_end(site, t0.elapsed());
        }
        #[cfg(not(feature = "bench-instrument"))]
        {
            let _ = site;
            self.refresh_dist_inner(ply);
        }
    }

    fn refresh_ab_skip_enabled() -> bool {
        static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("TITANIUM_REFRESH_AB_SKIP").is_some())
    }

    fn cat_child_dist_reuse_enabled() -> bool {
        static CACHED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        *CACHED.get_or_init(|| std::env::var_os("TITANIUM_CAT_CHILD_DIST_REUSE").is_some())
    }

    fn should_skip_cat_no_edge_refresh(&self) -> bool {
        if !self.cat_path_lmr {
            return false;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            if let Ok(v) = std::env::var("TITANIUM_CAT_NO_EDGE_SKIP") {
                if v == "0" || v.eq_ignore_ascii_case("false") {
                    return false;
                }
                if v == "1" || v.eq_ignore_ascii_case("true") {
                    return true;
                }
            }
        }
        self.cat_no_edge_skip
    }

    fn dist_indices_valid_for_current_topology(&self) -> bool {
        let (k_lo, k_hi) = self.wall_topology_key();
        let wkey = (k_hi as u64) << 32 | k_lo as u64;
        self.d0_key[self.dist0_idx] == wkey && self.d1_key[self.dist1_idx] == wkey
    }

    fn dist_refresh_already_valid(&self) -> bool {
        self.cached_stamp == self.g.wall_stamp && self.dist_indices_valid_for_current_topology()
    }

    fn maybe_refresh_dist_at_ab(&mut self, ply: usize) {
        let after_cat = self.pending_cat_child_ply == Some(ply);
        if after_cat {
            self.pending_cat_child_ply = None;
            self.ab_after_cat_child = true;
            crate::bench_instr::bump_u64(|b| &mut b.cat_child_ab_entries);
            if self.cached_stamp == self.g.wall_stamp {
                crate::bench_instr::bump_u64(|b| &mut b.cat_child_ab_dup_valid);
                if Self::cat_child_dist_reuse_enabled() || Self::refresh_ab_skip_enabled() {
                    crate::bench_instr::bump_u64(|b| &mut b.cat_child_ab_dup_avoided);
                    return;
                }
                crate::bench_instr::bump_u64(|b| &mut b.cat_child_ab_dup_refresh);
            }
        } else {
            self.ab_after_cat_child = false;
            if Self::refresh_ab_skip_enabled() && self.dist_refresh_already_valid() {
                crate::bench_instr::bump_ab_refresh_skipped();
                return;
            }
        }
        self.refresh_dist_site(ply, crate::bench_instr::REFRESH_SITE_AB);
    }

    /// Rare: drop spill for a topology key (no-op if absent).
    #[cold]
    fn dist_spill_remove(&mut self, wkey: u64) {
        self.dist_layer_spill.remove(&wkey);
    }

    /// Rare path: restore layers beyond [`DIST_LAYER_INLINE`] from the spill map.
    /// Marked `cold` so the common inline-only load stays a straight copy.
    /// Deep marker is `d > INLINE` — no spill_id on the hot entry.
    #[cold]
    fn dist_lru_load_spill_layers(&mut self, slot: usize, ply: usize, player: usize, d: usize) {
        crate::bench_instr::bump(|b| &mut b.dist_lru_spill_load);
        let wkey = self.dist_lru[slot].key;
        let spill = self
            .dist_layer_spill
            .get(&wkey)
            .expect("dist_lru spill missing on load");
        if player == 0 {
            let n = DIST_LAYER_INLINE.min(d);
            self.d0_layers[ply][..n]
                .copy_from_slice(&self.dist_lru[slot].d0_layers[..n]);
            debug_assert_eq!(spill.d0_tail.len(), d - DIST_LAYER_INLINE);
            self.d0_layers[ply][DIST_LAYER_INLINE..d].copy_from_slice(&spill.d0_tail);
        } else {
            let n = DIST_LAYER_INLINE.min(d);
            self.d1_layers[ply][..n]
                .copy_from_slice(&self.dist_lru[slot].d1_layers[..n]);
            debug_assert_eq!(spill.d1_tail.len(), d - DIST_LAYER_INLINE);
            self.d1_layers[ply][DIST_LAYER_INLINE..d].copy_from_slice(&spill.d1_tail);
        }
    }

    /// Rare path: insert/replace spill tails for a deep topology key.
    #[cold]
    fn dist_lru_store_spill(
        &mut self,
        wkey: u64,
        i0: usize,
        i1: usize,
        d0: usize,
        d1: usize,
        old_key: u64,
        old_was_deep: bool,
    ) {
        crate::bench_instr::bump(|b| &mut b.dist_lru_spill_store);
        if old_was_deep && old_key != u64::MAX && old_key != wkey {
            self.dist_spill_remove(old_key);
        }
        let spill = DistLayerSpill {
            d0_tail: if d0 > DIST_LAYER_INLINE {
                self.d0_layers[i0][DIST_LAYER_INLINE..d0].to_vec()
            } else {
                Vec::new()
            },
            d1_tail: if d1 > DIST_LAYER_INLINE {
                self.d1_layers[i1][DIST_LAYER_INLINE..d1].to_vec()
            } else {
                Vec::new()
            },
        };
        self.dist_layer_spill.insert(wkey, spill);
    }

    /// Copy one player's fields for `wkey` from the topology cache into ply slot.
    fn dist_lru_load(&mut self, wkey: u64, ply: usize, player: usize) -> bool {
        let slot = dist_lru_slot(wkey, self.dist_lru_bits);
        if self.dist_lru[slot].key != wkey {
            return false;
        }
        crate::bench_instr::record(
            |b| &mut b.dist_lru_hit,
            || {
                if self.net.route_active {
                    crate::bench_instr::bump(|b| &mut b.dist_lru_hit_layers);
                } else {
                    crate::bench_instr::bump(|b| &mut b.dist_lru_hit_scalar);
                }
                if player == 0 {
                    self.d0[ply] = self.dist_lru[slot].d0;
                    if self.net.route_active {
                        let d = self.dist_lru[slot].d0_depth as usize;
                        self.d0_layer_depth[ply] = d;
                        crate::bench_instr::record(
                            |b| &mut b.dist_lru_layer_copy,
                            || {
                                // Common (~99.96%): depth fits inline — copy `d`
                                // words only. Spill map is never consulted.
                                if d <= DIST_LAYER_INLINE {
                                    self.d0_layers[ply][..d].copy_from_slice(
                                        &self.dist_lru[slot].d0_layers[..d],
                                    );
                                } else {
                                    self.dist_lru_load_spill_layers(slot, ply, 0, d);
                                }
                            },
                        );
                        crate::bench_instr::add_u64(
                            |b| &mut b.dist_lru_layer_copy_bytes,
                            (d * std::mem::size_of::<u128>()) as u64,
                        );
                    }
                } else {
                    self.d1[ply] = self.dist_lru[slot].d1;
                    if self.net.route_active {
                        let d = self.dist_lru[slot].d1_depth as usize;
                        self.d1_layer_depth[ply] = d;
                        crate::bench_instr::record(
                            |b| &mut b.dist_lru_layer_copy,
                            || {
                                if d <= DIST_LAYER_INLINE {
                                    self.d1_layers[ply][..d].copy_from_slice(
                                        &self.dist_lru[slot].d1_layers[..d],
                                    );
                                } else {
                                    self.dist_lru_load_spill_layers(slot, ply, 1, d);
                                }
                            },
                        );
                        crate::bench_instr::add_u64(
                            |b| &mut b.dist_lru_layer_copy_bytes,
                            (d * std::mem::size_of::<u128>()) as u64,
                        );
                    }
                }
            },
        );
        true
    }

    /// Store BOTH players' current fields under `wkey` (contents at the live
    /// dist indices are valid for the current topology by construction).
    fn dist_lru_store(&mut self, wkey: u64) {
        let i0 = self.dist0_idx;
        let i1 = self.dist1_idx;
        let slot = dist_lru_slot(wkey, self.dist_lru_bits);
        if self.dist_lru[slot].key != u64::MAX && self.dist_lru[slot].key != wkey {
            crate::bench_instr::bump(|b| &mut b.dist_lru_replace);
        }
        if self.dist_lru[slot].key == u64::MAX {
            self.dist_lru_filled += 1;
        }

        let old_key = self.dist_lru[slot].key;
        let old_d0 = self.dist_lru[slot].d0_depth as usize;
        let old_d1 = self.dist_lru[slot].d1_depth as usize;
        let old_was_deep = old_d0 > DIST_LAYER_INLINE || old_d1 > DIST_LAYER_INLINE;

        self.dist_lru[slot].key = wkey;
        self.dist_lru[slot].d0 = self.d0[i0];
        self.dist_lru[slot].d1 = self.d1[i1];

        if self.net.route_active {
            crate::bench_instr::bump(|b| &mut b.dist_lru_store_layers);
            let d0 = self.d0_layer_depth[i0];
            let d1 = self.d1_layer_depth[i1];
            self.dist_lru[slot].d0_depth = d0 as u16;
            self.dist_lru[slot].d1_depth = d1 as u16;
            crate::bench_instr::bump_dist_layer_depth(d0);
            crate::bench_instr::bump_dist_layer_depth(d1);

            let n0 = d0.min(DIST_LAYER_INLINE);
            let n1 = d1.min(DIST_LAYER_INLINE);
            self.dist_lru[slot].d0_layers[..n0]
                .copy_from_slice(&self.d0_layers[i0][..n0]);
            self.dist_lru[slot].d1_layers[..n1]
                .copy_from_slice(&self.d1_layers[i1][..n1]);

            if d0 > DIST_LAYER_INLINE || d1 > DIST_LAYER_INLINE {
                self.dist_lru_store_spill(wkey, i0, i1, d0, d1, old_key, old_was_deep);
            } else if old_was_deep {
                self.dist_spill_remove(old_key);
            }
            // else: common→common — spill map untouched.
        } else if old_was_deep {
            crate::bench_instr::bump(|b| &mut b.dist_lru_store_scalar);
            self.dist_spill_remove(old_key);
        } else {
            crate::bench_instr::bump(|b| &mut b.dist_lru_store_scalar);
        }
        if self.dist_lru_growable
            && self.dist_lru_filled * 2 >= self.dist_lru.len()
            && self.dist_lru_bits < DIST_LRU_MAX_BITS
        {
            self.dist_lru_grow();
        }
    }

    /// Grow the dist LRU to the next size and rehash live entries.
    fn dist_lru_grow(&mut self) {
        let nb = (self.dist_lru_bits + 2).min(DIST_LRU_MAX_BITS);
        if nb <= self.dist_lru_bits {
            return;
        }
        let old = std::mem::replace(&mut self.dist_lru, vec![DistTopoEntry::default(); 1 << nb]);
        let mut filled = 0usize;
        for e in old {
            if e.key == u64::MAX {
                continue;
            }
            let ni = dist_lru_slot(e.key, nb);
            if self.dist_lru[ni].key == u64::MAX {
                filled += 1;
            } else {
                let displaced = &self.dist_lru[ni];
                if (displaced.d0_depth as usize) > DIST_LAYER_INLINE
                    || (displaced.d1_depth as usize) > DIST_LAYER_INLINE
                {
                    let k = displaced.key;
                    self.dist_spill_remove(k);
                }
            }
            self.dist_lru[ni] = e;
        }
        self.dist_lru_bits = nb;
        self.dist_lru_filled = filled;
    }

    fn refresh_dist_inner(&mut self, ply: usize) {
        let stamp = self.g.wall_stamp;
        if self.cached_stamp == stamp {
            crate::bench_instr::refresh_site_path(0);
            return; // refs already valid for these walls
        }
        if self.cached_stamp == stamp - 1 && self.g.hist_len > 0 {
            // exactly one wall added since the cached config: slots hold its dists.
            // recompute a player's field only if the wall cuts a shortest-path edge
            // (|dist diff| === 1); equal-dist edges lie on no shortest path.
            let m = self.g.hist_m[self.g.hist_len - 1];
            if is_wall_move(m) {
                let (refresh0, refresh1) =
                    wall_incr_refresh_flags(&self.d0[self.dist0_idx], &self.d1[self.dist1_idx], m);
                if !refresh0
                    && !refresh1
                    && crate::bench_instr::active_refresh_site()
                        == crate::bench_instr::REFRESH_SITE_CAT_PATH_LMR
                {
                    crate::bench_instr::bump_u64(|b| &mut b.cat_incr_no_edge_cut);
                }
                let masks = if refresh0 || refresh1 {
                    Some(self.current_dir_masks())
                } else {
                    None
                };
                let wkey = if refresh0 || refresh1 {
                    let (k_lo, k_hi) = self.wall_topology_key();
                    (k_hi as u64) << 32 | k_lo as u64
                } else {
                    0
                };
                // Restore-instead-of-reflood also applies here: iterative
                // deepening and transpositions revisit the same (ply, topology)
                // constantly — a key match means the slot already holds these
                // exact fields.
                let mut reflooded = false;
                if refresh0 {
                    self.dist0_idx = ply; // redirect first: never write an ancestor's array
                    if self.d0_key[ply] != wkey {
                        self.d0_key[ply] = wkey;
                        if !self.dist_lru_load(wkey, ply, 0) {
                            reflooded = true;
                            crate::bench_instr::refresh_site_reflood();
                            crate::bench_instr::record(
                                |b| &mut b.dist_lru_miss,
                                || {
                                    if self.net.route_active {
                                        self.d0_layer_depth[ply] = fill_ace_dist_layers_to_goal_p0(
                                            masks.expect("refresh masks"),
                                            &mut self.d0_layers[ply],
                                        );
                                        materialize_distance_layers(
                                            &self.d0_layers[ply],
                                            self.d0_layer_depth[ply],
                                            &mut self.d0[ply],
                                        );
                                    } else {
                                        fill_ace_dist_to_goal_with_masks_p0(
                                            masks.expect("refresh masks"),
                                            &mut self.d0[ply],
                                        );
                                    }
                                },
                            );
                        }
                    }
                }
                if refresh1 {
                    self.dist1_idx = ply;
                    if self.d1_key[ply] != wkey {
                        self.d1_key[ply] = wkey;
                        if !self.dist_lru_load(wkey, ply, 1) {
                            reflooded = true;
                            crate::bench_instr::refresh_site_reflood();
                            crate::bench_instr::record(
                                |b| &mut b.dist_lru_miss,
                                || {
                                    if self.net.route_active {
                                        self.d1_layer_depth[ply] = fill_ace_dist_layers_to_goal_p1(
                                            masks.expect("refresh masks"),
                                            &mut self.d1_layers[ply],
                                        );
                                        materialize_distance_layers(
                                            &self.d1_layers[ply],
                                            self.d1_layer_depth[ply],
                                            &mut self.d1[ply],
                                        );
                                    } else {
                                        fill_ace_dist_to_goal_with_masks_p1(
                                            masks.expect("refresh masks"),
                                            &mut self.d1[ply],
                                        );
                                    }
                                },
                            );
                        }
                    }
                }
                if reflooded {
                    self.dist_lru_store(wkey);
                }
                crate::bench_instr::refresh_site_path(1);
                self.cached_stamp = stamp;
                return;
            }
        }
        crate::bench_instr::refresh_site_path(2);
        self.dist0_idx = ply; // own arrays: ancestors stay intact
        self.dist1_idx = ply;
        let wkey = {
            let (k_lo, k_hi) = self.wall_topology_key();
            (k_hi as u64) << 32 | k_lo as u64
        };
        // Restore-instead-of-reflood: the slot already holds fields for this
        // exact wall topology (typical after unmaking a wall back to this node).
        let d0_ok = self.d0_key[ply] == wkey;
        let d1_ok = self.d1_key[ply] == wkey;
        if d0_ok && d1_ok {
            self.cached_stamp = stamp;
            return;
        }
        let d0_todo = !d0_ok && !self.dist_lru_load(wkey, ply, 0);
        let d1_todo = !d1_ok && !self.dist_lru_load(wkey, ply, 1);
        if d0_todo || d1_todo {
            if d0_todo {
                crate::bench_instr::refresh_site_reflood();
            }
            if d1_todo {
                crate::bench_instr::refresh_site_reflood();
            }
            let masks = self.current_dir_masks();
            crate::bench_instr::record(
                |b| &mut b.shortest_path,
                || {
                    if self.net.route_active {
                        if d0_todo {
                            crate::bench_instr::record(
                                |b| &mut b.dist_lru_miss,
                                || {
                                    self.d0_layer_depth[ply] = fill_ace_dist_layers_to_goal_p0(
                                        masks,
                                        &mut self.d0_layers[ply],
                                    );
                                    materialize_distance_layers(
                                        &self.d0_layers[ply],
                                        self.d0_layer_depth[ply],
                                        &mut self.d0[ply],
                                    );
                                },
                            );
                        }
                        if d1_todo {
                            crate::bench_instr::record(
                                |b| &mut b.dist_lru_miss,
                                || {
                                    self.d1_layer_depth[ply] = fill_ace_dist_layers_to_goal_p1(
                                        masks,
                                        &mut self.d1_layers[ply],
                                    );
                                    materialize_distance_layers(
                                        &self.d1_layers[ply],
                                        self.d1_layer_depth[ply],
                                        &mut self.d1[ply],
                                    );
                                },
                            );
                        }
                    } else {
                        if d0_todo {
                            crate::bench_instr::record(
                                |b| &mut b.dist_lru_miss,
                                || {
                                    fill_ace_dist_to_goal_with_masks_p0(masks, &mut self.d0[ply]);
                                },
                            );
                        }
                        if d1_todo {
                            crate::bench_instr::record(
                                |b| &mut b.dist_lru_miss,
                                || {
                                    fill_ace_dist_to_goal_with_masks_p1(masks, &mut self.d1[ply]);
                                },
                            );
                        }
                    }
                },
            );
        }
        self.d0_key[ply] = wkey;
        self.d1_key[ply] = wkey;
        if d0_todo || d1_todo {
            self.dist_lru_store(wkey);
        }
        self.cached_stamp = stamp;
    }

    /// Wall-topology key for `race_tbl` (pawns and turn XORed out).
    fn race_topology_key(&self) -> (u32, u32) {
        let z = &ZOBRIST;
        let mut k_lo = self.g.hash_lo ^ z.pawn_lo[0][self.g.pawn[0]] ^ z.pawn_lo[1][self.g.pawn[1]];
        let mut k_hi = self.g.hash_hi ^ z.pawn_hi[0][self.g.pawn[0]] ^ z.pawn_hi[1][self.g.pawn[1]];
        if self.g.turn == 1 {
            k_lo ^= z.turn_lo;
            k_hi ^= z.turn_hi;
        }
        (k_lo, k_hi)
    }

    /// Stage 1: LRU probe only — never builds or budget-gates.
    fn race_tbl_lru_probe(&mut self, k_lo: u32, k_hi: u32) -> Option<usize> {
        let li = self.rc_last;
        if li >= 0 && self.rc_key_lo[li as usize] == k_lo && self.rc_key_hi[li as usize] == k_hi {
            self.rc_hits += 1;
            return Some(li as usize);
        }
        for i in 0..RC_SLOTS {
            if self.rc_tbl[i].is_some() && self.rc_key_lo[i] == k_lo && self.rc_key_hi[i] == k_hi {
                self.rc_last = i as i32;
                self.rc_tick += 1;
                self.rc_use[i] = self.rc_tick;
                self.rc_hits += 1;
                return Some(i);
            }
        }
        None
    }

    #[inline]
    fn score_from_race_slot(&self, slot: usize) -> Option<i32> {
        let rv = self.race_value(slot) as i32;
        if rv > 0 {
            Some(RACE_MATE - rv)
        } else if rv < 0 {
            Some(-(RACE_MATE + rv))
        } else {
            None
        }
    }

    /// RaceProof: race table for the CURRENT wall config — LRU slot index, or
    /// `None` when the in-tree solve budget gates the build (JS `raceTbl`).
    /// Key = position hash with pawns and turn XORed out (wall config only).
    ///
    /// Only valid when both players have 0 walls in hand — the table indexes
    /// pawn pairs on a fixed wall topology, not wall-placement races.
    fn race_tbl(&mut self, force: bool) -> Option<usize> {
        if self.g.wl[0] != 0 || self.g.wl[1] != 0 {
            return None;
        }
        let (k_lo, k_hi) = self.race_topology_key();
        if let Some(slot) = self.race_tbl_lru_probe(k_lo, k_hi) {
            return Some(slot);
        }
        if !force && self.rc_blocked && k_lo == self.rc_miss_lo && k_hi == self.rc_miss_hi {
            return None;
        }
        if !force {
            // in-tree miss: build only when cheap to amortize (ticket16 SPRT-kill lesson)
            if !self.rp_build_ok
                || self.rc_think_solves >= self.rc_count_cap
                || (self.rc_think_solve_ms + self.rc_build_ms) as f64 > self.rc_solve_cap
                || Instant::now() + Duration::from_millis(self.rc_build_ms) > self.deadline
            {
                self.rc_blocked = true;
                self.rc_miss_lo = k_lo;
                self.rc_miss_hi = k_hi;
                return None;
            }
            self.rc_think_solves += 1;
        }
        let mut slot = 0usize;
        let mut min_use = u64::MAX;
        for i in 0..RC_SLOTS {
            if self.rc_tbl[i].is_none() {
                slot = i;
                break;
            }
            if self.rc_use[i] < min_use {
                min_use = self.rc_use[i];
                slot = i;
            }
        }
        let mut tbl = self.rc_tbl[slot]
            .take()
            .unwrap_or_else(|| vec![0i16; RACE_STATES].into_boxed_slice());
        if self.race_scratch.is_none() {
            self.race_scratch = Some(Box::new(RaceScratch::new()));
        }
        let t0 = Instant::now();
        crate::bench_instr::record(
            |b| &mut b.race_winner_table,
            || {
                solve_race_config(
                    &mut self.g,
                    self.race_scratch.as_mut().expect("race scratch"),
                    &mut tbl,
                );
            },
        );
        let dt0 = t0.elapsed().as_millis() as u64;
        self.rc_think_solve_ms += dt0;
        let dt = dt0 + 1;
        if dt > self.rc_build_ms {
            self.rc_build_ms = dt.min(50); // conservative adaptive gate, capped
        }
        self.rc_tbl[slot] = Some(tbl);
        self.rc_key_lo[slot] = k_lo;
        self.rc_key_hi[slot] = k_hi;
        self.rc_tick += 1;
        self.rc_use[slot] = self.rc_tick;
        self.rc_last = slot as i32;
        self.rc_solves += 1;
        Some(slot)
    }

    /// Race-table value for the game's current state (helper around a slot).
    #[inline]
    fn race_value(&self, slot: usize) -> i16 {
        let idx = (self.g.pawn[0] * 81 + self.g.pawn[1]) * 2 + self.g.turn;
        self.rc_tbl[slot].as_ref().expect("race slot")[idx]
    }

    fn exact_hands_empty_score(&mut self, force: bool) -> Option<i32> {
        if !self.race_proof || self.g.wl[0] != 0 || self.g.wl[1] != 0 {
            return None;
        }
        let slot = self.race_tbl(force)?;
        self.score_from_race_slot(slot)
    }

    /// Optional walls-remaining certificate as a typed alpha/beta bound.
    /// Its terminal-ply fields are guarantees, not exact DTM.
    fn wall_ignore_race_bound(&mut self) -> RaceBound {
        if !self.race_proof || self.g.wl[0] + self.g.wl[1] == 0 {
            return RaceBound::Unknown;
        }
        use crate::titanium::wall_ignore_cert::{
            try_wall_ignorance_loss_cert, wall_ignore_loss_cert_enabled, CertScratch,
        };
        let enabled = match self.wall_ignore_cert_resolved {
            Some(v) => v,
            None => {
                let v = self.wall_ignore_cert_override.unwrap_or(false)
                    || wall_ignore_loss_cert_enabled();
                self.wall_ignore_cert_resolved = Some(v);
                v
            }
        };
        if !enabled {
            return RaceBound::Unknown;
        }
        self.race_outcome_stats.wall_ignore_calls += 1;
        let mut scratch = CertScratch::new();
        let Some(verdict) = try_wall_ignorance_loss_cert(&mut self.g, &mut scratch, true) else {
            self.race_outcome_stats.wall_ignore_unknown += 1;
            return RaceBound::Unknown;
        };
        self.race_outcome_stats.wall_ignore_decisive += 1;
        if verdict.winner == self.g.turn {
            RaceBound::Lower(RACE_WIN_FLOOR)
        } else {
            RaceBound::Upper(-RACE_WIN_FLOOR)
        }
    }

    /// Hands-empty endgame pipeline (cheap → heavy). Caller must ensure
    /// `wl[0] == 0 && wl[1] == 0` and leaf eligibility.
    ///
    /// 1. `race_tbl` LRU probe (memo hit with exact retrograde value)
    /// 2. `race_tbl(false)` budget-gated exact build
    /// 3. Distance heuristic (unproven)
    ///
    /// Bound-only deductions run in `ab()`, where the search window exists.
    /// Static evaluation never promotes a lower/upper bound to exact.
    #[inline]
    fn one_wall_race_enabled(&mut self) -> bool {
        match self.one_wall_race_resolved {
            Some(enabled) => enabled,
            None => {
                let enabled = std::env::var("TITANIUM_RACE_ONE_WALL")
                    .ok()
                    .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
                self.one_wall_race_resolved = Some(enabled);
                enabled
            }
        }
    }

    #[inline]
    fn two_wall_race_enabled(&mut self) -> bool {
        match self.two_wall_race_resolved {
            Some(enabled) => enabled,
            None => {
                let enabled = std::env::var("TITANIUM_RACE_TWO_WALL")
                    .ok()
                    .is_some_and(|value| value == "1" || value.eq_ignore_ascii_case("true"));
                self.two_wall_race_resolved = Some(enabled);
                enabled
            }
        }
    }

    /// Exact fixed-topology winner after discarding every remaining wall.
    /// The result is a winner, not DTM: optional-wall callers expose a bound.
    fn zero_wall_winner_for_current_topology(&mut self) -> Option<usize> {
        self.zero_wall_exact_score_for_current_topology()
            .map(|score| {
                if score > 0 {
                    self.g.turn
                } else {
                    self.g.turn ^ 1
                }
            })
    }

    /// Exact race score (`±(RACE_MATE - dtm)`) after discarding remaining hands.
    /// Checks only the temporary `wl=[0,0]` state — original hands do not gate it.
    fn zero_wall_exact_score_for_current_topology(&mut self) -> Option<i32> {
        let saved_hands = self.g.wl;
        self.g.wl = [0, 0];
        let score = self.exact_hands_empty_score(false);
        self.g.wl = saved_hands;
        score
    }

    #[inline]
    fn pawns_are_race_separated(&self) -> bool {
        let p0 = self.g.pawn[0];
        let p1 = self.g.pawn[1];
        (p0 / 9).abs_diff(p1 / 9).max((p0 % 9).abs_diff(p1 % 9)) > 2
    }

    fn opponent_distance_to_goal(&self) -> u8 {
        let opponent = self.g.turn ^ 1;
        let mut dist = [u8::MAX; 81];
        self.g.compute_dist(opponent, &mut dist);
        dist[self.g.pawn[opponent]]
    }

    /// Exact-outcome subset for one remaining wall when its holder moves.
    fn one_wall_holder_winner(&mut self) -> Option<usize> {
        if self.g.wl[0] + self.g.wl[1] != 1
            || self.g.wl[self.g.turn] != 1
            || !self.pawns_are_race_separated()
        {
            return None;
        }

        let holder = self.g.turn;
        let pure_winner = self.zero_wall_winner_for_current_topology()?;
        if pure_winner == holder {
            return Some(holder);
        }
        if self.opponent_distance_to_goal() != 1 {
            return None;
        }

        let mut saw_legal_wall = false;
        let mut saw_unknown = false;
        for wall_type in 0..2usize {
            for slot in 0..64usize {
                if !self.g.wall_legal(wall_type, slot) {
                    continue;
                }
                saw_legal_wall = true;
                let mv = if wall_type == 0 {
                    MOVE_HW_BASE + slot as i16
                } else {
                    MOVE_VW_BASE + slot as i16
                };
                self.g.make_move(mv);
                let child_winner = self.zero_wall_winner_for_current_topology();
                self.g.unmake_move();
                self.cached_stamp = -1;

                match child_winner {
                    Some(winner) if winner == holder => return Some(holder),
                    Some(_) => {}
                    None => saw_unknown = true,
                }
            }
        }
        if saw_legal_wall && !saw_unknown {
            Some(holder ^ 1)
        } else {
            None
        }
    }

    /// One-wall layer when the wall-less player moves. A win needs one covered
    /// pawn reply; a loss requires every reply to be covered.
    fn one_wall_nonholder_winner(&mut self) -> Option<usize> {
        if self.g.wl[0] + self.g.wl[1] != 1 || self.g.wl[self.g.turn] != 0 {
            return None;
        }
        let mover = self.g.turn;
        let mut moves = [0i16; 16];
        let count = self.g.gen_pawn_moves(&mut moves, 0);
        if count == 0 {
            return None;
        }

        let mut saw_unknown = false;
        for &mv in &moves[..count] {
            self.g.make_move(mv);
            let child_winner = if self.g.winner() == mover as i32 {
                Some(mover)
            } else {
                self.one_wall_holder_winner()
            };
            self.g.unmake_move();
            self.cached_stamp = -1;

            match child_winner {
                Some(winner) if winner == mover => return Some(mover),
                Some(_) => {}
                None => saw_unknown = true,
            }
        }
        if saw_unknown {
            None
        } else {
            Some(mover ^ 1)
        }
    }

    fn one_wall_race_bound(&mut self) -> RaceBound {
        if !self.one_wall_race_enabled() || self.g.wl[0] + self.g.wl[1] != 1 {
            return RaceBound::Unknown;
        }
        self.race_outcome_stats.one_wall_calls += 1;
        let winner = if self.g.wl[self.g.turn] == 1 {
            self.one_wall_holder_winner()
        } else {
            self.one_wall_nonholder_winner()
        };
        match winner {
            Some(winner) => {
                self.race_outcome_stats.one_wall_decisive += 1;
                if winner == self.g.turn {
                    RaceBound::Lower(RACE_WIN_FLOOR)
                } else {
                    RaceBound::Upper(-RACE_WIN_FLOOR)
                }
            }
            None => {
                self.race_outcome_stats.one_wall_unknown += 1;
                RaceBound::Unknown
            }
        }
    }

    /// Exactly one side has 0 walls in hand: refuse-to-place race bounds.
    ///
    /// Arrival-time vs game-result: a pure-race projection is NOT always a cut.
    /// Only these two theorems are used:
    /// - opp has 0 walls and STM wins pure race → Lower (STM can refuse to place)
    /// - STM has 0 walls and STM loses pure race → Upper (opp can refuse to place)
    ///
    /// Bounds preserve exact projected DTM (`±(RACE_MATE - dtm)`), not just the floor.
    ///
    /// Not a cut: wallless side wins while the other still has walls (they can spoil);
    /// or walled side loses pure race (they can still spend walls to reverse).
    fn one_side_broke_race_bound(&mut self) -> RaceBound {
        if !self.race_proof {
            return RaceBound::Unknown;
        }
        let w0 = self.g.wl[0];
        let w1 = self.g.wl[1];
        // Exactly one side broke. Both-zero is the exact race path; both-armed declines.
        if (w0 == 0) == (w1 == 0) {
            return RaceBound::Unknown;
        }
        self.race_outcome_stats.broke_calls += 1;
        let Some(score) = self.zero_wall_exact_score_for_current_topology() else {
            self.race_outcome_stats.broke_unknown += 1;
            return RaceBound::Unknown;
        };
        let stm = self.g.turn;
        let opp = stm ^ 1;
        // Opp broke + STM wins pure race → Lower(exact projected win score).
        if self.g.wl[opp] == 0 && score > 0 {
            self.race_outcome_stats.broke_decisive += 1;
            self.race_outcome_stats.broke_lower += 1;
            return RaceBound::Lower(score);
        }
        // STM broke + STM loses pure race → Upper(exact projected loss score).
        if self.g.wl[stm] == 0 && score < 0 {
            self.race_outcome_stats.broke_decisive += 1;
            self.race_outcome_stats.broke_upper += 1;
            return RaceBound::Upper(score);
        }
        self.race_outcome_stats.broke_unknown += 1;
        RaceBound::Unknown
    }

    /// Sound monopoly-only subset for exactly two remaining walls.
    fn two_wall_monopoly_race_bound(&mut self) -> RaceBound {
        if !self.two_wall_race_enabled() || self.g.wl[0] + self.g.wl[1] != 2 {
            return RaceBound::Unknown;
        }
        self.race_outcome_stats.two_wall_calls += 1;

        let holder = self.g.turn;
        let winner = if self.g.wl[holder] != 2 || !self.pawns_are_race_separated() {
            None
        } else {
            match self.zero_wall_winner_for_current_topology() {
                Some(winner) if winner == holder => Some(holder),
                Some(_) if self.opponent_distance_to_goal() == 1 => {
                    let mut saw_legal_wall = false;
                    let mut saw_unknown = false;
                    let mut holder_can_win = false;
                    'walls: for wall_type in 0..2usize {
                        for slot in 0..64usize {
                            if !self.g.wall_legal(wall_type, slot) {
                                continue;
                            }
                            saw_legal_wall = true;
                            let mv = if wall_type == 0 {
                                MOVE_HW_BASE + slot as i16
                            } else {
                                MOVE_VW_BASE + slot as i16
                            };
                            self.g.make_move(mv);
                            let child_winner = self.one_wall_nonholder_winner();
                            self.g.unmake_move();
                            self.cached_stamp = -1;

                            match child_winner {
                                Some(winner) if winner == holder => {
                                    holder_can_win = true;
                                    break 'walls;
                                }
                                Some(_) => {}
                                None => saw_unknown = true,
                            }
                        }
                    }
                    if holder_can_win {
                        Some(holder)
                    } else if saw_legal_wall && !saw_unknown {
                        Some(holder ^ 1)
                    } else {
                        None
                    }
                }
                Some(_) | None => None,
            }
        };

        match winner {
            Some(winner) => {
                self.race_outcome_stats.two_wall_decisive += 1;
                if winner == self.g.turn {
                    RaceBound::Lower(RACE_WIN_FLOOR)
                } else {
                    RaceBound::Upper(-RACE_WIN_FLOOR)
                }
            }
            None => {
                self.race_outcome_stats.two_wall_unknown += 1;
                RaceBound::Unknown
            }
        }
    }

    fn try_hands_empty_endgame(&mut self, d_me_i: i32, d_opp_i: i32) -> HandsEmptyPipelineOutcome {
        // Stage 1: existing `race_tbl` LRU memo (probe only, no build).
        if self.race_proof {
            let (k_lo, k_hi) = self.race_topology_key();
            if let Some(slot) = self.race_tbl_lru_probe(k_lo, k_hi) {
                if let Some(score) = self.score_from_race_slot(slot) {
                    self.race_outcome_stats.resolved_memo += 1;
                    return HandsEmptyPipelineOutcome::Score(score);
                }
            }
        }

        // Stage 2: cached-distance Gate 1 (Service A).
        // Cheap proof bounds are consumed in `ab()`, where the real search
        // window is available. Static evaluation must not invent an exact
        // value from a lower or upper bound.

        // Stage 3: `race_tbl(false)` — LRU probe then budget-gated build.
        if self.race_proof {
            if let Some(slot) = self.race_tbl(false) {
                if let Some(score) = self.score_from_race_slot(slot) {
                    self.race_outcome_stats.resolved_race_tbl += 1;
                    return HandsEmptyPipelineOutcome::Score(score);
                }
            }
        }

        // Distance heuristic fallback (unproven).
        self.race_outcome_stats.resolved_race_heuristic += 1;
        if d_me_i <= d_opp_i {
            HandsEmptyPipelineOutcome::Score(3000 + (d_opp_i - d_me_i) * 50 - d_me_i)
        } else {
            HandsEmptyPipelineOutcome::Score(-3000 - (d_me_i - d_opp_i) * 50 + d_opp_i)
        }
    }

    #[inline(always)]
    fn ensure_nnue_wall_accumulators(&mut self, nw: &Net, b0: i32, b1: i32) {
        let cur_h = wall_slot_bits(&self.g.hw);
        let cur_v = wall_slot_bits(&self.g.vw);
        if b0 != self.np_b0 || b1 != self.np_b1v {
            crate::bench_instr::record(
                |b| &mut b.nnue_full_refresh,
                || {
                    self.np_acc0.fill(0.0);
                    self.np_acc1.fill(0.0);
                    let mut bits = cur_h;
                    while bits != 0 {
                        let s = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let o0 = (b0 as usize * 128 + s) * nw.h;
                        let o1 = (b1 as usize * 128 + NET_MIRS[s]) * nw.h;
                        for j in 0..nw.h {
                            self.np_acc0[j] += nw.w1c[o0 + j];
                            self.np_acc1[j] += nw.w1c[o1 + j];
                        }
                    }
                    let mut bits = cur_v;
                    while bits != 0 {
                        let s = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let o0 = (b0 as usize * 128 + 64 + s) * nw.h;
                        let o1 = (b1 as usize * 128 + 64 + NET_MIRS[s]) * nw.h;
                        for j in 0..nw.h {
                            self.np_acc0[j] += nw.w1c[o0 + j];
                            self.np_acc1[j] += nw.w1c[o1 + j];
                        }
                    }
                    self.np_hbits = cur_h;
                    self.np_vbits = cur_v;
                    self.np_b0 = b0;
                    self.np_b1v = b1;
                },
            );
        } else if cur_h != self.np_hbits || cur_v != self.np_vbits {
            crate::bench_instr::record(
                |b| &mut b.nnue_incr_update,
                || {
                    let mut bits = cur_h ^ self.np_hbits;
                    while bits != 0 {
                        let s = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let sg = if cur_h >> s & 1 != 0 { 1.0 } else { -1.0 };
                        let o0 = (b0 as usize * 128 + s) * nw.h;
                        let o1 = (b1 as usize * 128 + NET_MIRS[s]) * nw.h;
                        for j in 0..nw.h {
                            self.np_acc0[j] += sg * nw.w1c[o0 + j];
                            self.np_acc1[j] += sg * nw.w1c[o1 + j];
                        }
                    }
                    let mut bits = cur_v ^ self.np_vbits;
                    while bits != 0 {
                        let s = bits.trailing_zeros() as usize;
                        bits &= bits - 1;
                        let sg = if cur_v >> s & 1 != 0 { 1.0 } else { -1.0 };
                        let o0 = (b0 as usize * 128 + 64 + s) * nw.h;
                        let o1 = (b1 as usize * 128 + 64 + NET_MIRS[s]) * nw.h;
                        for j in 0..nw.h {
                            self.np_acc0[j] += sg * nw.w1c[o0 + j];
                            self.np_acc1[j] += sg * nw.w1c[o1 + j];
                        }
                    }
                    self.np_hbits = cur_h;
                    self.np_vbits = cur_v;
                },
            );
        }
    }

    /// RaceProof(c): budget-capped static win certificate for side `s` at the
    /// current position (`certify_win.js` 'all' mode = sound). Memoized; gen13
    /// runs it in node AND browser (cf. `certify.rs`). 1:1 with JS `certWin`.
    ///
    /// Memo: `1` = proven (permanent, sound). A failure is stored as `-work`
    /// (work = certify nodes burned, else the budget); it answers `false` only
    /// for weaker-or-equal retries (`bud <= work`); a richer call (bigger
    /// budget / fresh deadline) re-runs instead of inheriting a starved failure.
    fn cert_win(&mut self, s: usize, budget: u64, deadline_ms: u64) -> bool {
        // Grafted fast path: hands-empty is a pure pawn race — Titanium's tempo
        // classifier resolves it exactly (deterministic in the common case, a tiny
        // forward race-minimax only when paths overlap within 1 tempo). Sound: with
        // no walls the win-certificate reduces to the race outcome, so this returns
        // the same verdict as `certify` at a fraction of the node cost.
        if self.g.wl[0] == 0 && self.g.wl[1] == 0 {
            use crate::titanium::cert_bridge::hands_empty_race_stm_wins;
            if let Some(stm_wins) = hands_empty_race_stm_wins(&mut self.g) {
                return if s == self.g.turn {
                    stm_wins
                } else {
                    !stm_wins
                };
            }
        }
        let key = (
            self.g.hash_lo,
            self.g.hash_hi,
            s,
            self.g.wl[0],
            self.g.wl[1],
        );
        let bud: i64 = if budget == 0 { 2500 } else { budget as i64 };
        let prior = self.cw_cache.get(&key).copied();
        if let Some(c) = prior {
            if c == 1 {
                self.race_outcome_stats.resolved_cert_memo += 1;
                return true; // proven: permanent (sound)
            }
            if bud <= -(c as i64) {
                return false; // weaker-or-equal retry of a recorded failure
            }
            // richer retry: fall through and re-run
        }
        self.cw_think_calls += 1;
        let deadline = if deadline_ms > 0 {
            Some(Instant::now() + Duration::from_millis(deadline_ms))
        } else {
            None
        };
        let report = certify(
            &mut self.g,
            &CertifyOpts {
                budget: bud as u64,
                deadline,
                mode_pruned: false,
                slack: 2,
                side: Some(s),
                recommit: true,
            },
        );
        let res = report.proven == Some(s);
        let mut work = bud;
        if !res && report.nodes > 0 {
            work = report.nodes as i64; // deadline-starved: stamp only work done
        }
        if res {
            self.race_outcome_stats.resolved_cert_win += 1;
        }
        if self.cw_cache.len() > 16384 {
            self.cw_cache.clear();
        }
        if !res {
            if let Some(c) = prior {
                if -(c as i64) > work {
                    work = -(c as i64); // never weaken a recorded failure
                }
            }
        }
        self.cw_cache
            .insert(key, if res { 1 } else { -(work as i32) });
        res
    }

    /// Free proof lookup only. Used for low-wall transpositions where another
    /// path already paid for a certificate solve; never launches the solver.
    #[inline(always)]
    fn cert_win_cache_hit(&mut self, s: usize) -> bool {
        let key = (
            self.g.hash_lo,
            self.g.hash_hi,
            s,
            self.g.wl[0],
            self.g.wl[1],
        );
        if self.cw_cache.get(&key).copied() == Some(1) {
            self.race_outcome_stats.resolved_cert_memo += 1;
            return true;
        }
        false
    }

    /// Materialize the existing HalfPW child representation without computing
    /// route fields, legal-wall count, or the value projection. Probe/shadow only.
    fn current_hidden_features(&mut self) -> [f64; MAX_NET_H] {
        let nw = self.net;
        let b0 = NET_BKT[self.g.pawn[0]] as i32;
        let b1 = NET_BKT[NET_MIRC[self.g.pawn[1]]] as i32;
        self.ensure_nnue_wall_accumulators(nw, b0, b1);

        let mut hidden = [0.0; MAX_NET_H];
        if self.g.turn == 0 {
            let po = self.g.pawn[0] * nw.h;
            let px = self.g.pawn[1] * nw.h;
            for j in 0..nw.h {
                hidden[j] =
                    (nw.b1[j] + self.np_acc0[j] + nw.po[po + j] + nw.px[px + j]).clamp(0.0, 1.0);
            }
        } else {
            let po = NET_MIRC[self.g.pawn[1]] * nw.h;
            let px = NET_MIRC[self.g.pawn[0]] * nw.h;
            for j in 0..nw.h {
                hidden[j] =
                    (nw.b1[j] + self.np_acc1[j] + nw.po[po + j] + nw.px[px + j]).clamp(0.0, 1.0);
            }
        }
        hidden
    }

    /// Static/quiescence eval. `depth <= 0` = leaf (cert oracle eligible when gated).
    fn evaluate(&mut self, depth: i32) -> i32 {
        let _eval_timer = crate::bench_instr::OpTimer::start(|b| &mut b.evaluate);
        if self.net.route_active {
            crate::bench_instr::bump(|b| &mut b.route_active_eval);
        } else {
            crate::bench_instr::bump(|b| &mut b.route_inactive_eval);
        }
        let me = self.g.turn;
        let opp = 1 - me;
        let mut d_me_i = if me == 0 {
            self.d0[self.dist0_idx][self.g.pawn[0]] as i32
        } else {
            self.d1[self.dist1_idx][self.g.pawn[1]] as i32
        };
        let mut d_opp_i = if opp == 0 {
            self.d0[self.dist0_idx][self.g.pawn[0]] as i32
        } else {
            self.d1[self.dist1_idx][self.g.pawn[1]] as i32
        };
        let w_me_i = self.g.wl[me];
        let w_opp_i = self.g.wl[opp];
        let bff_d_me_i = d_me_i;
        let bff_d_opp_i = d_opp_i;

        if self.race_proof && w_me_i == 0 && w_opp_i == 0 {
            let d0 = &self.d0[self.dist0_idx];
            let d1 = &self.d1[self.dist1_idx];
            self.race_outcome_stats.jump_dist_calls += 1;
            if bff_tempo_margin_close(&self.g, d0, d1) {
                let ja = jump_aware_goal_distances(&mut self.g);
                self.race_outcome_stats.jump_dist_upgrades += 1;
                let new_me = if me == 0 { ja.d0 } else { ja.d1 };
                let new_opp = if opp == 0 { ja.d0 } else { ja.d1 };
                if new_me != u8::MAX {
                    d_me_i = new_me as i32;
                }
                if new_opp != u8::MAX {
                    d_opp_i = new_opp as i32;
                }
                let bff_fav_me = bff_d_me_i <= bff_d_opp_i;
                let jump_fav_me = d_me_i <= d_opp_i;
                if bff_fav_me != jump_fav_me {
                    self.race_outcome_stats.jump_dist_cuts_avoided += 1;
                }
            }
        }

        if w_me_i == 0 && w_opp_i == 0 && (!self.cert_eval_leaves_only || depth <= 0) {
            match self.try_hands_empty_endgame(d_me_i, d_opp_i) {
                HandsEmptyPipelineOutcome::Score(s) => return s,
            }
        }
        // Soft eval when the refuse-to-place theorem already proves the race.
        // Uses a mid-band score (not RACE_WIN_FLOOR) so search can still refine.
        if self.race_proof && (w_me_i == 0) != (w_opp_i == 0) {
            match self.one_side_broke_race_bound() {
                RaceBound::Lower(_) => return 1800,
                RaceBound::Upper(_) => return -1800,
                RaceBound::Exact(_) | RaceBound::Unknown => {}
            }
        }
        if self.race_proof
            && w_me_i + w_opp_i <= 2
            && (w_me_i + w_opp_i) > 0
            && self.cert_win_cache_hit(me)
        {
            return 2500;
        }

        let hash64 = (self.g.hash_hi as u64) << 32 | self.g.hash_lo as u64;
        let ec_idx = eval_cache_slot(hash64, self.eval_cache_bits);
        let ec_meta = ((self.g.wl[0] as u16) << 8) | (self.g.wl[1] as u16);
        {
            let e = &self.eval_cache[ec_idx];
            if e.key == hash64 && e.meta == ec_meta {
                #[cfg(feature = "eval_cache_baseline")]
                let out = e.val;
                #[cfg(not(feature = "eval_cache_baseline"))]
                let out = e.val as f64;
                crate::bench_instr::bump(|b| &mut b.eval_cache_hit);
                return self.evaluate_tail(out, depth, me, d_me_i, d_opp_i, w_me_i, w_opp_i);
            }
        }
        crate::bench_instr::bump(|b| &mut b.eval_cache_miss);

        let d_me = d_me_i as f64;
        let d_opp = d_opp_i as f64;
        let w_me = w_me_i as f64;
        let w_opp = w_opp_i as f64;
        let nw = self.net;
        let ws = &nw.ws;

        let mut out = crate::bench_instr::record(
            |b| &mut b.eval_misc_scalar,
            || {
                let pd = d_opp - d_me;
                let wd = w_me - w_opp;
                let mut out = ws[0]
                    + ws[1] * pd
                    + ws[2] * wd
                    + ws[3] * d_me
                    + ws[4] * d_opp
                    + ws[9] * pd * (w_me + w_opp) / 20.0
                    + ws[10] * wd * (d_me + d_opp) / 16.0;
                if w_opp_i == 0 {
                    out += ws[6];
                    if d_me <= d_opp {
                        out += ws[5];
                    }
                } else if w_me_i == 0 {
                    out += ws[8];
                    if d_opp <= d_me - 1.0 {
                        out += ws[7];
                    }
                }
                if d_opp <= 4.0 {
                    out += ws[11] * if w_me < 3.0 { w_me } else { 3.0 };
                }
                if d_me <= 4.0 {
                    out += ws[12] * if w_opp < 3.0 { w_opp } else { 3.0 };
                }
                out += ws[13] * pd * w_opp / 10.0;
                out
            },
        );
        let (route_score, _, _) = self.route_feature_score(nw);
        out += route_score;
        // CAT impact heatmap as a direct net input plane. Zeroed in legacy weights
        // (loader zero-pads) → `cat_active` false → NOT computed, so the live net is
        // byte-for-byte unaffected. Retrain-ready: a blob with learned `cat_heat`
        // weights activates it, giving the net the combined CAT signal alongside the
        // atomic route/near/contested planes (so it needn't reconstruct CAT itself).
        if nw.cat_active {
            let _cat_timer = crate::bench_instr::OpTimer::start(|b| &mut b.eval_cat_heat);
            if let Some(bridge) = self.bridge.as_ref() {
                let cat = crate::cat::build::build_catv5_heatmaps(&bridge.board);
                let (raw_me, raw_opp, prop_me, prop_opp) = if me == 0 {
                    (
                        &cat.witness_p0,
                        &cat.witness_p1,
                        &cat.propagated_p0,
                        &cat.propagated_p1,
                    )
                } else {
                    (
                        &cat.witness_p1,
                        &cat.witness_p0,
                        &cat.propagated_p1,
                        &cat.propagated_p0,
                    )
                };
                let mut cat_score = 0.0;
                for sq in 0..81usize {
                    let canon = if me == 0 { sq } else { NET_MIRC[sq] };
                    cat_score += nw.cat_raw_me[canon] * (f64::from(raw_me[sq]) / 4.0)
                        + nw.cat_raw_opp[canon] * (f64::from(raw_opp[sq]) / 4.0)
                        + nw.cat_propagated_me[canon] * (f64::from(prop_me[sq]) / 200.0)
                        + nw.cat_propagated_opp[canon] * (f64::from(prop_opp[sq]) / 200.0)
                        + nw.cat_propagated_combined[canon]
                            * (f64::from(cat.propagated[sq]) / 400.0);
                }
                out += cat_score;
            }
        }
        // ws[14] legal-wall-count input is retired from live search. The cheap
        // remaining-wall counts are already present as scalar features.
        // ws[15]: opponent corridor width on their goal field (matches halfpw.py).
        let width_opp = {
            let _width_timer = crate::bench_instr::OpTimer::start(|b| &mut b.eval_width_opp);
            if self.net.route_active {
                (if me == 0 {
                    width_in_layers(
                        &self.d1_layers[self.dist1_idx],
                        self.d1_layer_depth[self.dist1_idx],
                        d_opp_i as u8,
                    )
                } else {
                    width_in_layers(
                        &self.d0_layers[self.dist0_idx],
                        self.d0_layer_depth[self.dist0_idx],
                        d_opp_i as u8,
                    )
                }) as usize
            } else if me == 0 {
                self.d1[self.dist1_idx]
                    .iter()
                    .filter(|&&d| d as i32 == d_opp_i)
                    .count()
            } else {
                self.d0[self.dist0_idx]
                    .iter()
                    .filter(|&&d| d as i32 == d_opp_i)
                    .count()
            }
        } as f64;
        out += ws[15] * width_opp;
        // ws[16]/ws[17] path-cross inputs are retired from live search. They
        // cost a second legal-wall pass per eval and NNUE retraining is planned
        // to absorb/remap the feature slots.
        // CAT eval features (ws[18]/ws[19]) are DECOUPLED: computing them per leaf
        // (a full corridor-attention build + two legal-movegen passes in
        // best_pawn_cat_heats) was ~⅔ of total search time. CAT is being rebuilt
        // as a cheap BFF heatmap used for LMR move ordering only. Until that lands
        // and the net is retrained on CAT-free data, the two CAT inputs are 0 —
        // i.e. the live net runs as a non-CAT NNUE. Sanity check: if the engine
        // opens with a non-pawn move, the net needs retraining.
        // out += ws[18] * cat_me / 256.0 + ws[19] * cat_opp / 256.0;  // re-enable post-retrain

        let b0 = NET_BKT[self.g.pawn[0]] as i32;
        let b1 = NET_BKT[NET_MIRC[self.g.pawn[1]]] as i32;
        {
            let _nnue_prep = crate::bench_instr::OpTimer::start(|b| &mut b.eval_nnue_prep);
            self.ensure_nnue_wall_accumulators(nw, b0, b1);
        }
        crate::bench_instr::record(
            |b| &mut b.eval_nnue_infer,
            || {
                if me == 0 {
                    let po = self.g.pawn[0] * nw.h;
                    let px = self.g.pawn[1] * nw.h;
                    for j in 0..nw.h {
                        let h = nw.b1[j] + self.np_acc0[j] + nw.po[po + j] + nw.px[px + j];
                        out += nw.w2[j] * h.clamp(0.0, 1.0) * 200.0;
                    }
                } else {
                    let po = NET_MIRC[self.g.pawn[1]] * nw.h;
                    let px = NET_MIRC[self.g.pawn[0]] * nw.h;
                    for j in 0..nw.h {
                        let h = nw.b1[j] + self.np_acc1[j] + nw.po[po + j] + nw.px[px + j];
                        out += nw.w2[j] * h.clamp(0.0, 1.0) * 200.0;
                    }
                }
            },
        );
        let slot = &mut self.eval_cache[ec_idx];
        if slot.meta != u16::MAX && slot.key != hash64 {
            crate::bench_instr::bump(|b| &mut b.eval_cache_replace);
        }
        *slot = EvalCacheEntry {
            key: hash64,
            #[cfg(feature = "eval_cache_baseline")]
            val: out,
            #[cfg(not(feature = "eval_cache_baseline"))]
            val: out as f32,
            meta: ec_meta,
        };
        self.evaluate_tail(out, depth, me, d_me_i, d_opp_i, w_me_i, w_opp_i)
    }

    /// Cert/race floors applied on top of the cached pure static eval `out`.
    /// Budgeted + memoized state lives here, OUTSIDE the eval cache, so caching
    /// `out` cannot change cert behavior.
    #[allow(clippy::too_many_arguments)]
    fn evaluate_tail(
        &mut self,
        out: f64,
        depth: i32,
        me: usize,
        d_me_i: i32,
        d_opp_i: i32,
        w_me_i: i32,
        w_opp_i: i32,
    ) -> i32 {
        let _tail_timer = crate::bench_instr::OpTimer::start(|b| &mut b.eval_tail);
        // Integer centipawns (JS `out | 0` / halfpw `int(out)`).
        let mut ret = out as i32;
        let cert_ok = if self.cert_eval_leaves_only {
            depth <= 0 && w_me_i == 0 && w_opp_i == 0
        } else {
            w_me_i <= 2
        };
        if self.race_proof
            && cert_ok
            && ret < 2500
            && out > -700.0
            && out < 700.0
            && d_me_i <= d_opp_i + 1
        {
            let key = (
                self.g.hash_lo,
                self.g.hash_hi,
                me,
                self.g.wl[0],
                self.g.wl[1],
            );
            if (self.cw_think_calls < self.cw_cap || self.cw_cache.contains_key(&key))
                && self.cert_win(me, 1200, 0)
            {
                ret = 2500;
            }
        }
        // Correction history: apply the learned per-wall-structure eval bias.
        // Net-eval band only — the cert_win 2500 above and every early return
        // (proven/cert/mate scores) stay untouched, and the corrected value is
        // clamped inside the band so it can never impersonate a cert score.
        if ret > -2000 && ret < 2000 {
            let c = self.corr_hist[me][self.wall_corr_index()] as i32;
            ret = (ret + c).clamp(-1999, 1999);
        }
        ret
    }

    /// Race-root ordering: fastest win / slowest loss first; tie-break by net eval
    /// so we play the materially strongest move when plies-to-mate are equal.
    fn race_root_pick(&mut self, slot: usize, rv: i32) -> Option<(i16, i32, i32)> {
        let tbl = self.rc_tbl[slot].as_ref().expect("race slot").clone();
        let me = self.g.turn;
        let mut buf = [0i16; 16];
        let nm = self.g.gen_pawn_moves(&mut buf, 0);
        let mut best_m: i16 = -1;
        let mut best_v: i32 = 0;
        let mut best_key = i32::MIN;
        let mut best_eval = i32::MIN;
        for &c in &buf[..nm] {
            let cu = c as usize;
            let my_v = if (me == 0 && cu < 9) || (me == 1 && cu >= 72) {
                1
            } else {
                let v = tbl[if me == 0 {
                    (cu * 81 + self.g.pawn[1]) * 2 + 1
                } else {
                    (self.g.pawn[0] * 81 + cu) * 2
                }] as i32;
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
            self.g.make_move(c);
            self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_RACE_PICK);
            let d_me = if me == 0 {
                self.d0[self.dist0_idx][self.g.pawn[0]] as i32
            } else {
                self.d1[self.dist1_idx][self.g.pawn[1]] as i32
            };
            let d_opp = if me == 0 {
                self.d1[self.dist1_idx][self.g.pawn[1]] as i32
            } else {
                self.d0[self.dist0_idx][self.g.pawn[0]] as i32
            };
            let tie_eval = d_opp - d_me;
            self.g.unmake_move();
            self.cached_stamp = -1;
            if key > best_key || (key == best_key && tie_eval > best_eval) {
                best_key = key;
                best_eval = tie_eval;
                best_m = c;
                best_v = my_v;
            }
        }
        if best_m >= 0 && best_v == rv {
            Some((best_m, best_v, best_eval))
        } else {
            None
        }
    }

    /// Resolve a hands-empty root without conflating outcome and distance.
    ///
    /// The theorem/attractor tier first proves each candidate's outcome and
    /// supplies a walking ETA.  ETA intervals use a conservative +/-1 ply
    /// tolerance.  They select a move only when one interval is strictly best;
    /// overlapping finalists trigger the exact fixed-topology retrograde.  No
    /// ordinary alpha-beta simulation is used by this shortcut.
    fn semi_terminal_race_root(&mut self) -> Option<RaceRootSolution> {
        if !self.cheap_cert
            || self.g.wl[0] != 0
            || self.g.wl[1] != 0
            || self.g.pawn[0] < 9
            || self.g.pawn[1] >= 72
        {
            return None;
        }

        let root_side = self.g.turn;
        let mut moves = [0i16; 16];
        let move_count = self.g.gen_pawn_moves(&mut moves, 0);
        let mut candidates = Vec::with_capacity(move_count);
        let mut unresolved = 0usize;

        if self.race_scratch.is_none() {
            self.race_scratch = Some(Box::new(RaceScratch::new()));
        }

        for &mv in &moves[..move_count] {
            self.g.make_move(mv);
            let immediate =
                (root_side == 0 && self.g.pawn[0] < 9) || (root_side == 1 && self.g.pawn[1] >= 72);

            let candidate = if immediate {
                Some(RaceRootCandidate {
                    mv,
                    root_wins: true,
                    approximate_plies: Some(1),
                    exact_dtm: Some(1),
                })
            } else {
                let deduction = race_outcome_detailed(
                    &mut self.g,
                    self.race_scratch.as_mut().expect("race scratch"),
                );
                let root_wins = match deduction.bound {
                    RaceBound::Upper(_) => Some(true),
                    RaceBound::Lower(_) => Some(false),
                    RaceBound::Exact(v) => Some(v < 0),
                    RaceBound::Unknown => None,
                };
                root_wins.map(|root_wins| RaceRootCandidate {
                    mv,
                    root_wins,
                    approximate_plies: match deduction.estimated_plies {
                        PlyEstimate::Approx(v) => Some(v.saturating_add(1)),
                        PlyEstimate::Unknown => None,
                    },
                    exact_dtm: None,
                })
            };

            self.g.unmake_move();
            self.cached_stamp = -1;
            if let Some(candidate) = candidate {
                candidates.push(candidate);
            } else {
                unresolved += 1;
            }
        }

        // A direct goal is an exact one-ply result and cannot be beaten.
        if let Some(candidate) = candidates.iter().find(|c| c.exact_dtm == Some(1)) {
            return Some(RaceRootSolution {
                mv: candidate.mv,
                score: RACE_MATE - 1,
                info: RaceResultInfo::exact(1, 1),
                exact: true,
                legal_moves: move_count,
            });
        }

        let root_has_proven_win = candidates.iter().any(|c| c.root_wins);
        if !root_has_proven_win && unresolved != 0 {
            // A root loss requires every legal reply to be covered. Unknown is
            // a mandatory ordinary-search fallback, never an assumed loss.
            return None;
        }
        let finalists: Vec<_> = candidates
            .iter()
            .copied()
            .filter(|c| c.root_wins == root_has_proven_win)
            .collect();
        if finalists.is_empty() {
            return None;
        }

        let unique_by_interval = if finalists.len() == 1 {
            Some(finalists[0])
        } else {
            let mut definite = finalists
                .iter()
                .copied()
                .filter(|candidate| race_candidate_definitely_best(*candidate, &finalists));
            let first = definite.next();
            if definite.next().is_none() {
                first
            } else {
                None
            }
        };

        if let Some(candidate) = unique_by_interval {
            let outcome = if candidate.root_wins { 1 } else { -1 };
            return Some(RaceRootSolution {
                mv: candidate.mv,
                score: outcome as i32 * RACE_WIN_FLOOR,
                info: RaceResultInfo::approximate(outcome, candidate.approximate_plies),
                exact: false,
                legal_moves: move_count,
            });
        }

        // The +/-1 intervals overlap, so the approximate ETA cannot honestly
        // break the tie. Refine with the dedicated retrograde table, not the
        // general search. The table is exact for this immutable topology.
        let slot = self.race_tbl(true)?;
        let root_value = self.race_value(slot) as i32;
        let (mv, exact_dtm, _) = self.race_root_pick(slot, root_value)?;
        let outcome = root_value.signum() as i8;
        Some(RaceRootSolution {
            mv,
            score: if outcome > 0 {
                RACE_MATE - exact_dtm.abs()
            } else {
                -(RACE_MATE - exact_dtm.abs())
            },
            info: RaceResultInfo::exact(outcome, exact_dtm.unsigned_abs() as u16),
            exact: true,
            legal_moves: move_count,
        })
    }

    fn gen_moves(&mut self, ply: usize, depth: i32, tt_move: i16, out: &mut [i16; 160]) -> usize {
        crate::bench_instr::record(
            |b| &mut b.gen_moves,
            || self.gen_moves_inner(ply, depth, tt_move, out),
        )
    }

    fn gen_moves_inner(
        &mut self,
        ply: usize,
        depth: i32,
        tt_move: i16,
        out: &mut [i16; 160],
    ) -> usize {
        let check_legal = ply == 0;
        // MoveGen+ : Titanium legal movegen at EVERY node (perft-parity search).
        // Fully legal walls — no lazy seal checks needed downstream, and inner
        // nodes can never search (or suggest via TT) a Titanium-illegal move.
        // The CAT hybrid keeps its own filtered path at inner nodes.
        if self.ti_movegen && (check_legal || (!self.cat_walls && !self.dead_zone_prune)) {
            return self
                .bridge
                .as_mut()
                .expect("ti movegen needs bridge")
                .gen_legal_ace(out);
        }
        let mut n = self.g.gen_pawn_moves(out, 0);
        if self.g.wl[self.g.turn] <= 0 {
            return n;
        }
        if self.cat_walls && !check_legal {
            return self.gen_walls_cat_filtered(depth, tt_move, out, n);
        }
        if self.dead_zone_prune && !check_legal {
            return self.gen_walls_deadzone_filtered(out, n);
        }
        for slot in 0..64 {
            if check_legal {
                if self.g.wall_legal(0, slot) {
                    out[n] = MOVE_HW_BASE + slot as i16;
                    n += 1;
                }
                if self.g.wall_legal(1, slot) {
                    out[n] = MOVE_VW_BASE + slot as i16;
                    n += 1;
                }
            } else {
                // lazy: geometry only; path-seal checked when the move is searched
                if self.g.wall_fits(0, slot) {
                    crate::titanium::lazy_seal::lazy_seal_record_wall_generated();
                    out[n] = MOVE_HW_BASE + slot as i16;
                    n += 1;
                }
                if self.g.wall_fits(1, slot) {
                    crate::titanium::lazy_seal::lazy_seal_record_wall_generated();
                    out[n] = MOVE_VW_BASE + slot as i16;
                    n += 1;
                }
            }
        }
        n
    }

    /// Hybrid wall generation: lazy geometry + CAT relevance filter.
    ///
    /// CAT (multi-route corridor heat) only above the leaf layer — depth-1 nodes
    /// dominate the tree and only need witness-path tactics, not breadth
    /// (mirrors `search::alphabeta`). The TT move always survives the filter.
    fn gen_walls_cat_filtered(
        &mut self,
        depth: i32,
        tt_move: i16,
        out: &mut [i16; 160],
        mut n: usize,
    ) -> usize {
        let me = self.g.turn;
        let our_dist = if me == 0 {
            self.d0[self.dist0_idx][self.g.pawn[0]]
        } else {
            self.d1[self.dist1_idx][self.g.pawn[1]]
        };
        let opp_dist = if me == 0 {
            self.d1[self.dist1_idx][self.g.pawn[1]]
        } else {
            self.d0[self.dist0_idx][self.g.pawn[0]]
        };
        let white_dist = if me == 0 { our_dist } else { opp_dist };
        let black_dist = if me == 0 { opp_dist } else { our_dist };
        let opp_player = if me == 0 { Player::Two } else { Player::One };

        let bridge = self.bridge.as_mut().expect("cat bridge");
        let (cat, opp_path, opp_path_len, reachable) = if depth >= 2 {
            let data =
                crate::cat::build::build_corridor_search_data(&mut bridge.bfs, &bridge.board);
            (
                data.attention,
                data.opponent_path,
                data.opponent_path_len,
                data.reachable,
            )
        } else {
            let mut path = [0u8; 81];
            let path_len = get_shortest_path(&bridge.board, opp_player, &mut bridge.bfs, &mut path);
            let reachable = bridge.bfs.both_reachable_mask(&bridge.board);
            (CorridorAttention::default(), path, path_len, reachable)
        };
        let gap_zone = gap_play_zone_mask(reachable);
        let mut wall_candidates = [BoardMove::Pawn { row: 0, col: 0 }; 128];
        let mut wall_direct_heats = [0i32; 128];
        let mut wall_candidate_n = 0usize;

        for slot in 0..64 {
            for (wall_type, base) in [(0usize, MOVE_HW_BASE), (1usize, MOVE_VW_BASE)] {
                if !self.g.wall_fits(wall_type, slot) {
                    continue;
                }
                let m = base + slot as i16;
                let mv = move_id_to_board(m);
                wall_candidates[wall_candidate_n] = mv;
                wall_direct_heats[wall_candidate_n] = move_corridor_attention_with_path(
                    &mut bridge.board,
                    mv,
                    &cat,
                    white_dist,
                    black_dist,
                    &mut bridge.bfs,
                );
                wall_candidate_n += 1;
            }
        }

        for i in 0..wall_candidate_n {
            let mv = wall_candidates[i];
            let m = match mv {
                BoardMove::Wall {
                    row,
                    col,
                    orientation,
                } => {
                    let slot = i16::from(row) * 8 + i16::from(col);
                    match orientation {
                        WallOrientation::Horizontal => MOVE_HW_BASE + slot,
                        WallOrientation::Vertical => MOVE_VW_BASE + slot,
                    }
                }
                BoardMove::Pawn { .. } => continue,
            };
            let boosted_heat = move_corridor_attention_with_denial(
                &bridge.board,
                mv,
                &cat,
                &wall_candidates[..wall_candidate_n],
                &wall_direct_heats[..wall_candidate_n],
                wall_candidate_n,
            );
            let denied_hot_neighbor = boosted_heat > wall_direct_heats[i];
            let keep = m == tt_move
                || denied_hot_neighbor
                || wall_should_search(
                    mv,
                    &cat,
                    reachable,
                    gap_zone,
                    &mut bridge.board,
                    our_dist,
                    opp_dist,
                    &opp_path,
                    opp_path_len,
                    &mut bridge.bfs,
                );
            if keep {
                out[n] = m;
                n += 1;
            }
        }
        n
    }

    /// Wall generation with the SOUND dead-zone skip ONLY: emit every geometrically
    /// legal wall EXCEPT those whose every touched square is unreachable (a wall in
    /// a pure void). Those touch no pawn-reachable cell, block no path, and only
    /// burn inventory — never the best move, so pruning is NPS-only and can't cost
    /// Elo. A wall touching even one reachable square (incl. half-in-void) is kept.
    fn gen_walls_deadzone_filtered(&mut self, out: &mut [i16; 160], mut n: usize) -> usize {
        let bridge = self.bridge.as_mut().expect("dead-zone bridge");
        let reachable = bridge.bfs.both_reachable_mask(&bridge.board);
        for slot in 0..64 {
            for (wall_type, base) in [(0usize, MOVE_HW_BASE), (1usize, MOVE_VW_BASE)] {
                if !self.g.wall_fits(wall_type, slot) {
                    continue;
                }
                let m = base + slot as i16;
                if wall_in_dead_zone(move_id_to_board(m), reachable) {
                    continue;
                }
                out[n] = m;
                n += 1;
            }
        }
        n
    }

    fn order_moves(&self, ply: usize, moves: &mut [i16], tt_move: i16, cm_move: i16) {
        self.order_moves_prior(ply, moves, tt_move, cm_move, 0, None);
    }

    /// Ordering: TT > pawn progress > killers > countermove > history. CAT heat
    /// is a fallback prior ONLY for walls the history table is silent on
    /// (h == 0), and never for tail moves (≤ 10% of node max attention) — so
    /// insignificant walls get no ordering credit they haven't earned.
    fn order_moves_prior(
        &self,
        ply: usize,
        moves: &mut [i16],
        tt_move: i16,
        cm_move: i16,
        prev_move: i16,
        cat_prior: Option<(&[i32; HIST_SPAN], u32)>,
    ) {
        let dist_me = if self.g.turn == 0 {
            &self.d0[self.dist0_idx]
        } else {
            &self.d1[self.dist1_idx]
        };
        let k = &self.killers[ply];
        let n = moves.len();
        let mut sc = [0i32; 160];
        for i in 0..n {
            let m = moves[i];
            sc[i] = if m == tt_move {
                2_000_000_000
            } else if is_pawn_move(m) {
                let progress = 1_000_000 - dist_me[m as usize] as i32 * 1000;
                // Legacy mode intentionally keeps its exact pawn ordering. In
                // the SF-history A/B, history is only a ±499 tie-break; one
                // shortest-path step remains worth 1000 points.
                if self.sf_history {
                    progress + sf_pawn_history_tiebreak(self.move_hist(self.g.turn, m))
                } else {
                    progress
                }
            } else if m == k[0] {
                900_000
            } else if m == cm_move {
                870_000
            } else if m == k[1] {
                850_000
            } else {
                // main history + half-weight continuation history (reply
                // quality to the specific previous move); CAT heat stays the
                // fallback only when BOTH stat surfaces are silent.
                let h = self.move_hist(self.g.turn, m) + self.cont_hist_read(prev_move, m) / 2;
                if h != 0 {
                    h
                } else if let Some((heat, max_h)) = cat_prior {
                    let cm = heat[m as usize].max(0) as u32;
                    if cm * 10 > max_h {
                        cm as i32
                    } else {
                        0
                    }
                } else {
                    0
                }
            };
        }
        if ply == 0 {
            if let Some(order) = &self.opening_book_order {
                let attn = self.opening_book_attention.as_deref();
                for (rank, &bmv) in order.iter().enumerate() {
                    if let Some(pos) = moves.iter().position(|&m| m == bmv) {
                        let boost = attn.and_then(|a| a.get(rank).copied()).unwrap_or(0);
                        // Above TT move; win-rate + Ishtar tier sets relative priority.
                        let book_score = 2_050_000_000i32 + boost + (1000 - rank as i32);
                        sc[pos] = sc[pos].max(book_score);
                    }
                }
            }
            use crate::titanium::move_id_to_algebraic;
            use crate::titanium::opening_book::opening_move_would_be_denied;
            for i in 0..n {
                let alg = move_id_to_algebraic(moves[i]);
                if opening_move_would_be_denied(&self.g, &alg) {
                    sc[i] = i32::MIN / 4;
                }
            }
        }
        // stable insertion sort, descending — must match JS tie order exactly
        for a in 1..n {
            let mv = moves[a];
            let ms = sc[a];
            let mut b = a as isize - 1;
            while b >= 0 && sc[b as usize] < ms {
                moves[(b + 1) as usize] = moves[b as usize];
                sc[(b + 1) as usize] = sc[b as usize];
                b -= 1;
            }
            moves[(b + 1) as usize] = mv;
            sc[(b + 1) as usize] = ms;
        }
    }

    /// True when the current board hash already appeared in real game history
    /// (since the last wall — same rule as the in-search repetition cutoff).
    fn repeats_game_history(&self) -> bool {
        let lwp = self.g.last_wall_ply as isize;
        let mut gi = self.g.hist_len as isize * 2 - 4;
        while gi >= lwp * 2 {
            if self.g.hashes_u[gi as usize] == self.g.hash_lo
                && self.g.hashes_u[gi as usize + 1] == self.g.hash_hi
            {
                return true;
            }
            gi -= 2;
        }
        false
    }

    fn move_repeats_game_history(&mut self, m: i16) -> bool {
        self.g.make_move(m);
        let rep = self.repeats_game_history();
        self.g.unmake_move();
        rep
    }

    fn lmr_thread_id(&self) -> usize {
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            self.lazy_worker_id
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        {
            0
        }
    }

    fn q_search_jump_race_trigger(&self) -> bool {
        let me = self.g.turn;
        let from = self.g.pawn[me] as usize;
        let dcur = if me == 0 {
            self.d0[self.dist0_idx][from]
        } else {
            self.d1[self.dist1_idx][from]
        };
        let mut buf = [0i16; 16];
        let n = self.g.gen_pawn_moves(&mut buf, 0);
        for i in 0..n {
            let to = buf[i] as usize;
            let d = if me == 0 {
                self.d0[self.dist0_idx][to]
            } else {
                self.d1[self.dist1_idx][to]
            };
            if d <= dcur.saturating_sub(2) {
                return true;
            }
        }
        false
    }

    fn q_search_wall_dist_changed(&mut self, ply: usize, wall_move: i16) -> bool {
        if !is_wall_move(wall_move) || self.g.hist_len == 0 {
            return false;
        }
        let p0 = self.g.pawn[0] as usize;
        let p1 = self.g.pawn[1] as usize;
        let d0_now = self.d0[self.dist0_idx][p0];
        let d1_now = self.d1[self.dist1_idx][p1];
        self.g.unmake_move();
        self.refresh_dist_site(
            ply.saturating_sub(1),
            crate::bench_instr::REFRESH_SITE_QSEARCH_UNMAKE,
        );
        let d0_was = self.d0[self.dist0_idx][p0];
        let d1_was = self.d1[self.dist1_idx][p1];
        self.g.make_move(wall_move);
        self.refresh_dist_site(ply, crate::bench_instr::REFRESH_SITE_QSEARCH_REMAKE);
        d0_now != d0_was || d1_now != d1_was
    }

    fn q_search_should_extend(&mut self, ply: usize, prev_move: i16, static_ev: i32) -> bool {
        if ply == 0 {
            return false;
        }
        let parent_static = self.eval_stack[ply - 1];
        if parent_static == i32::MIN {
            return false;
        }
        if is_wall_move(prev_move) {
            if !self.q_search_wall_dist_changed(ply, prev_move) {
                return false;
            }
            if self.q_swing_cp > 0 && (static_ev + parent_static).abs() < self.q_swing_cp {
                return false;
            }
            return true;
        }
        self.q_search_jump_race_trigger()
    }

    fn ab(
        &mut self,
        depth: i32,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        allow_null: bool,
        prev_move: i16,
        q_left: i32,
    ) -> Result<i32, TimeUp> {
        self.nodes += 1;
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        if let Some(runtime) = self.lazy_runtime.as_ref() {
            runtime.global_nodes.fetch_add(1, Ordering::Relaxed);
        }
        self.check_time()?;
        self.sub_min[ply] = MAX_PLY as i32;
        let prev = 1 - self.g.turn;
        if (prev == 0 && self.g.pawn[0] < 9) || (prev == 1 && self.g.pawn[1] >= 72) {
            return Ok(-(MATE - ply as i32));
        }
        if ply >= MAX_PLY - 1 {
            // truncation-zero is unverified — taint ancestors (ZeroFence)
            self.sub_min[ply] = -1;
            self.sub_anc_lo[ply] = 0;
            self.sub_anc_hi[ply] = 0;
            return Ok(0);
        }
        self.path_lo[ply] = self.g.hash_lo;
        self.path_hi[ply] = self.g.hash_hi;
        if ply > 0 {
            // repetition: search line, then game history back to last wall
            for ri in (0..ply).rev() {
                if self.path_lo[ri] == self.g.hash_lo && self.path_hi[ri] == self.g.hash_hi {
                    // path-dependent zero: record the external dependency window
                    if (ri as i32) < self.sub_min[ply] {
                        self.sub_min[ply] = ri as i32;
                        self.sub_anc_lo[ply] = self.g.hash_lo;
                        self.sub_anc_hi[ply] = self.g.hash_hi;
                    }
                    return Ok(0);
                }
            }
            let lwp = self.g.last_wall_ply as isize;
            let mut gi = self.g.hist_len as isize * 2 - 4;
            while gi >= lwp * 2 {
                if self.g.hashes_u[gi as usize] == self.g.hash_lo
                    && self.g.hashes_u[gi as usize + 1] == self.g.hash_hi
                {
                    // game-history rep: path-independent, no taint
                    return Ok(0);
                }
                gi -= 2;
            }
        }

        self.maybe_refresh_dist_at_ab(ply);
        let nd0 = self.dist0_idx; // restored on every unmake
        let nd1 = self.dist1_idx;
        let nst = self.cached_stamp;
        let ndm_lo = self.dir_masks_key_lo;
        let ndm_hi = self.dir_masks_key_hi;
        let ndm_cache = self.dir_masks_cache;
        let pv_node = beta > alpha.saturating_add(1);
        // Exactly one side out of walls: refuse-to-place race cuts (covers [k,0]/[0,k]).
        if (self.g.wl[0] == 0) != (self.g.wl[1] == 0) {
            match self.one_side_broke_race_bound() {
                RaceBound::Lower(value) if value >= beta => {
                    self.race_outcome_stats.broke_cut_fail_high += 1;
                    return Ok(beta);
                }
                RaceBound::Upper(value) if value <= alpha => {
                    self.race_outcome_stats.broke_cut_fail_low += 1;
                    return Ok(alpha);
                }
                RaceBound::Lower(_)
                | RaceBound::Upper(_)
                | RaceBound::Exact(_)
                | RaceBound::Unknown => {}
            }
        }
        if self.g.wl[0] + self.g.wl[1] == 1 && (!self.one_wall_race_pv_only || pv_node) {
            match self.one_wall_race_bound() {
                RaceBound::Lower(value) if value >= beta => return Ok(beta),
                RaceBound::Upper(value) if value <= alpha => return Ok(alpha),
                RaceBound::Lower(_)
                | RaceBound::Upper(_)
                | RaceBound::Exact(_)
                | RaceBound::Unknown => {}
            }
        }
        if self.g.wl[0] + self.g.wl[1] == 2 && (!self.two_wall_race_pv_only || pv_node) {
            match self.two_wall_monopoly_race_bound() {
                RaceBound::Lower(value) if value >= beta => return Ok(beta),
                RaceBound::Upper(value) if value <= alpha => return Ok(alpha),
                RaceBound::Lower(_)
                | RaceBound::Upper(_)
                | RaceBound::Exact(_)
                | RaceBound::Unknown => {}
            }
        }
        if self.g.wl[0] == 0 && self.g.wl[1] == 0 {
            // Service A is a typed alpha/beta bound, not a static score.  It
            // may cut only when it crosses the current window.  A PV/wide
            // window falls through to the exact retrograde or ordinary search.
            if self.cheap_cert {
                let d0 = &self.d0[self.dist0_idx];
                let d1 = &self.d1[self.dist1_idx];
                let bound = crate::bench_instr::record(
                    |b| &mut b.eval_race_bound,
                    || race_outcome_with_dist(&self.g, d0, d1, &mut self.race_outcome_stats),
                );
                match bound {
                    RaceBound::Lower(v) if v >= beta => {
                        self.race_outcome_stats.resolved_gate1 += 1;
                        return Ok(beta);
                    }
                    RaceBound::Upper(v) if v <= alpha => {
                        self.race_outcome_stats.resolved_gate1_loss += 1;
                        return Ok(alpha);
                    }
                    RaceBound::Lower(_)
                    | RaceBound::Upper(_)
                    | RaceBound::Exact(_)
                    | RaceBound::Unknown => {}
                }
            }
            if let Some(score) = self.exact_hands_empty_score(false) {
                return Ok(score);
            }
        }
        if depth <= 0 {
            match self.wall_ignore_race_bound() {
                RaceBound::Lower(v) if v >= beta => {
                    self.race_outcome_stats.wall_ignore_cut_fail_high += 1;
                    return Ok(beta);
                }
                RaceBound::Upper(v) if v <= alpha => {
                    self.race_outcome_stats.wall_ignore_cut_fail_low += 1;
                    return Ok(alpha);
                }
                RaceBound::Lower(_)
                | RaceBound::Upper(_)
                | RaceBound::Exact(_)
                | RaceBound::Unknown => {}
            }
            let static_ev = self.evaluate(depth);
            if self.q_search
                && q_left > 0
                && ply > 0
                && static_ev > -2000
                && static_ev < 2000
                && self.q_search_should_extend(ply, prev_move, static_ev)
            {
                return self.ab(1, alpha, beta, ply, allow_null, prev_move, q_left - 1);
            }
            return Ok(static_ev);
        }

        // TT probe (typed, always-replace)
        let idx = (self.g.hash_lo & self.tt_mask) as usize;
        let mut tt_move: i16 = 0;
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let shared_entry = self
            .shared_tt
            .as_ref()
            .and_then(|tt| tt.probe(self.g.hash_lo, self.g.hash_hi));
        // Lazy SMP: when a shared TT is installed it is the ONLY TT — helper
        // workers carry no local TT (it would cost ~26MB each and blow the wasm
        // memory cap), so a shared miss must NOT fall back to the local arrays.
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let meta = match shared_entry {
            Some(entry) => entry.meta,
            None if self.shared_tt.is_some() => 0,
            None => self.tt_meta[idx],
        };
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let meta = self.tt_meta[idx];
        crate::bench_instr::bump(|b| &mut b.tt_probe);
        if meta != 0 && {
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            {
                shared_entry.is_some()
                    || (self.tt_key_hi[idx] == self.g.hash_hi
                        && self.tt_key_lo[idx] == self.g.hash_lo)
            }
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            {
                self.tt_key_hi[idx] == self.g.hash_hi && self.tt_key_lo[idx] == self.g.hash_lo
            }
        } {
            crate::bench_instr::bump(|b| &mut b.tt_hit);
            tt_move = (meta & 1023) as i16;
            let tdepth = tt_unpack_depth(meta);
            let tflag = (meta >> 10) & 3;
            if tdepth >= depth && ply > 0 {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                let mut es = match shared_entry {
                    Some(entry) => entry.score,
                    None => self.tt_score[idx],
                };
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                let mut es = self.tt_score[idx]; // mate scores stored node-relative
                if es > MATE - 2 * MAX_PLY as i32 {
                    es -= ply as i32;
                } else if es < -(MATE - 2 * MAX_PLY as i32) {
                    es += ply as i32;
                }
                if (tflag == 0) || (tflag == 1 && es >= beta) || (tflag == 2 && es <= alpha) {
                    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                    let tt_rep = match shared_entry {
                        Some(entry) => entry.rep,
                        None => self.tt_rep[idx],
                    };
                    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                    let tt_rep = self.tt_rep[idx];
                    if tt_rep == 0 {
                        crate::bench_instr::bump(|b| &mut b.tt_cutoff);
                        if self.ab_after_cat_child {
                            crate::bench_instr::bump_u64(|b| {
                                &mut b.cat_child_ab_tt_cutoff_before_eval
                            });
                        }
                        return Ok(es);
                    }
                    // tainted-zero entry: PLAIN ZeroFence ships with the anchor
                    // rescue disabled (`ghiAnchor=false` — the single min-ply
                    // anchor slot under-covers multi-dependency certificates),
                    // so a tainted entry never produces a score cutoff. The
                    // stored move is still used for ordering.
                }
            }
        }

        match self.wall_ignore_race_bound() {
            RaceBound::Lower(v) if v >= beta => {
                self.race_outcome_stats.wall_ignore_cut_fail_high += 1;
                return Ok(beta);
            }
            RaceBound::Upper(v) if v <= alpha => {
                self.race_outcome_stats.wall_ignore_cut_fail_low += 1;
                return Ok(alpha);
            }
            RaceBound::Lower(_)
            | RaceBound::Upper(_)
            | RaceBound::Exact(_)
            | RaceBound::Unknown => {}
        }

        // Static eval once per node (the internal eval cache absorbs
        // re-visits): feeds reverse futility, null move, the SF `improving`
        // flag, and the correction-history update at node completion.
        let static_ev = self.evaluate(depth);
        self.eval_stack[ply] = static_ev;
        // improving: static eval rose vs 2 plies ago (same side to move).
        // Stale same-slot values from sibling lines are tolerated, as in SF.
        let improving = ply >= 2
            && self.eval_stack[ply - 2] != i32::MIN
            && static_ev > self.eval_stack[ply - 2];

        // reverse futility: hopeless to fall below beta at shallow depth.
        // The ACE candidate is deliberately a single-axis change: it narrows
        // the depth gate and uses a fixed 100 cp/depth margin.
        if let Some(margin) =
            reverse_futility_margin(depth, improving, self.ace_rfp, self.ace_rfp_max_depth)
        {
            let adaptive_depth_four = depth == 4 && self.ace_rfp_max_depth == 4;
            let admissible_window = !adaptive_depth_four || beta == alpha.saturating_add(1);
            if admissible_window && beta > -2000 && beta < 2000 && static_ev - margin >= beta {
                return Ok(static_ev);
            }
        }

        // Opt-in ProbCut: when the cheap static eval is already comfortably
        // above an ordinary beta, ask a four-ply-shallower null-window search
        // to verify the fail-high. The verification reuses this position (no
        // make/unmake is needed), disables `allow_null` to prevent recursion,
        // and propagates TimeUp normally. Never run this at the root/PV node.
        if self.probcut && probcut_is_eligible(depth, alpha, beta, ply, allow_null, static_ev) {
            let verified = self.ab(
                depth - PROBCUT_REDUCTION,
                beta - 1,
                beta,
                ply,
                false,
                prev_move,
                q_left,
            )?;
            if verified >= beta {
                return Ok(beta); // fail-hard: never leak the speculative score
            }
        }

        // null move
        if allow_null && depth >= 3 && ply > 0 {
            let ev = static_ev;
            if ev >= beta {
                let z = &ZOBRIST;
                self.g.turn ^= 1;
                self.g.hash_lo ^= z.turn_lo;
                self.g.hash_hi ^= z.turn_hi;
                if let Some(bridge) = self.bridge.as_mut() {
                    // keep the mirrored board's side in sync (wall accounting)
                    bridge.board.side_to_move = bridge.board.side_to_move.opposite();
                }
                let res = self.ab(depth - 3, -beta, -beta + 1, ply + 1, false, 0, q_left);
                let z = &ZOBRIST;
                self.g.turn ^= 1;
                self.g.hash_lo ^= z.turn_lo;
                self.g.hash_hi ^= z.turn_hi;
                if let Some(bridge) = self.bridge.as_mut() {
                    bridge.board.side_to_move = bridge.board.side_to_move.opposite();
                }
                self.dist0_idx = nd0;
                self.dist1_idx = nd1;
                self.cached_stamp = nst;
                self.dir_masks_key_lo = ndm_lo;
                self.dir_masks_key_hi = ndm_hi;
                self.dir_masks_cache = ndm_cache;
                if self.sub_min[ply + 1] < self.sub_min[ply] {
                    self.sub_min[ply] = self.sub_min[ply + 1];
                    self.sub_anc_lo[ply] = self.sub_anc_lo[ply + 1];
                    self.sub_anc_hi[ply] = self.sub_anc_hi[ply + 1];
                }
                let ns = -res?;
                if ns >= beta && ns < MATE - 200 {
                    return Ok(beta);
                }
            }
        }

        let mut moves = [0i16; 160];
        let mut n = self.gen_moves(ply, depth, tt_move, &mut moves);
        if n == 0 {
            return Ok(self.evaluate(depth));
        }
        let cm_move = if prev_move >= 0 {
            self.cm[prev_move as usize]
        } else {
            0
        };

        // CAT impact heat, computed BEFORE ordering so it can serve as the
        // ordering prior for walls the history table knows nothing about.
        // Cheap BFF impact heatmap (bitboard path-set + flood): a move's
        // impact is a heatmap lookup (wall = hottest touched square).
        let mut heat_by_id = [0i32; HIST_SPAN];
        let mut max_move_impact = 0u32;
        // Walls and pawn moves are normalized separately: a pawn destination's
        // heat can dwarf every wall's heat, which previously made the best wall
        // on the board look like a cold ~9% move and forced it to depth 1.
        let mut max_wall_impact = 0u32;
        let cat_lmr_active = self.cat_lmr_v16 && depth >= 2 && n > 0;
        if cat_lmr_active {
            if let Some(bridge) = self.bridge.as_mut() {
                let cat = crate::cat::build::build_impact_heatmap(&bridge.board);
                for i in 0..n {
                    let mv = move_id_to_board(moves[i]);
                    let h = move_impact_heat(mv, &cat);
                    heat_by_id[moves[i] as usize] = h;
                    max_move_impact = max_move_impact.max(h.max(0) as u32);
                    if is_wall_move(moves[i]) {
                        max_wall_impact = max_wall_impact.max(h.max(0) as u32);
                    }
                }
            }
            // Cheap cold-start nudge (experimental, off by default): a wall
            // touching a cell on EITHER player's shortest-route set gets a
            // small flat bonus, on top of (not instead of) the CAT heat
            // above. Deliberately NOT "does this wall actually block the
            // path" (that needs walking the path's edges, the expensive
            // check this replaces) -- just "is it near/adjacent to the
            // route", from the already-leaf-cheap route masks (bit-parallel
            // flood, same cost class as the CAT heatmap itself). Small
            // enough (ROUTE_TOUCH_ORDER_BONUS) to nudge otherwise-similar
            // moves earlier without overriding a strong CAT signal or
            // distorting iterative deepening.
            if self.route_touch_ordering {
                let d0f = self.d0[self.dist0_idx];
                let d1f = self.d1[self.dist1_idx];
                let mut route0 = [0u8; 81];
                let mut route1 = [0u8; 81];
                let mut flank_scratch = [0u8; 81];
                fill_sparse_route_masks(
                    &self.g,
                    self.g.pawn[0],
                    &d0f,
                    &mut route0,
                    &mut flank_scratch,
                );
                fill_sparse_route_masks(
                    &self.g,
                    self.g.pawn[1],
                    &d1f,
                    &mut route1,
                    &mut flank_scratch,
                );
                for i in 0..n {
                    if is_pawn_move(moves[i]) {
                        continue; // pawn moves -- route-touch only makes sense for walls
                    }
                    if let crate::core::board::Move::Wall {
                        row,
                        col,
                        orientation,
                    } = move_id_to_board(moves[i])
                    {
                        if wall_touches_route(row, col, orientation, &route0, &route1) {
                            heat_by_id[moves[i] as usize] += ROUTE_TOUCH_ORDER_BONUS;
                            max_move_impact =
                                max_move_impact.max(heat_by_id[moves[i] as usize].max(0) as u32);
                            max_wall_impact =
                                max_wall_impact.max(heat_by_id[moves[i] as usize].max(0) as u32);
                        }
                    }
                }
            }
        }
        let cat_order_prior = if cat_lmr_active && max_move_impact > 0 {
            Some((&heat_by_id, max_move_impact))
        } else {
            None
        };
        self.order_moves_prior(
            ply,
            &mut moves[..n],
            tt_move,
            cm_move,
            prev_move,
            cat_order_prior,
        );
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        if ply == 0 {
            if let Some(root_moves) = self.lazy_root_moves.as_ref() {
                let allowed = self
                    .lazy_root_allowed
                    .min(root_moves.len())
                    .min(moves.len());
                for (dst, src) in moves.iter_mut().zip(root_moves.iter()).take(allowed) {
                    *dst = *src;
                }
                n = allowed;
            }
        }

        let mut cat_heats = [0i32; 160];
        for i in 0..n {
            cat_heats[i] = heat_by_id[moves[i] as usize];
        }

        let lazy_walls_active =
            ply > 0 && !(self.ti_movegen && !self.cat_walls && !self.dead_zone_prune);
        let lazy_seal_mode = crate::titanium::lazy_seal::LazySealMode::from_env();
        let mut lazy_seal = if lazy_walls_active {
            Some(crate::titanium::lazy_seal::LazySealNode::from_game(
                &self.g,
                lazy_seal_mode,
            ))
        } else {
            None
        };

        let mut best = i32::MIN; // JS -Infinity
        let mut best_move: i16 = 0;
        let mut flag = 2;
        // SF-history maluses need the walls that were actually SEARCHED at
        // this node (LMP/seal-check skips must not be penalized — they were
        // never tried, only ordered).
        let mut searched_walls = [0i16; 160];
        let mut searched_wall_count = 0usize;
        // Pawn candidates are tracked separately so wall continuation history
        // remains unchanged and only actually searched pawns receive maluses.
        let mut searched_pawns = [0i16; 160];
        let mut searched_pawn_count = 0usize;

        for i in 0..n {
            let m = moves[i];
            // Frontier LMP. The default v17 policy remains unchanged; ACE-LMP
            // is an opt-in, conservative extension to depth 3 with a later
            // wall-only cutoff. All existing tactical/TT/history safeguards
            // remain shared between both policies.
            let lmp_cutoff = if self.ace_lmp {
                depth <= 3 && i >= 24
            } else {
                depth <= 2 && i >= if improving { 14 } else { 8 }
            };
            if lmp_cutoff
                && ply > 0
                && is_wall_move(m)
                && m != tt_move
                && self.move_hist(self.g.turn, m) <= 0
                && best > -MATE + 200
            {
                continue;
            }
            // Seal check only needed for ACE's lazy pseudo-legal walls; with
            // MoveGen+ (Titanium legal gen at every node) all walls are legal.
            // The CAT and dead-zone paths both emit geometry-only (pseudo-legal)
            // walls, so they STILL need the seal check — only the pure ti_movegen
            // path (full legal gen) can skip it.
            if is_wall_move(m) {
                if let Some(seal) = lazy_seal.as_mut() {
                    let wt = if is_hwall_move(m) { 0 } else { 1 };
                    let slot = wall_slot(m);
                    if !seal.allows_lazy_wall(&mut self.g, wt, slot) {
                        continue; // sealing wall: pseudo-legal only
                    }
                }
            }
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            if ply == 0 {
                let original_idx = self
                    .lazy_root_visit_map
                    .as_ref()
                    .and_then(|map| map.get(i).copied())
                    .unwrap_or(i);
                self.lazy_root_visits.push(original_idx);
            }
            let probe_parent_hash = if self.reduction_probe_enabled {
                Some((self.g.hash_lo, self.g.hash_hi))
            } else {
                None
            };
            let mover = self.g.turn;
            let pre_d0 = self.d0[self.dist0_idx][self.g.pawn[0]];
            let pre_d1 = self.d1[self.dist1_idx][self.g.pawn[1]];
            crate::bench_instr::record(
                |b| &mut b.make_move,
                || {
                    self.g.make_move(m);
                    if let Some(bridge) = self.bridge.as_mut() {
                        bridge.push(m);
                    }
                },
            );
            // TT prefetch: the child's hash is now final; pull its TT lines
            // toward L1 while the LMR plan / EME gates below do their work,
            // masking the probe's memory latency inside the recursive call.
            // Local-array TT only (lazy-SMP workers carry 1-element dummies).
            #[cfg(target_arch = "x86_64")]
            {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                let local_tt = self.shared_tt.is_none();
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                let local_tt = true;
                if local_tt {
                    let idx = (self.g.hash_lo & self.tt_mask) as usize;
                    unsafe {
                        use std::arch::x86_64::{_mm_prefetch, _MM_HINT_T0};
                        _mm_prefetch(self.tt_meta.as_ptr().add(idx) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(self.tt_key_lo.as_ptr().add(idx) as *const i8, _MM_HINT_T0);
                        _mm_prefetch(self.tt_score.as_ptr().add(idx) as *const i8, _MM_HINT_T0);
                    }
                }
            }
            let new_depth = depth - 1;
            let result = if ply == 0 && self.multipv > 1 {
                self.ab(new_depth, -INF, INF, ply + 1, true, m, q_left)
                    .map(|s| -s)
            } else if self.eme
                && i > 0
                && i <= ACE_EME_TOP_MOVES
                && depth >= ACE_LMR_MIN_DEPTH
                && is_wall_move(m)
                && m != tt_move
            {
                // EME — extend only the top ordered walls (see ACE_EME_TOP_MOVES)
                let ext = ace_graduated_eme_extension(i, depth);
                let ed = new_depth + ext;
                self.ab(ed, -beta, -alpha, ply + 1, true, m, q_left)
                    .map(|s| -s)
            } else if i >= ACE_LMR_AFTER_MOVE
                && depth >= ACE_LMR_MIN_DEPTH
                && is_wall_move(m)
                && m != tt_move
            {
                let attention_ratio = if cat_lmr_active && max_wall_impact > 0 {
                    cat_heats[i].max(0) as f64 / max_wall_impact as f64
                } else {
                    1.0
                };
                let wall_opponent_delay = 0;
                let v16_plan = if cat_lmr_active {
                    plan_v16_wall_lmr(i, depth, new_depth, attention_ratio, 0, 0)
                } else {
                    let ace_base = ace_graduated_lmr_reduction(i, depth);
                    let final_reduction = ace_base.min((new_depth - 1).max(0));
                    super::v16_lmr::V16LmrPlan {
                        ace_base_reduction: ace_base,
                        hard_override: V16HardOverride::None,
                        final_reduction,
                        child_depth_used: (new_depth - final_reduction).max(0),
                    }
                };
                let path_plan =
                    if self.cat_path_lmr && cat_lmr_active && v16_plan.final_reduction > 0 {
                        crate::bench_instr::bump_u64(|b| &mut b.cat_edge_test_calls);
                        let (refresh0, refresh1) = wall_incr_refresh_flags(
                            &self.d0[self.dist0_idx],
                            &self.d1[self.dist1_idx],
                            m,
                        );
                        if self.should_skip_cat_no_edge_refresh() && !refresh0 && !refresh1 {
                            crate::bench_instr::bump_u64(|b| &mut b.cat_no_edge_skip);
                            apply_lmr_path_correction(
                                v16_plan.final_reduction.max(0) as u32,
                                new_depth.max(0) as u32,
                                0,
                                attention_ratio,
                                false,
                            )
                        } else {
                            // Distance refresh needed for exact race_gain scalars.
                            self.refresh_dist_site(
                                ply + 1,
                                crate::bench_instr::REFRESH_SITE_CAT_PATH_LMR,
                            );
                            self.pending_cat_child_ply = Some(ply + 1);
                            let post_d0 = self.d0[self.dist0_idx][self.g.pawn[0]];
                            let post_d1 = self.d1[self.dist1_idx][self.g.pawn[1]];
                            let (pre_our, pre_opp, post_our, post_opp) = if mover == 0 {
                                (pre_d0, pre_d1, post_d0, post_d1)
                            } else {
                                (pre_d1, pre_d0, post_d1, post_d0)
                            };
                            let (_, _, race_gain) = super::cat_index_lmr::compute_race_gain(
                                pre_our, pre_opp, post_our, post_opp,
                            );
                            apply_lmr_path_correction(
                                v16_plan.final_reduction.max(0) as u32,
                                new_depth.max(0) as u32,
                                race_gain,
                                attention_ratio,
                                false,
                            )
                        }
                    } else {
                        apply_lmr_path_correction(
                            v16_plan.final_reduction.max(0) as u32,
                            new_depth.max(0) as u32,
                            0,
                            attention_ratio,
                            true,
                        )
                    };
                let red = path_plan.final_reduction as i32;
                let child_depth_used = (new_depth - red).max(0);
                if self.reduction_sidecar.is_some() {
                    let started = Instant::now();
                    let hidden_full = self.current_hidden_features();
                    // reduction_sidecar is a separate, independently-calibrated
                    // model with a fixed 32-wide input contract -- unrelated to
                    // the main net's hidden width (which may now be > 32).
                    let hidden: [f64; 32] = hidden_full[..32].try_into().unwrap();
                    let context = [
                        ((depth - 1).max(0) as f64 / 30.0).clamp(0.0, 1.0),
                        (i as f64 / 128.0).clamp(0.0, 1.0),
                        (red as f64 / 4.0).clamp(0.0, 1.0),
                        if is_hwall_move(m) { 1.0 } else { 0.0 },
                        if is_vwall_move(m) { 1.0 } else { 0.0 },
                    ];
                    let sidecar = self.reduction_sidecar.as_ref().expect("checked above");
                    let probability = sidecar.predict(&hidden, &context);
                    self.reduction_shadow_stats.evaluations += 1;
                    self.reduction_shadow_stats.hypothetical_activations +=
                        u64::from(sidecar.would_activate(probability));
                    self.reduction_shadow_stats.inference_nanos +=
                        started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
                }
                let probe_ordinal =
                    if self.reduction_probe_enabled && depth >= self.reduction_probe_min_depth {
                        let ordinal = self.reduction_probe_next;
                        self.reduction_probe_next += 1;
                        Some(ordinal)
                    } else {
                        None
                    };
                let extra_reduction = probe_ordinal
                    .is_some_and(|ordinal| self.reduction_probe_target == Some(ordinal));
                let rd = (child_depth_used - i32::from(extra_reduction)).max(0);
                let nodes_before = self.nodes;
                let mut verification_triggered = false;
                let pipeline_result =
                    match self.ab(rd, -alpha - 1, -alpha, ply + 1, true, m, q_left) {
                        Ok(s) => {
                            let mut score = -s;
                            if score > alpha {
                                verification_triggered = true;
                                match self.ab(new_depth, -beta, -alpha, ply + 1, true, m, q_left) {
                                    Ok(s2) => score = -s2,
                                    Err(e) => {
                                        self.unwind_move(nd0, nd1, nst, ndm_lo, ndm_hi, ndm_cache);
                                        return Err(e);
                                    }
                                }
                            }
                            Ok(score)
                        }
                        Err(e) => Err(e),
                    };
                if let (Some(ordinal), Ok(score), Some((parent_hash_lo, parent_hash_hi))) =
                    (probe_ordinal, pipeline_result.as_ref(), probe_parent_hash)
                {
                    let should_record = self.reduction_probe_events.len()
                        < self.reduction_probe_limit
                        && (self.reduction_probe_target.is_none()
                            || self.reduction_probe_target == Some(ordinal));
                    if should_record {
                        let hidden = self.current_hidden_features();
                        self.reduction_probe_events.push(ReductionProbeEvent {
                            ordinal,
                            parent_hash_lo,
                            parent_hash_hi,
                            child_hash_lo: self.g.hash_lo,
                            child_hash_hi: self.g.hash_hi,
                            mv: m,
                            depth,
                            ply,
                            alpha,
                            beta,
                            move_index: i,
                            base_reduction: v16_plan.ace_base_reduction,
                            applied_extra_reduction: extra_reduction,
                            verification_triggered,
                            self_gain: 0,
                            opponent_delay: wall_opponent_delay,
                            race_gain: 0,
                            path_adjustment: v16_plan.final_reduction - v16_plan.ace_base_reduction,
                            final_reduction: red,
                            thread_aggression_percent: cat_lmr_tuning_percent(),
                            score: *score,
                            nodes: self.nodes.saturating_sub(nodes_before),
                            hidden,
                            total_legal_moves: n,
                            history_score: self.history_tbl[m as usize],
                        });
                    }
                }
                pipeline_result
            } else if self.cat_lmr_v16
                && is_pawn_move(m)
                && i > 0
                && depth >= ACE_LMR_MIN_DEPTH
                && m != tt_move
            {
                // Pawn moves do not change wall topology, so the parent distance
                // fields remain valid after the pawn coordinate changes.
                let post_d0 = self.d0[nd0][self.g.pawn[0]];
                let post_d1 = self.d1[nd1][self.g.pawn[1]];
                let (pre_our, post_our) = if mover == 0 {
                    (pre_d0, post_d0)
                } else {
                    (pre_d1, post_d1)
                };
                let self_gain = i32::from(pre_our) - i32::from(post_our);
                if let Some(v16_plan) = plan_v16_pawn_lmr(i, depth, new_depth, self_gain) {
                    let rd = v16_plan.child_depth_used;
                    match self.ab(rd, -alpha - 1, -alpha, ply + 1, true, m, q_left) {
                        Ok(s) => {
                            let mut score = -s;
                            if score > alpha {
                                match self.ab(new_depth, -beta, -alpha, ply + 1, true, m, q_left) {
                                    Ok(s2) => score = -s2,
                                    Err(e) => {
                                        self.unwind_move(nd0, nd1, nst, ndm_lo, ndm_hi, ndm_cache);
                                        return Err(e);
                                    }
                                }
                            }
                            Ok(score)
                        }
                        Err(e) => Err(e),
                    }
                } else {
                    match self.ab(new_depth, -alpha - 1, -alpha, ply + 1, true, m, q_left) {
                        Ok(s) => {
                            let mut score = -s;
                            if score > alpha && score < beta {
                                match self.ab(new_depth, -beta, -alpha, ply + 1, true, m, q_left) {
                                    Ok(s2) => score = -s2,
                                    Err(e) => {
                                        self.unwind_move(nd0, nd1, nst, ndm_lo, ndm_hi, ndm_cache);
                                        return Err(e);
                                    }
                                }
                            }
                            Ok(score)
                        }
                        Err(e) => Err(e),
                    }
                }
            } else if i > 0 {
                match self.ab(new_depth, -alpha - 1, -alpha, ply + 1, true, m, q_left) {
                    Ok(s) => {
                        let mut score = -s;
                        if score > alpha && score < beta {
                            match self.ab(new_depth, -beta, -alpha, ply + 1, true, m, q_left) {
                                Ok(s2) => score = -s2,
                                Err(e) => {
                                    self.unwind_move(nd0, nd1, nst, ndm_lo, ndm_hi, ndm_cache);
                                    return Err(e);
                                }
                            }
                        }
                        Ok(score)
                    }
                    Err(e) => Err(e),
                }
            } else {
                self.ab(new_depth, -beta, -alpha, ply + 1, true, m, q_left)
                    .map(|s| -s)
            };
            self.unwind_move(nd0, nd1, nst, ndm_lo, ndm_hi, ndm_cache);
            if self.sub_min[ply + 1] < self.sub_min[ply] {
                self.sub_min[ply] = self.sub_min[ply + 1];
                self.sub_anc_lo[ply] = self.sub_anc_lo[ply + 1];
                self.sub_anc_hi[ply] = self.sub_anc_hi[ply + 1];
            }
            let score = result?;
            if ply == 0 {
                if let Some((_, sc)) = self.stream_root_moves.iter_mut().find(|(mv, _)| *mv == m) {
                    if score > *sc {
                        *sc = score;
                    }
                } else {
                    self.stream_root_moves.push((m, score));
                }
            }
            if self.sf_history {
                if is_wall_move(m) {
                    searched_walls[searched_wall_count] = m;
                    searched_wall_count += 1;
                } else {
                    searched_pawns[searched_pawn_count] = m;
                    searched_pawn_count += 1;
                }
            }

            // RaceProof(b): best non-wall root alternative
            if ply == 0 && is_pawn_move(m) && score > self.root_pawn_score {
                self.root_pawn_score = score;
                self.root_pawn_best = m;
            }

            let prefer_non_repeat = ply == 0
                && score == best
                && best_move != 0
                && self.move_repeats_game_history(best_move)
                && !self.move_repeats_game_history(m);

            if score > best || prefer_non_repeat {
                best = score;
                best_move = m;
                if score > alpha || prefer_non_repeat {
                    alpha = score;
                    flag = 0;
                    if ply == 0 {
                        self.root_best = m;
                        self.root_score = score;
                        // New best move at root → push an info-card update now
                        // (forced; bypasses the periodic throttle).
                        if self.stream_last_best != m {
                            self.stream_last_best = m;
                            self.stream_root_score = score;
                            self.emit_stream_progress(true);
                        }
                    }
                    if alpha >= beta {
                        flag = 1;
                        if is_wall_move(m) {
                            if self.killers[ply][0] != m {
                                self.killers[ply][1] = self.killers[ply][0];
                                self.killers[ply][0] = m;
                            }
                            self.history_tbl[m as usize] += depth * depth;
                            if self.history_tbl[m as usize] > 100_000_000 {
                                for h in self.history_tbl.iter_mut() {
                                    *h >>= 1;
                                }
                            }
                        }
                        if self.sf_history {
                            // Side-split gravity credit for the cutoff wall,
                            // maluses for every wall searched before it (on a
                            // pawn cutoff the bonus is skipped but the tried
                            // walls still failed against "just walk" — demote
                            // them too). Continuation history mirrors both,
                            // keyed by the previous move.
                            let stm = self.g.turn;
                            let bonus = depth * depth;
                            if is_wall_move(m) {
                                self.sf_hist_apply(stm, m, bonus);
                                if prev_move >= 0 {
                                    self.cont_hist_apply(prev_move, m, bonus);
                                }
                            } else {
                                self.sf_hist_apply(stm, m, bonus);
                            }
                            for j in 0..searched_wall_count {
                                let pm = searched_walls[j];
                                if pm != m {
                                    self.sf_hist_apply(stm, pm, -bonus);
                                    if prev_move >= 0 {
                                        self.cont_hist_apply(prev_move, pm, -bonus);
                                    }
                                }
                            }
                            // The cutoff pawn is already in this array, so the
                            // inequality leaves its bonus intact while every
                            // prior pawn that was truly searched gets a malus.
                            // Skipped LMP/seal-check moves never enter it.
                            for j in 0..searched_pawn_count {
                                let pm = searched_pawns[j];
                                if pm != m {
                                    self.sf_hist_apply(stm, pm, -bonus);
                                }
                            }
                        }
                        if prev_move >= 0 {
                            self.cm[prev_move as usize] = m;
                        }
                        break;
                    }
                }
            }
        }

        if best == i32::MIN {
            return Ok(self.evaluate(depth)); // all pseudo-legal moves were sealing walls
        }
        let mut ts = best; // store mate scores node-relative
        if ts > MATE - 2 * MAX_PLY as i32 {
            ts += ply as i32;
        } else if ts < -(MATE - 2 * MAX_PLY as i32) {
            ts -= ply as i32;
        }
        // Correction history update: teach the static eval its bias on this
        // wall structure when the search verdict can actually contradict it
        // (SF condition: a fail-low can't prove the static was too low, a
        // fail-high can't prove it was too high). Weight grows with depth.
        if static_ev > -2000
            && static_ev < 2000
            && best > -2000
            && best < 2000
            && !(flag == 2 && best >= static_ev)
            && !(flag == 1 && best <= static_ev)
        {
            let idx = self.wall_corr_index();
            let stm = self.g.turn;
            let w = (depth + 1).min(16);
            let step = (best - static_ev) * w / 64;
            let e = &mut self.corr_hist[stm][idx];
            *e = ((*e as i32) + step).clamp(-CORR_MAX, CORR_MAX) as i16;
        }
        // ZeroFence-A store: claim leans on an external (path-dependent) rep-0
        let mut sf = flag;
        let mut rb = 0u8;
        if self.sub_min[ply] < ply as i32 {
            if best > 0 {
                if sf == 0 {
                    sf = 1;
                } else if sf == 2 {
                    rb = 1;
                }
            } else if best < 0 {
                if sf == 0 {
                    sf = 2;
                } else if sf == 1 {
                    rb = 1;
                }
            } else {
                rb = 1;
            }
        }
        // Depth-preferred replacement (gen-aware when pure_mode=false).
        // Recompute idx: a child may have grown the TT (adaptive path) after our probe.
        let idx = (self.g.hash_lo & self.tt_mask) as usize;
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        if let Some(shared) = self.shared_tt.as_ref() {
            crate::bench_instr::bump(|b| &mut b.tt_store);
            shared.store(
                self.g.hash_lo,
                self.g.hash_hi,
                self.tt_gen,
                self.pure_mode,
                SharedTtEntry {
                    key_hi: self.g.hash_hi,
                    key_lo: self.g.hash_lo,
                    meta: best_move as i32 | (sf << 10) | tt_pack_depth(depth),
                    score: ts,
                    rep: rb,
                    anc_lo: if rb != 0 { self.sub_anc_lo[ply] } else { 0 },
                    anc_hi: if rb != 0 { self.sub_anc_hi[ply] } else { 0 },
                    entry_gen: self.tt_gen,
                },
            );
        } else {
            let was_empty = self.tt_meta[idx] == 0;
            let stale_gen = !self.pure_mode && !was_empty && self.tt_entry_gen[idx] != self.tt_gen;
            let deeper = !was_empty
                && !stale_gen
                && depth.clamp(0, TT_DEPTH_MAX) >= tt_unpack_depth(self.tt_meta[idx]);
            if was_empty || stale_gen || deeper {
                crate::bench_instr::bump(|b| &mut b.tt_store);
                self.tt_key_hi[idx] = self.g.hash_hi;
                self.tt_key_lo[idx] = self.g.hash_lo;
                self.tt_meta[idx] = best_move as i32 | (sf << 10) | tt_pack_depth(depth);
                self.tt_score[idx] = ts;
                self.tt_rep[idx] = rb;
                self.tt_entry_gen[idx] = self.tt_gen;
                if rb != 0 {
                    self.tt_anc_lo[idx] = self.sub_anc_lo[ply];
                    self.tt_anc_hi[idx] = self.sub_anc_hi[ply];
                }
                // Overflow-driven cache-tier growth (idx is dead after this — safe to grow).
                if was_empty {
                    self.tt_filled += 1;
                    if self.tt_adaptive
                        && self.tt_bits < self.tt_max
                        && self.tt_filled.saturating_mul(2) >= (1usize << self.tt_bits)
                    {
                        self.tt_grow();
                    }
                }
            }
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        {
            let was_empty = self.tt_meta[idx] == 0;
            let stale_gen = !self.pure_mode && !was_empty && self.tt_entry_gen[idx] != self.tt_gen;
            let deeper = !was_empty
                && !stale_gen
                && depth.clamp(0, TT_DEPTH_MAX) >= tt_unpack_depth(self.tt_meta[idx]);
            if was_empty || stale_gen || deeper {
                crate::bench_instr::bump(|b| &mut b.tt_store);
                self.tt_key_hi[idx] = self.g.hash_hi;
                self.tt_key_lo[idx] = self.g.hash_lo;
                self.tt_meta[idx] = best_move as i32 | (sf << 10) | tt_pack_depth(depth);
                self.tt_score[idx] = ts;
                self.tt_rep[idx] = rb;
                self.tt_entry_gen[idx] = self.tt_gen;
                if rb != 0 {
                    self.tt_anc_lo[idx] = self.sub_anc_lo[ply];
                    self.tt_anc_hi[idx] = self.sub_anc_hi[ply];
                }
                if was_empty {
                    self.tt_filled += 1;
                    if self.tt_adaptive
                        && self.tt_bits < self.tt_max
                        && self.tt_filled.saturating_mul(2) >= (1usize << self.tt_bits)
                    {
                        self.tt_grow();
                    }
                }
            }
        }
        Ok(best)
    }

    /// Restore after a time abort mid-move (JS `finally` semantics).
    fn unwind_move(
        &mut self,
        nd0: usize,
        nd1: usize,
        nst: i32,
        ndm_lo: u32,
        ndm_hi: u32,
        ndm_cache: DirMasks,
    ) {
        crate::bench_instr::record(
            |b| &mut b.unmake_move,
            || {
                self.g.unmake_move();
                if let Some(bridge) = self.bridge.as_mut() {
                    bridge.pop();
                }
            },
        );
        self.dist0_idx = nd0;
        self.dist1_idx = nd1;
        self.cached_stamp = nst;
        self.dir_masks_key_lo = ndm_lo;
        self.dir_masks_key_hi = ndm_hi;
        self.dir_masks_cache = ndm_cache;
    }

    /// Lost-position root defense: full-depth search of every legal root move with
    /// stubborn-loser move selection (slowest proven loss, static-eval tie-break).
    fn root_defense_verify(&mut self, depth: i32) -> Result<i32, TimeUp> {
        // Invalidate shallow LMR-reduced root-move TT entries from the iteration
        // that just completed so every candidate is searched at full depth.
        if !self.pure_mode && !self.is_pondering {
            self.tt_gen = self.tt_gen.wrapping_add(1);
        }
        self.root_defense_diag.clear();
        let root_side = self.g.turn;
        let mut moves = [0i16; 160];
        let tt_hint = if self.root_best >= 0 {
            self.root_best
        } else {
            0
        };
        let n = self.gen_moves(0, depth, tt_hint, &mut moves);
        if n == 0 {
            return Ok(self.root_score);
        }
        self.order_moves(0, &mut moves[..n], tt_hint, 0);
        let n = crate::titanium::opening_book::filter_denied_opening_legal_moves(
            &self.g, &mut moves, n,
        );
        if n == 0 {
            return Ok(self.root_score);
        }

        let child_depth = depth - 1;
        let mut best_move = moves[0];
        let mut best_score = i32::MIN;
        let mut best_static = i32::MIN;
        let mut best_own_dist_after = i32::MAX;
        let mut best_opp_dist_after = i32::MIN;
        let mut best_order = 0usize;

        self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_ROOT_DEF_INIT);
        let own_dist_before: i32 = if root_side == 0 {
            self.d0[self.dist0_idx][self.g.pawn[0]] as i32
        } else {
            self.d1[self.dist1_idx][self.g.pawn[1]] as i32
        };

        for i in 0..n {
            if Instant::now() >= self.deadline {
                return Err(TimeUp);
            }
            let m = moves[i];
            let nodes_before = self.nodes;

            self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_ROOT_DEF_BEFORE);
            let nd0 = self.dist0_idx;
            let nd1 = self.dist1_idx;
            let nst = self.cached_stamp;
            let ndm_lo = self.dir_masks_key_lo;
            let ndm_hi = self.dir_masks_key_hi;
            let ndm_cache = self.dir_masks_cache;

            crate::bench_instr::record(
                |b| &mut b.make_move,
                || {
                    self.g.make_move(m);
                    if let Some(bridge) = self.bridge.as_mut() {
                        bridge.push(m);
                    }
                },
            );
            self.refresh_dist_site(1, crate::bench_instr::REFRESH_SITE_ROOT_DEF_AFTER);
            let static_eval = {
                let ev = self.evaluate(0);
                if self.g.turn == root_side {
                    ev
                } else {
                    -ev
                }
            };
            let own_dist_after: i32 = if root_side == 0 {
                self.d0[self.dist0_idx][self.g.pawn[0]] as i32
            } else {
                self.d1[self.dist1_idx][self.g.pawn[1]] as i32
            };
            let opp_dist_after: i32 = if root_side == 0 {
                self.d1[self.dist1_idx][self.g.pawn[1]] as i32
            } else {
                self.d0[self.dist0_idx][self.g.pawn[0]] as i32
            };
            let search_score = match self.ab(child_depth, -INF, INF, 1, true, m, 0) {
                Ok(s) => -s,
                Err(e) => {
                    self.unwind_move(nd0, nd1, nst, ndm_lo, ndm_hi, ndm_cache);
                    return Err(e);
                }
            };
            self.unwind_move(nd0, nd1, nst, ndm_lo, ndm_hi, ndm_cache);

            let move_nodes = self.nodes.saturating_sub(nodes_before);
            self.root_defense_diag.push(RootDefenseDiag {
                mv: m,
                full_depth_searched: true,
                child_depth_used: child_depth,
                result_class: score_result_class(search_score),
                dtm: proven_score_dtm(search_score),
                search_score,
                static_eval,
                nodes: move_nodes,
                selection_key: defense_selection_key(
                    search_score,
                    static_eval,
                    own_dist_after > own_dist_before,
                    own_dist_after,
                    opp_dist_after,
                ),
                own_dist_before,
                own_dist_after,
                opp_dist_after,
            });

            if better_defense_candidate(
                search_score,
                static_eval,
                own_dist_before,
                own_dist_after,
                opp_dist_after,
                i,
                best_score,
                best_static,
                best_own_dist_after,
                best_opp_dist_after,
                best_order,
            ) {
                best_move = m;
                best_score = search_score;
                best_static = static_eval;
                best_own_dist_after = own_dist_after;
                best_opp_dist_after = opp_dist_after;
                best_order = i;
            }
        }

        self.root_best = best_move;
        self.root_score = best_score;
        if is_pawn_move(best_move) {
            self.root_pawn_best = best_move;
            self.root_pawn_score = best_score;
        }
        Ok(best_score)
    }

    /// Won-position root selection: full-depth search of every legal root move
    /// with clean-winner move selection (see `better_clean_win_candidate`) --
    /// among moves tied at the exact same proven-win score, prefer pawn
    /// progress over a wall placement the win didn't actually need, and among
    /// tied wall placements prefer whichever pushes the opponent back further.
    /// Mirrors `root_defense_verify`; same full-width re-search cost, same
    /// deadline handling (an aborted pass just keeps whatever `root_best` the
    /// normal alpha-beta pass already found).
    fn root_clean_win_verify(&mut self, depth: i32) -> Result<i32, TimeUp> {
        if !self.pure_mode && !self.is_pondering {
            self.tt_gen = self.tt_gen.wrapping_add(1);
        }
        let root_side = self.g.turn;
        let mut moves = [0i16; 160];
        let tt_hint = if self.root_best >= 0 {
            self.root_best
        } else {
            0
        };
        let n = self.gen_moves(0, depth, tt_hint, &mut moves);
        if n == 0 {
            return Ok(self.root_score);
        }
        self.order_moves(0, &mut moves[..n], tt_hint, 0);
        let n = crate::titanium::opening_book::filter_denied_opening_legal_moves(
            &self.g, &mut moves, n,
        );
        if n == 0 {
            return Ok(self.root_score);
        }

        // CAT impact heat per candidate, computed once against the root
        // position (same model that drives move ordering / LMR elsewhere) --
        // used only as a tie-break signal below, not for pruning or ordering.
        let mut heat_by_id = [0i32; 264];
        if let Some(bridge) = self.bridge.as_ref() {
            let cat = crate::cat::build::build_impact_heatmap(&bridge.board);
            for &mv_id in &moves[..n] {
                let mv = move_id_to_board(mv_id);
                heat_by_id[mv_id as usize] = move_impact_heat(mv, &cat);
            }
        }

        let child_depth = depth - 1;
        let mut best_move = moves[0];
        let mut best_score = i32::MIN;
        let mut best_static = i32::MIN;
        let mut best_opp_dist_after = i32::MIN;
        let mut best_cat_heat = i32::MIN;
        let mut best_order = 0usize;

        for i in 0..n {
            if Instant::now() >= self.deadline {
                return Err(TimeUp);
            }
            let m = moves[i];

            self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_ROOT_WIN_BEFORE);
            let nd0 = self.dist0_idx;
            let nd1 = self.dist1_idx;
            let nst = self.cached_stamp;
            let ndm_lo = self.dir_masks_key_lo;
            let ndm_hi = self.dir_masks_key_hi;
            let ndm_cache = self.dir_masks_cache;

            crate::bench_instr::record(
                |b| &mut b.make_move,
                || {
                    self.g.make_move(m);
                    if let Some(bridge) = self.bridge.as_mut() {
                        bridge.push(m);
                    }
                },
            );
            self.refresh_dist_site(1, crate::bench_instr::REFRESH_SITE_ROOT_WIN_AFTER);
            let static_eval = {
                let ev = self.evaluate(0);
                if self.g.turn == root_side {
                    ev
                } else {
                    -ev
                }
            };
            let opp_dist_after: i32 = if root_side == 0 {
                self.d1[self.dist1_idx][self.g.pawn[1]] as i32
            } else {
                self.d0[self.dist0_idx][self.g.pawn[0]] as i32
            };
            let search_score = match self.ab(child_depth, -INF, INF, 1, true, m, 0) {
                Ok(s) => -s,
                Err(e) => {
                    self.unwind_move(nd0, nd1, nst, ndm_lo, ndm_hi, ndm_cache);
                    return Err(e);
                }
            };
            self.unwind_move(nd0, nd1, nst, ndm_lo, ndm_hi, ndm_cache);

            let is_pawn_move = is_pawn_move(m);
            let cat_heat = heat_by_id[m as usize];
            if better_clean_win_candidate(
                search_score,
                static_eval,
                is_pawn_move,
                opp_dist_after,
                cat_heat,
                i,
                best_score,
                best_static,
                crate::titanium::is_pawn_move(best_move),
                best_opp_dist_after,
                best_cat_heat,
                best_order,
            ) {
                best_move = m;
                best_score = search_score;
                best_static = static_eval;
                best_opp_dist_after = opp_dist_after;
                best_cat_heat = cat_heat;
                best_order = i;
            }
        }

        self.root_best = best_move;
        self.root_score = best_score;
        if is_pawn_move(best_move) {
            self.root_pawn_best = best_move;
            self.root_pawn_score = best_score;
        }
        Ok(best_score)
    }

    /// Entry: pathfix/RaceProof(a) — exact race endgame at ROOT. Cheap-cert
    /// engines resolve the no-wall race with the path-aware classifier plus
    /// tiny forward minimax only for volatile child states; faithful modes keep
    /// the old full race table.
    pub fn think(
        &mut self,
        time_ms: u64,
        max_depth: i32,
        full: bool,
        log: bool,
        engine_label: &str,
    ) -> ThinkResult {
        let mut stop_reason: &'static str = "unknown";
        if let Some(direct_mv) = self.prepare_opening_book_at_root() {
            let t0 = Instant::now();
            self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_OPENING);
            return ThinkResult {
                mv: direct_mv,
                score: 0,
                root_moves: Vec::new(),
                depth: 0,
                nodes: 0,
                main_thread_nodes: 0,
                helper_nodes: Vec::new(),
                total_nodes: 0,
                main_completed_depth: 0,
                helper_completed_depths: Vec::new(),
                root_widths: Vec::new(),
                root_visits: Vec::new(),
                root_move_ids: Vec::new(),
                ms: t0.elapsed().as_millis() as u64,
                white_dist: self.d0[self.dist0_idx][self.g.pawn[0]],
                black_dist: self.d1[self.dist1_idx][self.g.pawn[1]],
                depth_log: Vec::new(),
                stop_reason: "opening-book",
                race_outcome_stats: self.race_outcome_stats,
                opening_book: self.pending_opening_book_diag.take(),
                root_defense_diag: Vec::new(),
                race: RaceResultInfo::default(),
                timing: TimingDiag::default(),
            };
        }
        if self.cheap_cert {
            let rt0 = Instant::now();
            if let Some(solution) = self.semi_terminal_race_root() {
                self.rp_root_solves += 1;
                self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_CHEAP_CERT);
                if log {
                    emit_ace_progress(
                        engine_label,
                        &[],
                        99,
                        solution.legal_moves as u64,
                        solution.score,
                        &[],
                        self.d0[self.dist0_idx][self.g.pawn[0]],
                        self.d1[self.dist1_idx][self.g.pawn[1]],
                        rt0.elapsed().as_millis() as u64,
                        self.root_scores,
                        self.multipv,
                        solution.info,
                        #[cfg(feature = "wasm")]
                        Some(&mut self.wasm_progress),
                        #[cfg(feature = "wasm")]
                        self.wasm_progress_cb.as_ref(),
                    );
                }
                return ThinkResult {
                    mv: solution.mv,
                    score: solution.score,
                    root_moves: Vec::new(),
                    depth: 99,
                    nodes: solution.legal_moves as u64,
                    main_thread_nodes: solution.legal_moves as u64,
                    helper_nodes: Vec::new(),
                    total_nodes: solution.legal_moves as u64,
                    main_completed_depth: 99,
                    helper_completed_depths: Vec::new(),
                    root_widths: Vec::new(),
                    root_visits: Vec::new(),
                    root_move_ids: Vec::new(),
                    ms: rt0.elapsed().as_millis() as u64,
                    white_dist: self.d0[self.dist0_idx][self.g.pawn[0]],
                    black_dist: self.d1[self.dist1_idx][self.g.pawn[1]],
                    depth_log: Vec::new(),
                    stop_reason: if solution.exact {
                        "semi_terminal_race_exact"
                    } else {
                        "semi_terminal_race_bound"
                    },
                    race_outcome_stats: self.race_outcome_stats,
                    opening_book: None,
                    root_defense_diag: Vec::new(),
                    race: solution.info,
                    timing: TimingDiag {
                        allocated_hard_ms: time_ms,
                        elapsed_ms: rt0.elapsed().as_millis() as u64,
                        ..TimingDiag::default()
                    },
                };
            }
        }

        if self.race_proof
            && self.g.wl[0] == 0
            && self.g.wl[1] == 0
            && self.g.pawn[0] >= 9
            && self.g.pawn[1] < 72
        {
            let rt0 = Instant::now();
            // root-level: always allowed to build (force=true; deadline not set yet)
            let rv = self.race_tbl(true).map_or(0, |s| self.race_value(s)) as i32;
            if rv != 0 {
                let slot = self.rc_last as usize;
                let nm = self.g.gen_pawn_moves(&mut [0i16; 16], 0);
                if let Some((best_m, _best_v, _)) = self.race_root_pick(slot, rv) {
                    self.rp_root_solves += 1;
                    let rk = rv.abs();
                    self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_RACE_ROOT);
                    return ThinkResult {
                        mv: best_m,
                        score: if rv > 0 {
                            RACE_MATE - rk
                        } else {
                            -(RACE_MATE - rk)
                        },
                        root_moves: Vec::new(),
                        depth: 99,
                        nodes: nm as u64,
                        main_thread_nodes: nm as u64,
                        helper_nodes: Vec::new(),
                        total_nodes: nm as u64,
                        main_completed_depth: 99,
                        helper_completed_depths: Vec::new(),
                        root_widths: Vec::new(),
                        root_visits: Vec::new(),
                        root_move_ids: Vec::new(),
                        ms: rt0.elapsed().as_millis() as u64,
                        white_dist: self.d0[self.dist0_idx][self.g.pawn[0]],
                        black_dist: self.d1[self.dist1_idx][self.g.pawn[1]],
                        depth_log: Vec::new(),
                        stop_reason: "race_proof_root_table",
                        race_outcome_stats: self.race_outcome_stats,
                        opening_book: None,
                        root_defense_diag: Vec::new(),
                        race: RaceResultInfo::exact(rv.signum() as i8, rk as u16),
                        timing: TimingDiag {
                            allocated_hard_ms: time_ms,
                            elapsed_ms: rt0.elapsed().as_millis() as u64,
                            ..TimingDiag::default()
                        },
                    };
                }
            }
        }
        self.think_search(
            time_ms,
            max_depth,
            full,
            log,
            engine_label,
            &mut stop_reason,
        )
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    pub fn think_with_threads(
        &mut self,
        time_ms: u64,
        max_depth: i32,
        full: bool,
        log: bool,
        engine_label: &str,
        threads: usize,
    ) -> ThinkResult {
        if threads <= 1 {
            self.shared_tt = None;
            self.lazy_runtime = None;
            self.lazy_root_moves = None;
            self.lazy_root_visit_map = None;
            return self.think(time_ms, max_depth, full, log, engine_label);
        }
        let threads = threads.min(LAZY_SMP_MAX_THREADS);
        self.think_lazy_smp(time_ms, max_depth, full, log, engine_label, threads)
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn lazy_smp_helper_partial<'a>(
        main_result: &ThinkResult,
        helper_results: &'a [(usize, ThinkResult, Vec<usize>)],
        root_moves_raw: &[i16],
    ) -> Option<&'a ThinkResult> {
        if main_result.depth > 0 && main_result.mv != crate::titanium::TITANIUM_NO_MOVE {
            return None;
        }
        helper_results
            .iter()
            .map(|(_, result, _)| result)
            .filter(|result| {
                result.depth > 0
                    && result.mv != crate::titanium::TITANIUM_NO_MOVE
                    && root_moves_raw.contains(&result.mv)
            })
            .max_by_key(|result| (result.depth, result.nodes))
    }

    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
    fn think_lazy_smp(
        &mut self,
        time_ms: u64,
        max_depth: i32,
        full: bool,
        log: bool,
        engine_label: &str,
        threads: usize,
    ) -> ThinkResult {
        if self.g.winner() >= 0 {
            return self.think(time_ms, max_depth, full, log, engine_label);
        }

        if let Some(direct_mv) = self.prepare_opening_book_at_root() {
            let t0 = Instant::now();
            self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_OPENING);
            return ThinkResult {
                mv: direct_mv,
                score: 0,
                root_moves: Vec::new(),
                depth: 0,
                nodes: 0,
                main_thread_nodes: 0,
                helper_nodes: Vec::new(),
                total_nodes: 0,
                main_completed_depth: 0,
                helper_completed_depths: Vec::new(),
                root_widths: Vec::new(),
                root_visits: Vec::new(),
                root_move_ids: Vec::new(),
                ms: t0.elapsed().as_millis() as u64,
                white_dist: self.d0[self.dist0_idx][self.g.pawn[0]],
                black_dist: self.d1[self.dist1_idx][self.g.pawn[1]],
                depth_log: Vec::new(),
                stop_reason: "opening-book",
                race_outcome_stats: self.race_outcome_stats,
                opening_book: self.pending_opening_book_diag.take(),
                root_defense_diag: Vec::new(),
                race: RaceResultInfo::default(),
                timing: TimingDiag::default(),
            };
        }

        // Parallel search uses a fixed shared allocation. Live adaptive growth is
        // intentionally disabled because resizing a TT while other workers probe
        // it would invalidate the shared slots.
        if self.shared_tt.is_none() && self.tt_bits < TT_BITS {
            self.resize_tt(TT_BITS);
        }
        self.tt_adaptive = false;
        self.apply_think_start_state();

        let depth_limit = if max_depth > 0 {
            max_depth.min(TT_DEPTH_MAX)
        } else {
            128
        };
        let root_moves_raw = self.ordered_root_moves_snapshot(depth_limit);
        if root_moves_raw.is_empty() {
            return self.think(time_ms, max_depth, full, log, engine_label);
        }

        // EXPERIMENT: CAT impact-heat per root move, computed once against
        // the root position, used below to VALUE-filter (not count-filter)
        // each worker's root move list.
        let mut heat_by_id = [0i32; HIST_SPAN];
        let mut max_heat = 0i32;
        if let Some(bridge) = self.bridge.as_ref() {
            let cat = crate::cat::build::build_impact_heatmap(&bridge.board);
            for &mv_id in &root_moves_raw {
                let mv = move_id_to_board(mv_id);
                let h = move_impact_heat(mv, &cat);
                heat_by_id[mv_id as usize] = h;
                max_heat = max_heat.max(h);
            }
        }
        let filtered_by_worker: Vec<Vec<i16>> = (0..threads)
            .map(|worker_id| {
                let pct = lazy_smp_value_threshold_pct(worker_id);
                lazy_smp_value_filtered_moves(&root_moves_raw, &heat_by_id, max_heat, pct)
            })
            .collect();

        let root_position = self.g.clone();
        let shared_tt = self
            .shared_tt
            .clone()
            .unwrap_or_else(|| Arc::new(SharedTitaniumTt::from_search(self)));
        let deadline = Instant::now() + Duration::from_millis(time_ms.max(1));
        let runtime = Arc::new(LazySmpRuntime::new(deadline));
        let plans: Vec<WorkerPlan> = (0..threads)
            .map(|worker_id| WorkerPlan {
                worker_id,
                root_move_count: filtered_by_worker[worker_id].len(),
                root_moves_before_filter: root_moves_raw.len(),
                root_value_threshold_pct: lazy_smp_value_threshold_pct(worker_id) as usize,
                top_n_override: (self.lazy_topn && threads >= 3 && worker_id == threads - 1)
                    .then_some(LAZY_SMP_LAST_WORKER_TOP_N),
            })
            .collect();

        #[cfg(not(target_arch = "wasm32"))]
        let mut helper_results: Vec<(usize, ThinkResult, Vec<usize>)> = Vec::new();
        let main_allowed = filtered_by_worker[0].len();
        let (main_root_moves, main_visit_map) =
            Self::lazy_smp_profile_root_moves(&filtered_by_worker[0], 0, main_allowed, true);
        self.install_lazy_smp_context(
            0,
            shared_tt.clone(),
            runtime.clone(),
            Arc::new(main_root_moves),
            Arc::new(main_visit_map),
            main_allowed,
        );

        let helper_workers: Vec<(WorkerPlan, Box<TitaniumSearch>)> = plans
            .iter()
            .copied()
            .skip(1)
            .map(|plan| {
                let mut worker = self.fork_lazy_worker(&root_position);
                let allowed = plan.allowed_root_moves();
                let (profiled_root_moves, visit_map) = Self::lazy_smp_profile_root_moves(
                    &filtered_by_worker[plan.worker_id],
                    plan.worker_id,
                    allowed,
                    plan.top_n_override.is_some(),
                );
                worker.install_lazy_smp_context(
                    plan.worker_id,
                    shared_tt.clone(),
                    runtime.clone(),
                    Arc::new(profiled_root_moves),
                    Arc::new(visit_map),
                    allowed,
                );
                (plan, worker)
            })
            .collect();

        #[cfg(not(target_arch = "wasm32"))]
        let mut main_result = std::thread::scope(|scope| {
            let mut handles = Vec::with_capacity(threads.saturating_sub(1));
            for (plan, mut worker) in helper_workers {
                handles.push(scope.spawn(move || {
                    let mut stop_reason = "unknown";
                    let result = worker.think_search(
                        time_ms,
                        max_depth,
                        full,
                        false,
                        engine_label,
                        &mut stop_reason,
                    );
                    (plan.worker_id, result, worker.lazy_root_visits)
                }));
            }

            let mut stop_reason = "unknown";
            let main_result = self.think_search(
                time_ms,
                max_depth,
                full,
                log,
                engine_label,
                &mut stop_reason,
            );
            runtime.stop.store(true, Ordering::Relaxed);

            for handle in handles {
                if let Ok(result) = handle.join() {
                    helper_results.push(result);
                }
            }
            main_result
        });

        #[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
        let helper_results_shared =
            Arc::new(Mutex::new(Vec::with_capacity(threads.saturating_sub(1))));
        // Lazy SMP on wasm: dispatch helper searches with fire-and-forget
        // `rayon::spawn` onto the wasm-bindgen-rayon pool, run the main search on
        // this (seat) worker thread, then wait on an atomic completion latch.
        //
        // We deliberately AVOID `rayon::scope`/`join`: their join blocks the
        // external (non-pool) seat-worker thread on a Condvar that does not wake
        // under wasm-bindgen-rayon, which deadlocked every threaded search
        // The shared `stop` flag makes helpers return promptly once main
        // finishes, so the latch spin-wait below is brief.
        #[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
        let mut main_result = {
            let pending = Arc::new(AtomicUsize::new(helper_workers.len()));
            let engine_label_owned = engine_label.to_string();
            for (plan, mut worker) in helper_workers {
                let helper_results_shared = helper_results_shared.clone();
                let pending = pending.clone();
                let engine_label = engine_label_owned.clone();
                rayon::spawn(move || {
                    crate::wasm::note_helper_start();
                    let mut stop_reason = "unknown";
                    let result = worker.think_search(
                        time_ms,
                        max_depth,
                        full,
                        false,
                        &engine_label,
                        &mut stop_reason,
                    );
                    helper_results_shared.lock().expect("helper results").push((
                        plan.worker_id,
                        result,
                        worker.lazy_root_visits,
                    ));
                    pending.fetch_sub(1, Ordering::Release);
                });
            }

            let mut stop_reason = "unknown";
            let main_result = self.think_search(
                time_ms,
                max_depth,
                full,
                log,
                engine_label,
                &mut stop_reason,
            );
            runtime.stop.store(true, Ordering::Relaxed);
            // Helpers observe `stop` in check_time() and return within ~64 nodes.
            // The seat worker is a Web Worker (not the UI thread), so a brief spin
            // here is safe. Do not start a second movetime-sized wait after the
            // main search: that used to overrun the browser clock by seconds.
            let latch_deadline = runtime.deadline + Duration::from_millis(50);
            while pending.load(Ordering::Acquire) > 0 && Instant::now() < latch_deadline {
                std::hint::spin_loop();
            }
            main_result
        };

        // Drain the collected helper results by locking — NOT `Arc::try_unwrap`,
        // which races: a helper decrements `pending` (its last statement) before
        // its captured `Arc` clone drops, so once the latch sees `pending == 0`
        // the clones may still be alive and `try_unwrap` would fail, silently
        // discarding every helper result. Locking is correct regardless of how
        // many `Arc` clones remain; `pending == 0` already guarantees all pushes
        // completed.
        #[cfg(all(target_arch = "wasm32", feature = "wasm-threads"))]
        let mut helper_results = {
            let mut guard = helper_results_shared.lock().expect("helper results");
            std::mem::take(&mut *guard)
        };
        helper_results.sort_by_key(|(worker_id, _, _)| *worker_id);

        let main_completed_depth = main_result.depth;
        let main_nodes = main_result.nodes;
        if let Some(helper) =
            Self::lazy_smp_helper_partial(&main_result, &helper_results, &root_moves_raw)
        {
            main_result.mv = helper.mv;
            main_result.score = helper.score;
            main_result.depth = helper.depth;
            main_result.ms = main_result.ms.max(helper.ms);
            main_result.white_dist = helper.white_dist;
            main_result.black_dist = helper.black_dist;
            main_result.depth_log = helper.depth_log.clone();
            main_result.stop_reason = "lazy_smp_helper_partial";
        }

        let helper_nodes: Vec<u64> = helper_results.iter().map(|(_, r, _)| r.nodes).collect();
        let helper_depths: Vec<i32> = helper_results.iter().map(|(_, r, _)| r.depth).collect();
        let mut root_visits = vec![self.lazy_root_visits.clone()];
        root_visits.extend(helper_results.iter().map(|(_, _, visits)| visits.clone()));
        let total_nodes = main_nodes + helper_nodes.iter().copied().sum::<u64>();
        main_result.main_thread_nodes = main_nodes;
        main_result.helper_nodes = helper_nodes;
        main_result.total_nodes = total_nodes;
        main_result.nodes = total_nodes;
        main_result.main_completed_depth = main_completed_depth;
        main_result.helper_completed_depths = helper_depths;
        main_result.root_widths = plans;
        main_result.root_visits = root_visits;
        main_result.root_move_ids = filtered_by_worker;
        main_result
    }

    /// Iterative deepening within `time_ms`. `full` disables the easy-move stop.
    fn think_search(
        &mut self,
        time_ms: u64,
        max_depth: i32,
        full: bool,
        log: bool,
        engine_label: &str,
        stop_reason: &mut &'static str,
    ) -> ThinkResult {
        let t0 = Instant::now();
        crate::bench_instr::begin_search();
        self.ace_rfp_max_depth = rfp_depth_for_budget(self.rfp_tc_adaptive, time_ms);
        let rc_hits_at_start = self.rc_hits;
        let rc_solves_at_start = self.rc_solves;
        // pathfix/RaceProof(b): reserve the commitment gate's worst-case cost
        // out of the search deadline when the gate can fire — it runs after
        // the search loop and its raceTbl(force=true) call ignores deadline.
        let mut gate_reserve_ms = 0u64;
        if self.race_proof && !self.cheap_cert && self.g.wl[self.g.turn] == 1 {
            let cap = (0.3 * time_ms as f64) as u64;
            gate_reserve_ms = self
                .rc_build_ms
                .max(25)
                .max((time_ms as f64 * 0.15) as u64)
                .min(cap);
        }
        // Each thread derives its deadline from its OWN monotonic clock. Under
        // wasm, `web_time::Instant` is backed by per-Worker `performance.now()`
        // origins, so a deadline created on the main thread is meaningless to a
        // rayon helper thread — it would never time out and the scope join would
        // hang. Cross-thread early-exit is handled by `LazySmpRuntime::stop`
        // (checked in `check_time`), NOT by a shared Instant. (Native clocks are
        // cross-thread comparable, so per-thread t0 is equivalent there ±µs.)
        self.deadline = t0 + Duration::from_millis(time_ms.saturating_sub(gate_reserve_ms));
        self.nodes = 0;
        self.race_outcome_stats = RaceOutcomeStats::default();
        self.root_best = crate::titanium::TITANIUM_NO_MOVE;
        self.root_score = 0;
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let skip_setup = self.lazy_skip_setup;
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        {
            self.lazy_skip_setup = false;
        }
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        let skip_setup = false;
        if !skip_setup {
            // Advance TT generation and decay history once at think start.
            // Lazy SMP does this before forking workers so every worker stores
            // into the same generation and starts from the same ordered root.
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            self.apply_think_start_state();
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            if !self.pure_mode && !self.is_pondering {
                self.tt_gen = self.tt_gen.wrapping_add(1);
                for h in self.history_tbl.iter_mut() {
                    *h >>= 1;
                }
                for side in self.hist_sf.iter_mut() {
                    for h in side.iter_mut() {
                        *h /= 2;
                    }
                }
                for h in self.cont_hist.iter_mut() {
                    *h /= 2;
                }
            }
        }
        // RaceProof per-think solve budgets + caps
        self.rc_think_solve_ms = 0;
        self.rc_solve_cap = time_ms as f64 * 0.25;
        self.rc_blocked = false;
        self.rc_think_solves = 0;
        self.rp_root_empty = self.race_proof && self.g.wl[0] == 0 && self.g.wl[1] == 0;
        self.rp_build_ok = false;
        self.stream_log = log;
        self.stream_label = engine_label.to_string();
        self.stream_t0 = t0;
        self.stream_root_score = 0;
        self.stream_search_depth = 0;
        self.stream_depth_log.clear();
        self.stream_root_moves.clear();
        self.stream_last_emit_nodes = 0;
        self.stream_last_emit_ms = 0;
        self.stream_last_best = crate::titanium::TITANIUM_NO_MOVE;
        // Re-sync the mirrored Titanium board from the authoritative ACE game.
        // Kills any drift left over from a previous search (e.g. an unbalanced
        // push/pop on time-abort) before it can poison this move's root list.
        if self.bridge.is_some() {
            self.bridge = Some(TiBridge::from_game(&self.g));
        }
        let mut last_best: i16 = crate::titanium::TITANIUM_NO_MOVE;
        let mut last_score = 0;
        let mut last_depth = 0;
        let mut stable = 0;
        let mut best_move_changes: u32 = 0;
        let mut score_delta: i32 = 0;
        let mut soft_ms =
            Self::stability_soft_ms(time_ms, last_score, best_move_changes, score_delta, stable);
        let mut partial_iter_used = false;
        // RaceProof(b); -1 sentinel — pawn-move id 0 (a1) is legal
        let mut last_pawn_best: i16 = -1;
        let mut last_pawn_score: i32 = i32::MIN;
        let mut depth_log: Vec<AceDepthLogEntry> = Vec::new();
        let max_depth = if max_depth > 0 {
            max_depth.min(TT_DEPTH_MAX)
        } else {
            128
        };
        let root_q_left = if self.q_search { self.q_max } else { 0 };

        // Dynamic iterative-deepening startup: probe the TT for the root position.
        // If the prior think (or pondering) left a deep exact entry, skip the
        // shallow iterations we already know the answer to and resume from near
        // that depth. last_score is seeded from the TT so aspiration windows are
        // correctly centred on the first iteration we actually run.
        // Disabled in pure_mode (faithful JS baseline).
        let start_depth = if !self.pure_mode {
            let ridx = (self.g.hash_lo & self.tt_mask) as usize;
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            let root_entry = self
                .shared_tt
                .as_ref()
                .and_then(|tt| tt.probe(self.g.hash_lo, self.g.hash_hi));
            // Lazy SMP shared TT is authoritative; no local fallback (see ab()).
            #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
            let rmeta = match root_entry {
                Some(entry) => entry.meta,
                None if self.shared_tt.is_some() => 0,
                None => self.tt_meta[ridx],
            };
            #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
            let rmeta = self.tt_meta[ridx];
            if rmeta != 0 && {
                #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                {
                    root_entry.is_some()
                        || (self.tt_key_hi[ridx] == self.g.hash_hi
                            && self.tt_key_lo[ridx] == self.g.hash_lo)
                }
                #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                {
                    self.tt_key_hi[ridx] == self.g.hash_hi && self.tt_key_lo[ridx] == self.g.hash_lo
                }
            } {
                let tt_depth = tt_unpack_depth(rmeta);
                let tt_flag = (rmeta >> 10) & 3;
                if tt_depth >= 4 && tt_flag == 0 {
                    // Exact score: safe to use as aspiration seed and skip iterations.
                    #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
                    {
                        last_score = match root_entry {
                            Some(entry) => entry.score,
                            None => self.tt_score[ridx],
                        };
                    }
                    #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
                    {
                        last_score = self.tt_score[ridx];
                    }
                    (tt_depth - 2).max(1)
                } else {
                    1
                }
            } else {
                1
            }
        } else {
            1
        };

        for d in start_depth..=max_depth {
            soft_ms = Self::stability_soft_ms(
                time_ms,
                last_score,
                best_move_changes,
                score_delta,
                stable,
            );
            if !full && d > 1 && Self::soft_over_time_budget(t0, soft_ms) {
                *stop_reason = "stability_soft_budget_before_depth";
                break;
            }
            if self.use_predict_stop
                && !full
                && Self::predicted_over_time_budget(t0, soft_ms, &depth_log)
            {
                *stop_reason = "predicted_over_time_budget_before_depth";
                break;
            }
            if Instant::now() >= self.deadline {
                *stop_reason = "deadline_before_depth";
                break;
            }
            // RaceProof: in-tree solves only when cheap to amortize
            self.rp_build_ok = self.rp_root_empty || d >= 6;
            self.root_pawn_best = -1;
            self.root_pawn_score = i32::MIN;
            self.stream_root_score = last_score;
            self.stream_search_depth = d;
            self.stream_root_moves.clear();
            let nodes_at_depth = self.nodes;
            let result = if d >= 4 && last_score > -2000 && last_score < 2000 {
                // aspiration: graded widening (ka_ab.js-style) — a failed bound
                // widens 4x re-centred on the fail score before falling back to
                // fully open, so a single fail-high/low doesn't disable the
                // window's pruning power across the whole re-search; only a
                // second fail on the SAME side opens it.
                const ASP_WINDOW: i32 = 75;
                let mut lo = last_score - ASP_WINDOW;
                let mut hi = last_score + ASP_WINDOW;
                let mut low_fails = 0u32;
                let mut high_fails = 0u32;
                loop {
                    match self.ab(d, lo, hi, 0, true, 0, root_q_left) {
                        Ok(sc) => {
                            if sc <= lo && lo > -INF {
                                low_fails += 1;
                                lo = if low_fails >= 2 {
                                    -INF
                                } else {
                                    sc - 4 * ASP_WINDOW
                                };
                            } else if sc >= hi && hi < INF {
                                high_fails += 1;
                                hi = if high_fails >= 2 {
                                    INF
                                } else {
                                    sc + 4 * ASP_WINDOW
                                };
                            } else {
                                break Ok(sc);
                            }
                        }
                        Err(e) => break Err(e),
                    }
                }
            } else {
                self.ab(d, -INF, INF, 0, true, 0, root_q_left)
            };
            match result {
                Ok(sc) => {
                    if self.root_best != last_best
                        && last_best != crate::titanium::TITANIUM_NO_MOVE
                        && self.root_best >= 0
                    {
                        best_move_changes = best_move_changes.saturating_add(1);
                    }
                    stable = if self.root_best == last_best {
                        stable + 1
                    } else {
                        0
                    };
                    // Score swing vs previous completed ID step (0 on first).
                    if last_depth > 0 {
                        score_delta = sc - last_score;
                    } else {
                        score_delta = 0;
                    }
                    last_best = self.root_best;
                    last_score = sc;
                    last_depth = d;
                    soft_ms = Self::stability_soft_ms(
                        time_ms,
                        last_score,
                        best_move_changes,
                        score_delta,
                        stable,
                    );
                    if self.root_pawn_best >= 0 {
                        // RaceProof(b)
                        last_pawn_best = self.root_pawn_best;
                        last_pawn_score = self.root_pawn_score;
                    }
                    let elapsed_ms = t0.elapsed().as_millis() as u64;
                    let pv = if last_best >= 0 {
                        crate::titanium::move_id_to_algebraic(last_best)
                    } else {
                        String::new()
                    };
                    depth_log.push(AceDepthLogEntry {
                        depth: d,
                        score: last_score,
                        nodes: self.nodes,
                        elapsed_ms,
                        marginal_nodes: self.nodes.saturating_sub(nodes_at_depth),
                        pv,
                    });
                    if log {
                        self.sync_stream_meta(&depth_log, d, last_score);
                        self.emit_stream_progress(true);
                    }
                    if is_proven_loss_score(sc) {
                        match self.root_defense_verify(d) {
                            Ok(defense_score) => {
                                last_best = self.root_best;
                                last_score = defense_score;
                                if self.root_pawn_best >= 0 {
                                    last_pawn_best = self.root_pawn_best;
                                    last_pawn_score = self.root_pawn_score;
                                }
                                if let Some(entry) = depth_log.last_mut() {
                                    if entry.depth == d {
                                        entry.score = last_score;
                                        entry.pv = if last_best >= 0 {
                                            crate::titanium::move_id_to_algebraic(last_best)
                                        } else {
                                            String::new()
                                        };
                                    }
                                }
                                if log {
                                    self.sync_stream_meta(&depth_log, d, last_score);
                                    self.emit_stream_progress(true);
                                }
                            }
                            Err(TimeUp) => {
                                if self.use_partial_iter && self.root_best >= 0 {
                                    last_best = self.root_best;
                                    last_score = self.root_score;
                                    partial_iter_used = true;
                                }
                                *stop_reason = "time_up";
                                break;
                            }
                        }
                    } else if is_proven_win_score(sc) {
                        match self.root_clean_win_verify(d) {
                            Ok(win_score) => {
                                last_best = self.root_best;
                                last_score = win_score;
                                if self.root_pawn_best >= 0 {
                                    last_pawn_best = self.root_pawn_best;
                                    last_pawn_score = self.root_pawn_score;
                                }
                                if let Some(entry) = depth_log.last_mut() {
                                    if entry.depth == d {
                                        entry.score = last_score;
                                        entry.pv = if last_best >= 0 {
                                            crate::titanium::move_id_to_algebraic(last_best)
                                        } else {
                                            String::new()
                                        };
                                    }
                                }
                                if log {
                                    self.sync_stream_meta(&depth_log, d, last_score);
                                    self.emit_stream_progress(true);
                                }
                            }
                            Err(TimeUp) => {
                                if self.use_partial_iter && self.root_best >= 0 {
                                    last_best = self.root_best;
                                    last_score = self.root_score;
                                    partial_iter_used = true;
                                }
                                *stop_reason = "time_up";
                                break;
                            }
                        }
                    }
                    if last_score > MATE - 200 || last_score < -(MATE - 200) {
                        *stop_reason = "forced_mate_or_loss";
                        break; // forced result
                    }
                    // v8 easy-move stop (acev8_engine.js)
                    if !full
                        && d >= 9
                        && stable >= 3
                        && last_score > -120
                        && t0.elapsed().as_millis() as u64 > time_ms * 3 / 10
                    {
                        *stop_reason = "easy_move_stable";
                        break;
                    }
                }
                Err(TimeUp) => {
                    // Lague partial-iteration: the aborted depth-`d` iteration
                    // still searched its best-ordered root moves to full depth.
                    // `root_best` only updates after a root move's search FULLY
                    // completes (the `?` on an aborted child returns first), so it
                    // holds the best completed move — adopt it instead of falling
                    // back to depth d-1. On a pure fail-low (no alpha-raise this
                    // iteration) root_best/root_score still equal the prior depth's
                    // values, so this is a no-op exactly in the unsafe case.
                    if self.use_partial_iter && self.root_best >= 0 {
                        last_best = self.root_best;
                        last_score = self.root_score;
                        last_depth = d;
                        partial_iter_used = true;
                        if self.root_pawn_best >= 0 {
                            last_pawn_best = self.root_pawn_best;
                            last_pawn_score = self.root_pawn_score;
                        }
                    }
                    *stop_reason = "time_up";
                    break; // state already restored by unwinding unmakes
                }
            }
            if !full && Self::soft_over_time_budget(t0, soft_ms) {
                *stop_reason = "stability_soft_budget_after_depth";
                break;
            }
        }
        if *stop_reason == "unknown" {
            *stop_reason = "max_depth_completed";
        }

        // ---------- pathfix/RaceProof(b): last-wall commitment gate (DEMOTE, never forbid) ----------
        // About to commit our FINAL wall: demote it below the best non-wall
        // root alternative unless the post-wall position is PROVEN won/
        // not-lost for us. When the wall empties both hands, the k=0 race
        // oracle decides (verdict <= 0 for the opponent = we are not lost).
        // gen13: otherwise use the inlined certifier with REFUTATION semantics
        // — demote ONLY on positive evidence the wall LOSES (a certificate that
        // the OPPONENT, stm after our wall, wins). The v11 browser port kept
        // the wall here unconditionally (RP_CERT was null); gen13's certify_win
        // inlining makes this branch live. Proven-mate walls and positions
        // without a pawn alternative are kept. Worst-case gate cost was
        // reserved out of the search deadline up front (gate_reserve_ms).
        if self.race_proof
            && is_wall_move(last_best)
            && self.g.wl[self.g.turn] == 1
            && last_pawn_best >= 0
            && last_score < MATE - 200
            && last_pawn_score > -(MATE - 200)
        {
            self.g.make_move(last_best);
            let rp_ok = if self.g.wl[0] == 0 && self.g.wl[1] == 0 {
                use crate::titanium::cert_bridge::hands_empty_race_stm_wins;
                match hands_empty_race_stm_wins(&mut self.g) {
                    Some(opp_wins) => !opp_wins,
                    None => true, // unknown ⇒ do not demote without proof
                }
            } else if self.cert_eval_leaves_only {
                // Walls remain: search + EME cover tempo; skip recursive certify here.
                true
            } else {
                // gen13 refutation: demote only if the opponent's win is certified.
                let deadline_ms = 25u64.max(time_ms * 15 / 100);
                !self.cert_win(self.g.turn, 60_000, deadline_ms)
            };
            self.g.unmake_move();
            self.cached_stamp = -1;
            if !rp_ok {
                self.rp_demotions += 1;
                last_best = last_pawn_best;
                last_score = last_pawn_score;
            }
        }

        // Bridge desync detector: whenever control is back at the root the
        // mirrored board's undo stack MUST be empty. If not, a make/unmake
        // path leaked a frame (this is how "illegal move" crashes happen) —
        // log it loudly and rebuild from the authoritative game.
        if let Some(bridge) = self.bridge.as_ref() {
            if !bridge.undo_stack.is_empty() {
                eprintln!(
                    "info string ace bridge DESYNC: {} unpopped frames after search — rebuilding",
                    bridge.undo_stack.len()
                );
                self.bridge = Some(TiBridge::from_game(&self.g));
            }
        }

        // Root legality guard: never emit a move the true position rejects.
        // Regenerates the legal root list from clean state; if the searched
        // best move is not in it, substitute the best legal alternative.
        self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_THINK_GUARD);
        let mut legal = [0i16; 160];
        let nlegal = self.gen_moves(0, 1, last_best, &mut legal);
        let root_ok = nlegal > 0 && last_best >= 0 && legal[..nlegal].contains(&last_best);
        if !root_ok {
            if last_best >= 0 && nlegal > 0 {
                eprintln!(
                    "info string ace root guard: searched best {} is illegal in true position — substituting",
                    crate::titanium::move_id_to_algebraic(last_best)
                );
            }
            if nlegal > 0 {
                self.order_moves(0, &mut legal[..nlegal], 0, 0);
                last_best = legal[0];
            } else {
                last_best = crate::titanium::TITANIUM_NO_MOVE;
            }
        }

        self.refresh_dist_site(0, crate::bench_instr::REFRESH_SITE_THINK_FINAL);
        let white_dist = self.d0[self.dist0_idx][self.g.pawn[0]];
        let black_dist = self.d1[self.dist1_idx][self.g.pawn[1]];
        let ms = t0.elapsed().as_millis() as u64;

        if log {
            self.sync_stream_meta(&depth_log, last_depth, last_score);
            self.emit_stream_progress(true);
        }

        if std::env::var_os("TITANIUM_WALL_CACHE_STATS").is_some() {
            if let Some(s) = self.wall_cache_stats() {
                eprintln!(
                    "info string wall_cache hits_eval={} misses_eval={} hits_movegen={} misses_movegen={} wall_gen_calls={}",
                    s.hits_eval,
                    s.misses_eval,
                    s.hits_movegen,
                    s.misses_movegen,
                    s.wall_generation_calls
                );
            }
        }

        crate::bench_instr::set_stop_reason(stop_reason);
        crate::bench_instr::end_search(self.nodes);
        self.race_outcome_stats.race_tbl_lru_hits = self.rc_hits.saturating_sub(rc_hits_at_start);
        self.race_outcome_stats.race_tbl_lru_rebuilds =
            self.rc_solves.saturating_sub(rc_solves_at_start);

        let timing = TimingDiag::from_think(
            time_ms,
            gate_reserve_ms,
            last_score,
            ms,
            &depth_log,
            best_move_changes,
            partial_iter_used,
            soft_ms,
        );

        ThinkResult {
            mv: last_best,
            score: last_score,
            root_moves: self.stream_root_moves.clone(),
            depth: last_depth,
            nodes: self.nodes,
            main_thread_nodes: self.nodes,
            helper_nodes: Vec::new(),
            total_nodes: self.nodes,
            main_completed_depth: last_depth,
            helper_completed_depths: Vec::new(),
            root_widths: Vec::new(),
            root_visits: Vec::new(),
            root_move_ids: Vec::new(),
            ms,
            white_dist,
            black_dist,
            depth_log,
            stop_reason: *stop_reason,
            race_outcome_stats: self.race_outcome_stats,
            opening_book: self.pending_opening_book_diag.take(),
            root_defense_diag: self.root_defense_diag.clone(),
            race: RaceResultInfo::from_score(last_score),
            timing,
        }
    }
}
