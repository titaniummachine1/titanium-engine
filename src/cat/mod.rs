//! Corridor Attention Table (CAT) v3 — heatmaps, pruning, web viz.
//!
//! - `attention` — per-square / per-wall-edge heat types
//! - `build`     — construct CAT from BFS distance fields
//! - `constants` — HOT/COLD thresholds
//! - `prune`     — CAT-backed move filtering for search
//! - `viz`       — JSON snapshot for the CatV3 web tab

pub mod attention;
pub mod build;
pub mod constants;
pub mod prune;
pub mod viz;
pub mod viz_build;
pub mod slack;

pub use attention::CorridorAttention;
pub use build::{build_corridor_attention, corridor_bottleneck_count};
pub use constants::{CAT_COLD_CM, CAT_CORRIDOR_CM, CAT_HOT_CM, DIST_PENALTY};
pub use prune::{
    best_pawn_cat_heats, cat_v16_lmr_ceiling_from_env, cat_v16_lmr_fringe_pct_for_worker,
    cat_v16_lmr_reduction_plies, collect_search_moves, move_corridor_attention,
    move_corridor_attention_with_denial, move_corridor_attention_with_path, wall_net_race,
    wall_should_search, CatHeatRefs, CAT_V16_FRINGE_PCT_DEFAULT, CAT_V16_FRINGE_PCT_MAX,
    CAT_V16_FRINGE_PCT_STEP_PER_WORKER, CAT_V16_LMR_CEILINGS, CAT_V16_LMR_CEILING_DEFAULT,
};
pub use viz::cat_snapshot_json;
