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

pub use attention::CorridorAttention;
pub use constants::{CAT_COLD_CM, CAT_CORRIDOR_CM, CAT_HOT_CM, DIST_PENALTY};
pub use prune::{
    collect_search_moves, move_corridor_attention, wall_net_race, wall_should_search,
};
pub use viz::cat_snapshot_json;
