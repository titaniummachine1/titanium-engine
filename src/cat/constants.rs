//! CAT v3 thresholds — search ordering, LMR, and pruning cutoffs (centi-squares).

/// Heat on a player's shortest path square (delta = 0).
/// Combined two-player ceiling: `2 × CAT_CORRIDOR_CM = 400 cm`.
pub const CAT_CORRIDOR_CM: u16 = 200;

/// Exact and near-shortest corridors are search-relevant; larger detours are zero.
/// Keep at least four suboptimal route sets visible to avoid single-path tunnel vision.
pub const MAX_RELEVANT_CORRIDOR_DELTA: u16 = 4;
pub const BOTTLENECK_CORRIDOR_DELTA: u16 = 2;
pub const BOTTLENECK_BONUS_CM: u16 = 40;

/// Skip LMR / treat as tactical when local heat ≥ this.
pub const CAT_HOT_CM: u16 = 160;

/// Cold fringe — extra LMR reduction below this.
pub const CAT_COLD_CM: u16 = 60;

/// Sentinel when BFS finds no path.
pub const DIST_PENALTY: u8 = 255;
