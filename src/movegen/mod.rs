//! Legal move generation (no pruning — see `cat::prune`).

pub mod legal;
pub mod o1;
pub mod pawn_bits;
pub mod wall_masks;

pub use legal::{
    count_geometric_legal_walls, generate_legal_moves, generate_legal_moves_into,
    generate_legal_moves_slice, generate_legal_moves_slice_anchor_baseline,
    generate_legal_moves_slice_cached, generate_legal_moves_slice_mode,
    generate_pawn_moves_slice_mode, geometric_wall_len_cached, GeometricWallCache,
    GeometricWallCacheRole, GeometricWallCacheStats, GeometricWallKey, PawnGenMode,
    MAX_LEGAL_MOVES,
};
/// Force the cold-start pawn lookup tables to build now (so search/perft timing
/// excludes the build). No-op once built. See `o1::runtime`.
pub use o1::prewarm;
pub use pawn_bits::{
    generate_pawn_moves_bitboard_slice, generate_pawn_moves_bitboard_with_masks,
    generate_pawn_moves_shift_slice,
};
