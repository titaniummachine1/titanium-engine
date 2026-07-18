//! O1 pawn-table generation logic (single source of truth).
//!
//! Pure compute, no I/O: `discover_all_pawn_tables()` returns every pawn square's
//! metadata (catalog, per-enemy-layer wall slots, PEXT-ordered wall_bits, the
//! semantic remap, and the legal-move table) in memory. Two consumers:
//!
//! - `super::runtime` — builds the in-memory `PawnTables` at cold start (default).
//! - the `movegen-o1-gen` bin (`gen/emit.rs`) — formats it to the
//!   embedded `generated_tables_data.rs` + `generated_remap.bin` for the
//!   optional `embed-tables` build.
//!
//! Table generation lives entirely under `src/movegen/o1/gen/` (no `engine/build/`).

pub mod emit;
pub mod geometry;
pub mod pawn;
pub mod pseudo_moves;

pub use emit::generate;
pub use pawn::{
    discover_all_pawn_tables, discover_pawn_square, EnemyLayerMeta, PawnSquareMeta, ENEMY_LAYERS,
    MAX_WALL_SLOTS, PHYS_WALL_COMBOS, WALL_KEYS,
};
