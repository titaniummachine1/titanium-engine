//! O(1) two-layer pawn movegen + wall collision/topo tables.
//!
//! ```text
//! Pawns: enemy_key + wall_key → PAWN_LEGAL[sq][enemy_key][wall_key]
//! Walls: L1 empty → L2 collision → [topo O(1) flood-skip] → L3 flood (legal.rs)
//! ```

mod lookup;
mod tables;

pub use lookup::{
    encode_enemy_key, generate_pawn_moves_o1, generate_wall_candidates_o1, legal_pawn_move_mask,
    pack_wall_key, wall_collision_clear_h_mask, wall_collision_clear_v_mask, wall_l12_h_mask,
    wall_l12_v_mask, wall_needs_flood_h_mask, wall_needs_flood_v_mask, wall_physically_legal_o1,
};
