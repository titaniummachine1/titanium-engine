//! Direction bitmasks for bitwise flood fill on the pawn grid.

use crate::core::board::Board;
use crate::util::grid::{can_step, flood_bit_sq, square_index};

/// Bit `sq` set iff a pawn on `sq` may step in that direction.
#[derive(Clone, Copy, Default)]
pub struct DirMasks {
    pub north: u128,
    pub south: u128,
    pub east: u128,
    pub west: u128,
}

impl DirMasks {
    pub fn from_board(board: &Board) -> Self {
        let mut m = Self::default();
        for r in 0..=8u8 {
            for c in 0..=8u8 {
                let sq = square_index(r, c);
                let bit = flood_bit_sq(sq);
                if can_step(board, r, c, -1, 0) {
                    m.north |= bit;
                }
                if can_step(board, r, c, 1, 0) {
                    m.south |= bit;
                }
                if can_step(board, r, c, 0, 1) {
                    m.east |= bit;
                }
                if can_step(board, r, c, 0, -1) {
                    m.west |= bit;
                }
            }
        }
        m
    }
}
