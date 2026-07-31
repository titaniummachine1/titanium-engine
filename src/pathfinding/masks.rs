//! Direction bitmasks for bitwise flood fill on the pawn grid.

use crate::core::board::Board;
use crate::titanium::game::{GameState, BORDER};
use crate::util::grid::{can_step, flood_bit_sq, square_index, FLOOD_BIT_BY_SQ};

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

    /// Wall topology in Titanium internal cell order (row 0 = top) — no Board rebuild or row remap.
    pub fn from_ace_game(g: &GameState) -> Self {
        crate::bench_instr::record(
            |b| &mut b.dir_masks_from_ace,
            || Self::from_ace_blocked(&g.blocked),
        )
    }

    /// Build masks from ACE/Titanium's row-major blocked-direction table.
    /// This keeps the legacy ACE reference and production Titanium engines on
    /// exactly the same BFF topology adapter.
    pub fn from_ace_blocked(blocked_table: &[u8; 81]) -> Self {
        let mut m = Self::default();
        for sq in 0..81usize {
            let bit = FLOOD_BIT_BY_SQ[sq];
            let blocked = blocked_table[sq] | BORDER[sq];
            // ACE `can_step`: 0=N(-9), 1=S(+9), 2=W(-1), 3=E(+1)
            if blocked & 1 == 0 {
                m.north |= bit;
            }
            if blocked & 2 == 0 {
                m.south |= bit;
            }
            if blocked & 4 == 0 {
                m.west |= bit;
            }
            if blocked & 8 == 0 {
                m.east |= bit;
            }
        }
        m
    }

    /// Remove one undirected ACE-grid edge from an existing topology.
    /// Used by corridor proofs that ask whether a selected edge is a bridge.
    #[inline]
    pub fn without_ace_edge(mut self, a: usize, b: usize) -> Self {
        if a >= 81 || b >= 81 {
            debug_assert!(false, "ACE edge endpoint out of range");
            return self;
        }
        let (lo, hi) = if a <= b { (a, b) } else { (b, a) };
        match hi - lo {
            1 if lo / 9 == hi / 9 => {
                self.east &= !FLOOD_BIT_BY_SQ[lo];
                self.west &= !FLOOD_BIT_BY_SQ[hi];
            }
            9 => {
                self.south &= !FLOOD_BIT_BY_SQ[lo];
                self.north &= !FLOOD_BIT_BY_SQ[hi];
            }
            _ => debug_assert!(false, "ACE edge endpoints are not adjacent"),
        }
        self
    }

    #[inline]
    pub fn with_ace_wall(mut self, wall_type: usize, slot: usize) -> Self {
        let r = slot / 8;
        let c = slot % 8;
        let a = r * 9 + c;
        if wall_type == 0 {
            let b = a + 1;
            let cc = a + 9;
            let dd = b + 9;
            self.south &= !(FLOOD_BIT_BY_SQ[a] | FLOOD_BIT_BY_SQ[b]);
            self.north &= !(FLOOD_BIT_BY_SQ[cc] | FLOOD_BIT_BY_SQ[dd]);
        } else {
            let b = a + 9;
            let cc = a + 1;
            let dd = b + 1;
            self.east &= !(FLOOD_BIT_BY_SQ[a] | FLOOD_BIT_BY_SQ[b]);
            self.west &= !(FLOOD_BIT_BY_SQ[cc] | FLOOD_BIT_BY_SQ[dd]);
        }
        self
    }
}

#[cfg(test)]
mod incremental_tests {
    use super::*;
    use crate::core::board::{Board, WallOrientation};
    use crate::util::grid::set_wall;

    /// `with_ace_wall` is the incremental single-wall update the search relies on
    /// (`search_impl::current_dir_masks` uses it to avoid rebuilding all 81
    /// cells). It must agree exactly with a full rebuild of the walled board.
    #[test]
    fn with_ace_wall_matches_full_rebuild() {
        let mut mismatches = Vec::new();
        for wall_type in 0..2usize {
            for slot in 0..64usize {
                let row = (slot / 8) as u8;
                let col = (slot % 8) as u8;
                let orientation = if wall_type == 0 {
                    WallOrientation::Horizontal
                } else {
                    WallOrientation::Vertical
                };

                let base = Board::new();
                let incr = DirMasks::from_board(&base).with_ace_wall(wall_type, slot);

                let mut walled = Board::new();
                set_wall(&mut walled, row, col, orientation, true);
                let full = DirMasks::from_board(&walled);

                if (incr.north, incr.south, incr.east, incr.west)
                    != (full.north, full.south, full.east, full.west)
                {
                    mismatches.push((wall_type, slot, row, col));
                }
            }
        }
        assert!(
            mismatches.is_empty(),
            "with_ace_wall disagrees with full rebuild on {} of 128 slots; first few: {:?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(6)]
        );
    }
}
