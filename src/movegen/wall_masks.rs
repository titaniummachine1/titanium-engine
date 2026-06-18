//! Shift-based wall candidate masks (L1 empty ∧ L2 collision ∧ TOPO flood-skip).
//!
//! Production wall path — no runtime lookup tables. L3 flood lives in `legal.rs` / `path/parallel.rs`.

use crate::core::board::Board;
#[cfg(test)]
use crate::core::board::WallOrientation;

/// L2: passes overlap / cross / neighbor collision rules (`wall_collides` inverse).
#[inline]
pub fn wall_physically_legal_shift(board: &Board, row: u8, col: u8, horizontal: bool) -> bool {
    let masks = wall_masks(board);
    let mask = if horizontal { masks.l12_h } else { masks.l12_v };
    (mask >> ((row as u64) * 8 + col as u64)) & 1 != 0
}

const COL_0: u64 = 0x0101_0101_0101_0101;
const COL_7: u64 = COL_0 << 7;
const ROW_0: u64 = 0xFF;
const ROW_7: u64 = 0xFF << 56;

#[inline]
fn east1(x: u64) -> u64 {
    (x << 1) & !COL_0
}

#[inline]
fn east2(x: u64) -> u64 {
    (x << 2) & !(COL_0 | COL_0 << 1)
}

#[inline]
fn west1(x: u64) -> u64 {
    (x >> 1) & !COL_7
}

#[inline]
fn west2(x: u64) -> u64 {
    (x >> 2) & !(COL_7 | COL_7 >> 1)
}

#[inline]
fn two_of_three(a: u64, b: u64, m: u64) -> u64 {
    (a & b) | (m & (a | b))
}

#[inline]
fn topo_h_from(h: u64, v: u64) -> u64 {
    let side_a = COL_0 | east1(v) | east1(v >> 8) | east1(v << 8) | east2(h);
    let side_b = COL_7 | west1(v) | west1(v >> 8) | west1(v << 8) | west2(h);
    let middle = (v >> 8) | (v << 8);
    two_of_three(side_a, side_b, middle)
}

#[inline]
fn topo_v_from(h: u64, v: u64) -> u64 {
    let side_a = ROW_7 | (h >> 8) | east1(h >> 8) | west1(h >> 8) | (v >> 16);
    let side_b = ROW_0 | (h << 8) | east1(h << 8) | west1(h << 8) | (v << 16);
    let middle = east1(h) | west1(h);
    two_of_three(side_a, side_b, middle)
}

/// All wall candidate masks for one node — single read of the two wall bitboards.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WallMasks {
    pub l12_h: u64,
    pub l12_v: u64,
    pub topo_h: u64,
    pub topo_v: u64,
}

#[inline]
pub fn wall_masks(board: &Board) -> WallMasks {
    let h = board.horizontal_walls;
    let v = board.vertical_walls;
    let coll_h = !(h | v | east1(h) | west1(h));
    let coll_v = !(v | h | (v << 8) | (v >> 8));
    WallMasks {
        l12_h: !h & coll_h,
        l12_v: !v & coll_v,
        topo_h: topo_h_from(h, v),
        topo_v: topo_v_from(h, v),
    }
}

#[inline]
pub fn wall_collision_clear_h_mask(board: &Board) -> u64 {
    let h = board.horizontal_walls;
    !(h | board.vertical_walls | east1(h) | west1(h))
}

#[inline]
pub fn wall_collision_clear_v_mask(board: &Board) -> u64 {
    let v = board.vertical_walls;
    !(v | board.horizontal_walls | (v << 8) | (v >> 8))
}

pub fn wall_l12_h_mask(board: &Board) -> u64 {
    wall_masks(board).l12_h
}

pub fn wall_l12_v_mask(board: &Board) -> u64 {
    wall_masks(board).l12_v
}

#[inline]
pub fn wall_needs_flood_h_mask(board: &Board) -> u64 {
    topo_h_from(board.horizontal_walls, board.vertical_walls)
}

#[inline]
pub fn wall_needs_flood_v_mask(board: &Board) -> u64 {
    topo_v_from(board.horizontal_walls, board.vertical_walls)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::Board;

    #[test]
    fn wall_physical_matches_scalar_collides() {
        let b = Board::new();
        for hr in 0..8u8 {
            for hc in 0..8u8 {
                let shift = wall_physically_legal_shift(&b, hr, hc, true);
                let scalar = !crate::movegen::legal::wall_collides_test(
                    &b,
                    hr,
                    hc,
                    WallOrientation::Horizontal,
                );
                assert_eq!(shift, scalar, "h {hr},{hc}");
            }
        }
    }

    fn scalar_collision_clear_h_mask(board: &Board) -> u64 {
        let mut m = 0u64;
        for r in 0..8u8 {
            for c in 0..8u8 {
                if !crate::movegen::legal::wall_collides_test(
                    board,
                    r,
                    c,
                    WallOrientation::Horizontal,
                ) {
                    m |= 1 << ((r as u64) * 8 + c as u64);
                }
            }
        }
        m
    }

    #[test]
    fn wall_masks_agrees_with_split_masks() {
        let boards = [Board::new(), {
            let mut b = Board::new();
            b.horizontal_walls = 0x00_00_0A_00_14_00;
            b.vertical_walls = 0x01_02_04_00;
            b
        }];
        for b in &boards {
            let m = wall_masks(b);
            assert_eq!(
                m.l12_h,
                !b.horizontal_walls & wall_collision_clear_h_mask(b)
            );
            assert_eq!(m.l12_v, !b.vertical_walls & wall_collision_clear_v_mask(b));
            assert_eq!(m.topo_h, wall_needs_flood_h_mask(b));
            assert_eq!(m.topo_v, wall_needs_flood_v_mask(b));
        }
    }

    #[test]
    fn collision_clear_mask_matches_scalar() {
        let boards = [Board::new(), {
            let mut b = Board::new();
            b.horizontal_walls = 0x00_00_0A_00_14_00;
            b.vertical_walls = 0x01_02_04_00;
            b
        }];
        for b in &boards {
            assert_eq!(
                wall_collision_clear_h_mask(b),
                scalar_collision_clear_h_mask(b)
            );
        }
    }

    fn scalar_topo_h_mask(board: &Board) -> u64 {
        let mut m = 0u64;
        for r in 0..8u8 {
            for c in 0..8u8 {
                if crate::movegen::legal::can_wall_block_topology(
                    board,
                    r,
                    c,
                    WallOrientation::Horizontal,
                ) {
                    m |= 1 << ((r as u64) * 8 + c as u64);
                }
            }
        }
        m
    }

    fn scalar_topo_v_mask(board: &Board) -> u64 {
        let mut m = 0u64;
        for r in 0..8u8 {
            for c in 0..8u8 {
                if crate::movegen::legal::can_wall_block_topology(
                    board,
                    r,
                    c,
                    WallOrientation::Vertical,
                ) {
                    m |= 1 << ((r as u64) * 8 + c as u64);
                }
            }
        }
        m
    }

    #[test]
    fn topo_needs_flood_matches_scalar() {
        let boards = [
            Board::new(),
            {
                let mut b = Board::new();
                b.horizontal_walls = 0x00_00_0A_00_14_00;
                b.vertical_walls = 0x01_02_04_00;
                b
            },
            {
                let mut b = Board::new();
                b.horizontal_walls = 0xFF_FF_FF_FF_FF_FF;
                b.vertical_walls = 0xFF_FF_FF_FF_FF_FF;
                b
            },
        ];
        for b in &boards {
            assert_eq!(wall_needs_flood_h_mask(b), scalar_topo_h_mask(b), "h topo");
            assert_eq!(wall_needs_flood_v_mask(b), scalar_topo_v_mask(b), "v topo");
        }
    }

    #[test]
    fn topo_needs_flood_exhaustive_low_wall_count() {
        for hw in 0u64..64 {
            for vw in 0u64..64 {
                if hw.count_ones() + vw.count_ones() > 6 {
                    continue;
                }
                let mut b = Board::new();
                b.horizontal_walls = hw;
                b.vertical_walls = vw;
                assert_eq!(
                    wall_needs_flood_h_mask(&b),
                    scalar_topo_h_mask(&b),
                    "hw={hw:#x} vw={vw:#x} h"
                );
                assert_eq!(
                    wall_needs_flood_v_mask(&b),
                    scalar_topo_v_mask(&b),
                    "hw={hw:#x} vw={vw:#x} v"
                );
            }
        }
    }
}
