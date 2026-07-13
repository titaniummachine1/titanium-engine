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

const fn walls_touch(a: usize, b: usize) -> bool {
    let a_h = a < 64;
    let b_h = b < 64;
    let ai = a % 64;
    let bi = b % 64;
    let ar = (ai / 8) as i16;
    let ac = (ai % 8) as i16;
    let br = (bi / 8) as i16;
    let bc = (bi % 8) as i16;

    if a_h && b_h {
        return ar == br && (ac - bc).abs() <= 2;
    }
    if !a_h && !b_h {
        return ac == bc && (ar - br).abs() <= 2;
    }

    let (hr, hc, vr, vc) = if a_h {
        (ar, ac, br, bc)
    } else {
        (br, bc, ar, ac)
    };
    let vx = vc + 1;
    let hy = hr + 1;
    vx >= hc && vx <= hc + 2 && hy >= vr && hy <= vr + 2
}

const fn build_wall_touch_masks() -> [u128; 128] {
    let mut masks = [0u128; 128];
    let mut a = 0usize;
    while a < 128 {
        let mut b = 0usize;
        while b < 128 {
            if walls_touch(a, b) {
                masks[a] |= 1u128 << b;
            }
            b += 1;
        }
        a += 1;
    }
    masks
}

const fn build_edge_touch_mask() -> u128 {
    let mut edge = 0u128;
    let mut wall = 0usize;
    while wall < 128 {
        let slot = wall % 64;
        let row = slot / 8;
        let col = slot % 8;
        let touches = if wall < 64 {
            col == 0 || col == 7
        } else {
            row == 0 || row == 7
        };
        if touches {
            edge |= 1u128 << wall;
        }
        wall += 1;
    }
    edge
}

/// Wall-lattice contact masks. Horizontal slots are bits 0..63; vertical
/// slots are bits 64..127. The candidate itself and physical collisions are
/// included, although normal callers already apply the L1/L2 collision mask.
pub const WALL_TOUCH_MASKS: [u128; 128] = build_wall_touch_masks();
pub const WALL_EDGE_MASK: u128 = build_edge_touch_mask();

#[inline]
pub fn wall_occupied_mask(board: &Board) -> u128 {
    board.horizontal_walls as u128 | ((board.vertical_walls as u128) << 64)
}

#[cfg(test)]
#[inline]
pub fn wall_is_strictly_isolated(board: &Board, slot: usize, horizontal: bool) -> bool {
    let wall = slot + if horizontal { 0 } else { 64 };
    WALL_EDGE_MASK & (1u128 << wall) == 0 && wall_occupied_mask(board) & WALL_TOUCH_MASKS[wall] == 0
}

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

/// O(1) flood-skip mask from packed wall bitboards (no `Board` alloc).
#[inline]
pub fn wall_needs_flood_h_from_bits(horizontal_walls: u64, vertical_walls: u64) -> u64 {
    topo_h_from(horizontal_walls, vertical_walls)
}

#[inline]
pub fn wall_needs_flood_v_from_bits(horizontal_walls: u64, vertical_walls: u64) -> u64 {
    topo_v_from(horizontal_walls, vertical_walls)
}

/// True when a candidate wall can touch enough topology to possibly seal.
///
/// **Warning:** recomputes the whole-board topo mask per call. Prefer testing a
/// precomputed `topo_h`/`topo_v` bit from [`wall_masks`] inside candidate loops.
#[inline]
pub fn wall_slot_needs_flood_recomputing_mask(
    horizontal_walls: u64,
    vertical_walls: u64,
    horizontal: bool,
    slot: usize,
) -> bool {
    let mask = if horizontal {
        topo_h_from(horizontal_walls, vertical_walls)
    } else {
        topo_v_from(horizontal_walls, vertical_walls)
    };
    (mask >> slot) & 1 != 0
}

/// Alias kept for call sites that probe a single slot without a cached mask.
#[inline]
pub fn wall_slot_needs_flood(
    horizontal_walls: u64,
    vertical_walls: u64,
    horizontal: bool,
    slot: usize,
) -> bool {
    wall_slot_needs_flood_recomputing_mask(horizontal_walls, vertical_walls, horizontal, slot)
}

/// ACE/JS anchor-count precheck — conservative O(25) per slot (bench baseline).
#[inline]
pub fn wall_slot_needs_flood_anchor(
    horizontal_walls: u64,
    vertical_walls: u64,
    horizontal: bool,
    slot: usize,
) -> bool {
    let r = (slot / 8) as i32;
    let c = (slot % 8) as i32;
    let mut anchors = 0;
    if horizontal {
        if c == 0 {
            anchors += 1;
        }
        if c == 7 {
            anchors += 1;
        }
    } else {
        if r == 0 {
            anchors += 1;
        }
        if r == 7 {
            anchors += 1;
        }
    }
    let mut dr = -2;
    while dr <= 2 && anchors < 2 {
        let rr = r + dr;
        if rr < 0 || rr > 7 {
            dr += 1;
            continue;
        }
        let mut dc = -2;
        while dc <= 2 {
            let cc = c + dc;
            if cc < 0 || cc > 7 {
                dc += 1;
                continue;
            }
            let ss = (rr * 8 + cc) as usize;
            if (horizontal_walls >> ss) & 1 != 0 || (vertical_walls >> ss) & 1 != 0 {
                anchors += 1;
                if anchors >= 2 {
                    return true;
                }
            }
            dc += 1;
        }
        dr += 1;
    }
    anchors >= 2
}

fn wall_needs_flood_h_anchor_mask(horizontal_walls: u64, vertical_walls: u64) -> u64 {
    let mut mask = 0u64;
    for slot in 0..64usize {
        if wall_slot_needs_flood_anchor(horizontal_walls, vertical_walls, true, slot) {
            mask |= 1u64 << slot;
        }
    }
    mask
}

fn wall_needs_flood_v_anchor_mask(horizontal_walls: u64, vertical_walls: u64) -> u64 {
    let mut mask = 0u64;
    for slot in 0..64usize {
        if wall_slot_needs_flood_anchor(horizontal_walls, vertical_walls, false, slot) {
            mask |= 1u64 << slot;
        }
    }
    mask
}

/// Anchor-count flood-skip baseline — for A/B benches only (never call from production movegen).
#[inline]
pub fn wall_needs_flood_masks_anchor_baseline(board: &Board) -> (u64, u64) {
    let h = board.horizontal_walls;
    let v = board.vertical_walls;
    (
        wall_needs_flood_h_anchor_mask(h, v),
        wall_needs_flood_v_anchor_mask(h, v),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::Board;

    #[test]
    fn wall_touch_masks_match_lattice_examples() {
        let h = |r: usize, c: usize| r * 8 + c;
        let v = |r: usize, c: usize| 64 + r * 8 + c;
        assert_ne!(WALL_TOUCH_MASKS[h(3, 3)] & (1u128 << h(3, 5)), 0);
        assert_eq!(WALL_TOUCH_MASKS[h(3, 3)] & (1u128 << h(3, 6)), 0);
        assert_ne!(WALL_TOUCH_MASKS[h(3, 3)] & (1u128 << v(2, 2)), 0);
        assert_eq!(WALL_TOUCH_MASKS[h(3, 3)] & (1u128 << v(0, 0)), 0);
        assert_ne!(WALL_EDGE_MASK & (1u128 << h(4, 0)), 0);
        assert_ne!(WALL_EDGE_MASK & (1u128 << v(7, 4)), 0);
        assert_eq!(WALL_EDGE_MASK & (1u128 << h(4, 4)), 0);
    }

    fn barrier_points(wall: usize) -> [(usize, usize); 3] {
        let slot = wall % 64;
        let row = slot / 8;
        let col = slot % 8;
        if wall < 64 {
            [(row + 1, col), (row + 1, col + 1), (row + 1, col + 2)]
        } else {
            [(row, col + 1), (row + 1, col + 1), (row + 2, col + 1)]
        }
    }

    #[test]
    fn wall_contact_lut_matches_lattice_geometry() {
        for a in 0..128usize {
            let a_points = barrier_points(a);
            let a_edge = a_points
                .iter()
                .any(|&(row, col)| row == 0 || row == 9 || col == 0 || col == 9);
            assert_eq!(WALL_EDGE_MASK & (1u128 << a) != 0, a_edge, "edge {a}");
            for b in 0..128usize {
                let b_points = barrier_points(b);
                let touches = a_points.iter().any(|point| b_points.contains(point));
                assert_eq!(
                    WALL_TOUCH_MASKS[a] & (1u128 << b) != 0,
                    touches,
                    "contact {a},{b}"
                );
            }
        }
    }

    #[test]
    fn isolation_masks_exhaust_local_occupancy_patterns() {
        for wall in 0..128usize {
            let touching = WALL_TOUCH_MASKS[wall] & !(1u128 << wall);
            let bits: Vec<_> = (0..128)
                .filter(|&bit| touching & (1u128 << bit) != 0)
                .collect();
            assert!(
                bits.len() <= 16,
                "wall {wall} has {} touching slots",
                bits.len()
            );
            for pattern in 0usize..(1usize << bits.len()) {
                let mut occupied = 0u128;
                for (i, &bit) in bits.iter().enumerate() {
                    if pattern & (1usize << i) != 0 {
                        occupied |= 1u128 << bit;
                    }
                }
                let mut board = Board::new();
                board.horizontal_walls = occupied as u64;
                board.vertical_walls = (occupied >> 64) as u64;
                let isolated = wall_is_strictly_isolated(&board, wall % 64, wall < 64);
                assert_eq!(
                    isolated,
                    pattern == 0 && WALL_EDGE_MASK & (1u128 << wall) == 0
                );
            }
        }
    }

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
