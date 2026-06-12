//! Runtime pawn lookup + wall O(1) tables.
//!
//! Pawns: enemy_key + wall_key → `PAWN_LEGAL` mask.
//!
//! Walls (3 layers + topology opt):
//! - L1: empty slot (`!board.horizontal_walls` / `!vertical_walls`) — bitboard, no table
//! - L2: collision rules (overlap / cross / neighbor) — `WALL_PHYSICAL_TABLE` per slot
//! - Topo: `can_wall_block_topology` O(1) — skip L3 flood when isolated (not a placement rule)
//! - L3: parallel flood — `legal.rs` when topo says wall can cage someone

use crate::core::board::{Board, Move, WallOrientation};
use crate::util::grid::{has_wall, square_index};

use super::tables::{
    wall_remap_byte, wall_topo_h_remap_byte, wall_topo_v_remap_byte, PAWN_CATALOG, PAWN_LAYER_VALID,
    PAWN_LEGAL, PAWN_WALL_COMBO_COUNT, PAWN_WALL_DESC_COL, PAWN_WALL_DESC_H, PAWN_WALL_DESC_ROW,
    PAWN_WALL_SLOT_COUNT, WALL_COLLISION_MASK, WALL_PHYSICAL_TABLE, WALL_SLOT_COL,
    WALL_SLOT_HORIZONTAL, WALL_SLOT_ROW, WALL_TOPO_H, WALL_TOPO_H_PROBE_COL, WALL_TOPO_H_PROBE_COUNT,
    WALL_TOPO_H_PROBE_H, WALL_TOPO_H_PROBE_ROW, WALL_TOPO_V, WALL_TOPO_V_PROBE_COL,
    WALL_TOPO_V_PROBE_COUNT, WALL_TOPO_V_PROBE_H, WALL_TOPO_V_PROBE_ROW,
};

/// Layer 1: 0=opponent absent, 1=up, 2=down, 3=left, 4=right (edge-invalid → 0).
pub fn encode_enemy_key(board: &Board, side: usize, sq: u8) -> u8 {
    let sr = sq / 9;
    let sc = sq % 9;
    let (or, oc) = board.pawns[1 - side];
    let dr = or as i8 - sr as i8;
    let dc = oc as i8 - sc as i8;
    let ek = match (dr, dc) {
        (-1, 0) => 1,
        (1, 0) => 2,
        (0, -1) => 3,
        (0, 1) => 4,
        _ => return 0,
    };
    if PAWN_LAYER_VALID[sq as usize][ek as usize] == 0 {
        0
    } else {
        ek
    }
}

/// Pack physical wall combo (up to 12 local slots), remap to 8-bit semantic key.
pub fn pack_wall_key(board: &Board, sq: u8, enemy_key: u8) -> u8 {
    let nw = PAWN_WALL_SLOT_COUNT[sq as usize][enemy_key as usize] as usize;
    let mut phys = 0usize;
    for i in 0..nw {
        let r = PAWN_WALL_DESC_ROW[sq as usize][enemy_key as usize][i];
        let c = PAWN_WALL_DESC_COL[sq as usize][enemy_key as usize][i];
        let h = PAWN_WALL_DESC_H[sq as usize][enemy_key as usize][i] != 0;
        let orient = if h {
            WallOrientation::Horizontal
        } else {
            WallOrientation::Vertical
        };
        if has_wall(board, r, c, orient) {
            phys |= 1 << i;
        }
    }
    wall_remap_byte(sq, enemy_key, phys)
}

#[inline]
pub fn legal_pawn_move_mask(board: &Board, side: usize, sq: u8) -> u16 {
    let enemy_key = encode_enemy_key(board, side, sq);
    if PAWN_LAYER_VALID[sq as usize][enemy_key as usize] == 0 {
        return 0;
    }
    let wall_key = pack_wall_key(board, sq, enemy_key);
    let max = PAWN_WALL_COMBO_COUNT[sq as usize][enemy_key as usize] as usize;
    if wall_key as usize >= max {
        return 0;
    }
    PAWN_LEGAL[sq as usize][enemy_key as usize][wall_key as usize]
}

pub fn generate_pawn_moves_o1(board: &Board, out: &mut [Move]) -> usize {
    let side = board.side_to_move as usize;
    let (fr, fc) = board.pawns[side];
    let sq = square_index(fr, fc);
    let mask = legal_pawn_move_mask(board, side, sq);
    let catalog = &PAWN_CATALOG[sq as usize];
    let mut n = 0usize;
    let mut bits = mask;
    while bits != 0 {
        let slot = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let dest = catalog[slot];
        if dest == 255 {
            continue;
        }
        out[n] = Move::Pawn {
            row: dest / 9,
            col: dest % 9,
        };
        n += 1;
    }
    n
}

#[inline]
pub fn wall_slot_id(row: u8, col: u8, horizontal: bool) -> usize {
    let base = if horizontal { 0 } else { 64 };
    base + (row as usize) * 8 + col as usize
}

fn collision_occupied(board: &Board, probe: (u8, u8, bool)) -> bool {
    let (r, c, h) = probe;
    has_wall(
        board,
        r,
        c,
        if h {
            WallOrientation::Horizontal
        } else {
            WallOrientation::Vertical
        },
    )
}

fn collision_probes(row: u8, col: u8, horizontal: bool) -> [(u8, u8, bool); 6] {
    let mut out = [(255u8, 255, false); 6];
    out[0] = (row, col, true);
    out[1] = (row, col, false);
    let mut n = 2usize;
    if horizontal {
        if col > 0 {
            out[n] = (row, col - 1, true);
            n += 1;
        }
        if col < 7 {
            out[n] = (row, col + 1, true);
        }
    } else {
        if row > 0 {
            out[n] = (row - 1, col, false);
            n += 1;
        }
        if row < 7 {
            out[n] = (row + 1, col, false);
        }
    }
    out
}

#[inline]
pub fn pack_wall_collision_bits(board: &Board, slot: usize) -> u8 {
    let row = WALL_SLOT_ROW[slot];
    let col = WALL_SLOT_COL[slot];
    let horizontal = WALL_SLOT_HORIZONTAL[slot] != 0;
    let probes = collision_probes(row, col, horizontal);
    let mask = WALL_COLLISION_MASK[slot];
    let mut local = 0u8;
    let mut bit = 0u8;
    let mut m = mask;
    while m != 0 {
        let idx = m.trailing_zeros() as usize;
        m &= m - 1;
        if idx < 6 && probes[idx].0 != 255 && collision_occupied(board, probes[idx]) {
            local |= 1 << bit;
        }
        bit += 1;
    }
    local
}

/// L2: passes overlap / cross / neighbor collision rules (`wall_collides` inverse).
#[inline]
pub fn wall_physically_legal_o1(board: &Board, row: u8, col: u8, horizontal: bool) -> bool {
    let slot = wall_slot_id(row, col, horizontal);
    let local = pack_wall_collision_bits(board, slot);
    WALL_PHYSICAL_TABLE[slot][local as usize] != 0
}

fn collision_clear_mask(board: &Board, horizontal: bool) -> u64 {
    let mut m = 0u64;
    for row in 0..8u8 {
        for col in 0..8u8 {
            if wall_physically_legal_o1(board, row, col, horizontal) {
                m |= 1 << ((row as u64) * 8 + col as u64);
            }
        }
    }
    m
}

/// L2 horizontal: bit set ⇒ no collision with existing walls.
#[inline]
pub fn wall_collision_clear_h_mask(board: &Board) -> u64 {
    collision_clear_mask(board, true)
}

/// L2 vertical: bit set ⇒ no collision with existing walls.
#[inline]
pub fn wall_collision_clear_v_mask(board: &Board) -> u64 {
    collision_clear_mask(board, false)
}

/// L1∧L2 horizontal candidates (empty ∧ collision-clear).
pub fn wall_l12_h_mask(board: &Board) -> u64 {
    !board.horizontal_walls & wall_collision_clear_h_mask(board)
}

/// L1∧L2 vertical candidates.
pub fn wall_l12_v_mask(board: &Board) -> u64 {
    !board.vertical_walls & wall_collision_clear_v_mask(board)
}

pub fn generate_wall_candidates_o1(board: &Board, horizontal: bool, out: &mut [(u8, u8)]) -> usize {
    let bits = if horizontal {
        wall_l12_h_mask(board)
    } else {
        wall_l12_v_mask(board)
    };
    let mut n = 0usize;
    let mut free = bits;
    while free != 0 {
        let bit = free.trailing_zeros();
        free &= free - 1;
        out[n] = ((bit / 8) as u8, (bit % 8) as u8);
        n += 1;
    }
    n
}

#[inline]
pub fn pack_wall_topo_h_key(board: &Board) -> u8 {
    pack_wall_probe_key(
        board,
        &WALL_TOPO_H_PROBE_ROW[..],
        &WALL_TOPO_H_PROBE_COL[..],
        &WALL_TOPO_H_PROBE_H[..],
        WALL_TOPO_H_PROBE_COUNT,
        wall_topo_h_remap_byte,
    )
}

#[inline]
pub fn pack_wall_topo_v_key(board: &Board) -> u8 {
    pack_wall_probe_key(
        board,
        &WALL_TOPO_V_PROBE_ROW[..],
        &WALL_TOPO_V_PROBE_COL[..],
        &WALL_TOPO_V_PROBE_H[..],
        WALL_TOPO_V_PROBE_COUNT,
        wall_topo_v_remap_byte,
    )
}

/// Bit set ⇒ `can_wall_block_topology` — must run L3 flood before accepting.
#[inline]
pub fn wall_needs_flood_h_mask(board: &Board) -> u64 {
    WALL_TOPO_H[pack_wall_topo_h_key(board) as usize]
}

/// Bit set ⇒ `can_wall_block_topology` — must run L3 flood before accepting.
#[inline]
pub fn wall_needs_flood_v_mask(board: &Board) -> u64 {
    WALL_TOPO_V[pack_wall_topo_v_key(board) as usize]
}

fn pack_wall_probe_key(
    board: &Board,
    rows: &[u8],
    cols: &[u8],
    hs: &[u8],
    count: u8,
    remap: fn(usize) -> u8,
) -> u8 {
    let nw = count as usize;
    let mut phys = 0usize;
    for i in 0..nw {
        let orient = if hs[i] != 0 {
            WallOrientation::Horizontal
        } else {
            WallOrientation::Vertical
        };
        if has_wall(board, rows[i], cols[i], orient) {
            phys |= 1 << i;
        }
    }
    remap(phys)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::{Board, Player};
    use crate::movegen::legal::generate_pawn_moves_scalar_for;

    fn scalar_mask(board: &Board, player: Player, sq: u8) -> u16 {
        let mut moves = [Move::Pawn { row: 0, col: 0 }; 8];
        let n = generate_pawn_moves_scalar_for(board, player, &mut moves);
        let catalog = &PAWN_CATALOG[sq as usize];
        let mut mask = 0u16;
        for m in &moves[..n] {
            if let Move::Pawn { row, col } = m {
                let d = square_index(*row, *col);
                for (slot, &sq_id) in catalog.iter().enumerate() {
                    if sq_id != 255 && sq_id == d {
                        mask |= 1 << slot;
                        break;
                    }
                }
            }
        }
        mask
    }

    #[test]
    fn o1_pawn_matches_scalar_startpos() {
        let b = Board::new();
        for player in [Player::One, Player::Two] {
            let side = player as usize;
            let sq = square_index(b.pawns[side].0, b.pawns[side].1);
            assert_eq!(
                legal_pawn_move_mask(&b, side, sq),
                scalar_mask(&b, player, sq),
                "{player:?}"
            );
        }
    }

    #[test]
    fn o1_pawn_matches_scalar_walls() {
        let mut b = Board::new();
        b.horizontal_walls = 0x00_00_0A_00_14_00;
        b.vertical_walls = 0x01_02_04_00;
        for player in [Player::One, Player::Two] {
            let side = player as usize;
            let sq = square_index(b.pawns[side].0, b.pawns[side].1);
            assert_eq!(
                legal_pawn_move_mask(&b, side, sq),
                scalar_mask(&b, player, sq)
            );
        }
    }

    #[test]
    fn all_wall_slots_fit_8_bits() {
        for sq in 0u8..81 {
            for ek in 0u8..5 {
                if PAWN_LAYER_VALID[sq as usize][ek as usize] == 0 {
                    continue;
                }
                let max = PAWN_WALL_COMBO_COUNT[sq as usize][ek as usize];
                assert!(
                    max <= 256,
                    "sq {sq} enemy {ek}: {max} combos need >8 wall bits"
                );
                let nw = PAWN_WALL_SLOT_COUNT[sq as usize][ek as usize];
                assert!(nw <= 12, "sq {sq} enemy {ek}: {nw} wall slots");
            }
        }
    }

    #[test]
    fn wall_physical_matches_scalar_collides() {
        let b = Board::new();
        for hr in 0..8u8 {
            for hc in 0..8u8 {
                let o1 = wall_physically_legal_o1(&b, hr, hc, true);
                let scalar = !crate::movegen::legal::wall_collides_test(
                    &b,
                    hr,
                    hc,
                    WallOrientation::Horizontal,
                );
                assert_eq!(o1, scalar, "h {hr},{hc}");
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
    fn collision_clear_mask_matches_scalar() {
        let boards = [
            Board::new(),
            {
                let mut b = Board::new();
                b.horizontal_walls = 0x00_00_0A_00_14_00;
                b.vertical_walls = 0x01_02_04_00;
                b
            },
        ];
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
    fn topo_probe_count_fits_ten_bits() {
        assert!(WALL_TOPO_H_PROBE_COUNT as usize <= 10);
        assert!(WALL_TOPO_V_PROBE_COUNT as usize <= 10);
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
