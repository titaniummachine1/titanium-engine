//! Minimal Quoridor geometry for offline table discovery (mirrors `util::grid`).

#[derive(Clone, Copy, PartialEq, Eq, Default)]
pub struct MiniBoard {
    pub pawns: [(u8, u8); 2],
    pub horizontal_walls: u64,
    pub vertical_walls: u64,
}

#[inline]
pub fn square_index(row: u8, col: u8) -> u8 {
    row * 9 + col
}

#[inline]
fn has_horizontal(b: &MiniBoard, js_row: u8, col: u8) -> bool {
    if !(1..=8).contains(&js_row) || col >= 8 {
        return false;
    }
    let bit = ((js_row - 1) as u32) * 8 + col as u32;
    (b.horizontal_walls >> bit) & 1 != 0
}

#[inline]
fn has_vertical(b: &MiniBoard, js_row: u8, col: u8) -> bool {
    if !(1..=8).contains(&js_row) || col >= 8 {
        return false;
    }
    let bit = ((js_row - 1) as u32) * 8 + col as u32;
    (b.vertical_walls >> bit) & 1 != 0
}

#[inline]
pub fn has_wall(b: &MiniBoard, row: u8, col: u8, horizontal: bool) -> bool {
    let js_row = row + 1;
    if horizontal {
        has_horizontal(b, js_row, col)
    } else {
        has_vertical(b, js_row, col)
    }
}

#[inline]
pub fn can_step(b: &MiniBoard, row: u8, col: u8, dr: i8, dc: i8) -> bool {
    let nr = row as i16 + dr as i16;
    let nc = col as i16 + dc as i16;
    if !(0..=8).contains(&nr) || !(0..=8).contains(&nc) {
        return false;
    }
    let nr = nr as u8;
    let nc = nc as u8;
    let js_from = row + 1;
    let js_to = nr + 1;

    match (dr, dc) {
        (1, 0) => {
            !has_horizontal(b, js_from, col)
                && (col == 0 || !has_horizontal(b, js_from, col - 1))
        }
        (-1, 0) => {
            !has_horizontal(b, js_to, col)
                && (col == 0 || !has_horizontal(b, js_to, col - 1))
        }
        (0, 1) => !has_vertical(b, js_from, col) && !has_vertical(b, row, col),
        (0, -1) => !has_vertical(b, js_to, nc) && !has_vertical(b, nr, nc),
        _ => false,
    }
}

const DIRS: [(i8, i8); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

/// Destination squares (0..80) for legal pawn moves from `from_sq`.
pub fn pawn_move_dests(b: &MiniBoard, side: usize, from_sq: u8) -> ([u8; 8], usize) {
    let (fr, fc) = b.pawns[side];
    if square_index(fr, fc) != from_sq {
        return ([0; 8], 0);
    }
    let (or, oc) = b.pawns[1 - side];
    let mut dests = [0u8; 8];
    let mut n = 0usize;

    for (dr, dc) in DIRS {
        if !can_step(b, fr, fc, dr, dc) {
            continue;
        }
        let nr = (fr as i8 + dr) as u8;
        let nc = (fc as i8 + dc) as u8;

        if (nr, nc) != (or, oc) {
            dests[n] = square_index(nr, nc);
            n += 1;
            continue;
        }

        if can_step(b, nr, nc, dr, dc) {
            let jr = (nr as i8 + dr) as u8;
            let jc = (nc as i8 + dc) as u8;
            dests[n] = square_index(jr, jc);
            n += 1;
            continue;
        }

        let perp = if dr != 0 {
            [(0i8, 1i8), (0, -1)]
        } else {
            [(1, 0), (-1, 0)]
        };
        for (pdr, pdc) in perp {
            if can_step(b, nr, nc, pdr, pdc) {
                let sr = (nr as i8 + pdr) as u8;
                let sc = (nc as i8 + pdc) as u8;
                dests[n] = square_index(sr, sc);
                n += 1;
            }
        }
    }
    (dests, n)
}

/// Encode legal destinations into the 12 fixed semantic slots for this square.
pub fn encode_dest_slots_fixed(fixed: &[u8; 12], dests: &[u8], n: usize) -> u16 {
    let mut mask = 0u16;
    for &d in &dests[..n] {
        for (slot, &sq) in fixed.iter().enumerate() {
            if sq != 255 && sq == d {
                assert!(slot < 12);
                mask |= 1 << slot;
                break;
            }
        }
    }
    mask
}

/// Twelve possible pawn outcomes from `(sr, sc)`:
/// 0–3 cardinal steps, 4–7 jump-through (opp on cardinal + clear behind),
/// 8–11 diagonal slides when jump is blocked. `255` = off-board at edges.
pub fn fixed_twelve_destinations(sr: u8, sc: u8) -> [u8; 12] {
    const OFF: u8 = 255;
    let sq = |r: i16, c: i16| -> u8 {
        if (0..=8).contains(&r) && (0..=8).contains(&c) {
            square_index(r as u8, c as u8)
        } else {
            OFF
        }
    };
    let r = sr as i16;
    let c = sc as i16;
    [
        sq(r + 1, c),     // 0 step toward row+
        sq(r - 1, c),     // 1 step toward row-
        sq(r, c + 1),     // 2 step east
        sq(r, c - 1),     // 3 step west
        sq(r + 2, c),     // 4 jump over S-adjacent opp
        sq(r - 2, c),     // 5 jump over N-adjacent opp
        sq(r, c + 2),     // 6 jump over E-adjacent opp
        sq(r, c - 2),     // 7 jump over W-adjacent opp
        sq(r - 1, c + 1), // 8 slide when N or E jump blocked
        sq(r - 1, c - 1), // 9 slide when N or W jump blocked
        sq(r + 1, c + 1), // 10 slide when S or E jump blocked
        sq(r + 1, c - 1), // 11 slide when S or W jump blocked
    ]
}

pub fn valid_twelve_slot_count(fixed: &[u8; 12]) -> u8 {
    fixed.iter().filter(|&&sq| sq != 255).count() as u8
}

/// Physical wall placement — overlap/cross only (no path check).
pub fn wall_physically_legal(b: &MiniBoard, row: u8, col: u8, horizontal: bool) -> bool {
    if has_wall(b, row, col, horizontal) || has_wall(b, row, col, !horizontal) {
        return false;
    }
    if horizontal {
        if col > 0 && has_wall(b, row, col - 1, true) {
            return false;
        }
        if col < 7 && has_wall(b, row, col + 1, true) {
            return false;
        }
    } else if row > 0 && has_wall(b, row - 1, col, false) {
        return false;
    } else if row < 7 && has_wall(b, row + 1, col, false) {
        return false;
    }
    true
}

pub fn set_wall(b: &mut MiniBoard, row: u8, col: u8, horizontal: bool) {
    let bit = (row as u64) * 8 + col as u64;
    if horizontal {
        b.horizontal_walls |= 1 << bit;
    } else {
        b.vertical_walls |= 1 << bit;
    }
}

pub fn clear_wall(b: &mut MiniBoard, row: u8, col: u8, horizontal: bool) {
    let bit = (row as u64) * 8 + col as u64;
    if horizontal {
        b.horizontal_walls &= !(1 << bit);
    } else {
        b.vertical_walls &= !(1 << bit);
    }
}

/// Matches scraped `canWallBlock` — wall must touch existing topology to cage anyone.
pub fn can_wall_block_topology(b: &MiniBoard, row: u8, col: u8, horizontal: bool) -> bool {
    let js_col = col + 1;
    let js_row = row + 1;

    let (on_a, on_b) = if horizontal {
        (js_col == 1, js_col == 8)
    } else {
        (js_row == 8, js_row == 1)
    };

    let side_a = on_a || touching_side_a(b, row, col, horizontal);
    let side_b = on_b || touching_side_b(b, row, col, horizontal);
    let middle = touching_middle(b, row, col, horizontal);

    (side_a && side_b) || (side_a && middle) || (side_b && middle)
}

fn touching_side_a(b: &MiniBoard, row: u8, col: u8, horizontal: bool) -> bool {
    if horizontal {
        wall_at_offset(b, row, col, &[(0, -1)], false)
            || wall_at_offset(b, row, col, &[(1, 0), (0, -1)], false)
            || wall_at_offset(b, row, col, &[(-1, 0), (0, -1)], false)
            || wall_at_offset(b, row, col, &[(0, -1), (0, -1)], true)
    } else {
        wall_at_offset(b, row, col, &[(1, 0)], true)
            || wall_at_offset(b, row, col, &[(0, -1), (1, 0)], true)
            || wall_at_offset(b, row, col, &[(0, 1), (1, 0)], true)
            || wall_at_offset(b, row, col, &[(1, 0), (1, 0)], false)
    }
}

fn touching_side_b(b: &MiniBoard, row: u8, col: u8, horizontal: bool) -> bool {
    if horizontal {
        wall_at_offset(b, row, col, &[(0, 1)], false)
            || wall_at_offset(b, row, col, &[(1, 0), (0, 1)], false)
            || wall_at_offset(b, row, col, &[(-1, 0), (0, 1)], false)
            || wall_at_offset(b, row, col, &[(0, 1), (0, 1)], true)
    } else {
        wall_at_offset(b, row, col, &[(-1, 0)], true)
            || wall_at_offset(b, row, col, &[(0, -1), (-1, 0)], true)
            || wall_at_offset(b, row, col, &[(0, 1), (-1, 0)], true)
            || wall_at_offset(b, row, col, &[(-1, 0), (-1, 0)], false)
    }
}

fn touching_middle(b: &MiniBoard, row: u8, col: u8, horizontal: bool) -> bool {
    if horizontal {
        wall_at_offset(b, row, col, &[(1, 0)], false)
            || wall_at_offset(b, row, col, &[(-1, 0)], false)
    } else {
        wall_at_offset(b, row, col, &[(0, -1)], true)
            || wall_at_offset(b, row, col, &[(0, 1)], true)
    }
}

fn wall_at_offset(
    b: &MiniBoard,
    row: u8,
    col: u8,
    offsets: &[(i8, i8)],
    horizontal: bool,
) -> bool {
    let (wr, wc) = apply_offsets(row, col, offsets);
    if wr > 7 || wc > 7 {
        return false;
    }
    has_wall(b, wr, wc, horizontal)
}

fn apply_offsets(mut row: u8, mut col: u8, offsets: &[(i8, i8)]) -> (u8, u8) {
    for (dr, dc) in offsets {
        row = (row as i16 + *dr as i16) as u8;
        col = (col as i16 + *dc as i16) as u8;
    }
    (row, col)
}
