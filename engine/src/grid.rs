//! O(1) wall checks and step logic — matches scraped `gameLogic.js` pawnCanMove / hasWall.

use crate::board::{Board, Player, WallOrientation};

/// P1 races to row 8, P2 to row 0 (internal 0..8 indexing).
#[inline]
pub fn goal_row(player: Player) -> u8 {
    match player {
        Player::One => 8,
        Player::Two => 0,
    }
}

#[inline]
pub fn is_goal(player: Player, row: u8) -> bool {
    row == goal_row(player)
}

#[inline]
fn has_horizontal(board: &Board, js_row: u8, col: u8) -> bool {
    if !(1..=8).contains(&js_row) || col >= 8 {
        return false;
    }
    let bit = ((js_row - 1) as u32) * 8 + col as u32;
    (board.horizontal_walls >> bit) & 1 != 0
}

#[inline]
fn has_vertical(board: &Board, js_row: u8, col: u8) -> bool {
    if !(1..=8).contains(&js_row) || col >= 8 {
        return false;
    }
    let bit = ((js_row - 1) as u32) * 8 + col as u32;
    (board.vertical_walls >> bit) & 1 != 0
}

/// Can the pawn at `(row, col)` step by `(dr, dc)`? Both in 0..8, steps are -1/0/1.
#[inline]
pub fn can_step(board: &Board, row: u8, col: u8, dr: i8, dc: i8) -> bool {
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
            !has_horizontal(board, js_from, col)
                && (col == 0 || !has_horizontal(board, js_from, col - 1))
        }
        (-1, 0) => {
            !has_horizontal(board, js_to, col)
                && (col == 0 || !has_horizontal(board, js_to, col - 1))
        }
        // Lateral steps: match scraped `pawnCanMove` wall anchors (see docs/video/05-first-perft-bug.md).
        // Right — wallAnchor = from, sideAnchor = step(from, Down).
        (0, 1) => !has_vertical(board, js_from, col) && !has_vertical(board, row, col),
        // Left — wallAnchor = to, sideAnchor = step(to, Down).
        (0, -1) => !has_vertical(board, js_to, nc) && !has_vertical(board, nr, nc),
        _ => false,
    }
}

#[inline]
pub fn square_index(row: u8, col: u8) -> u8 {
    row * 9 + col
}

#[inline]
pub fn unpack_square(sq: u8) -> (u8, u8) {
    (sq / 9, sq % 9)
}

pub fn set_wall(board: &mut Board, row: u8, col: u8, orientation: WallOrientation, place: bool) {
    debug_assert!((1..=8).contains(&(row + 1)) && col < 8);
    let js_row = row + 1;
    let bit = 1u64 << (((js_row - 1) as u32) * 8 + col as u32);
    match orientation {
        WallOrientation::Horizontal => {
            if place {
                board.horizontal_walls |= bit;
            } else {
                board.horizontal_walls &= !bit;
            }
        }
        WallOrientation::Vertical => {
            if place {
                board.vertical_walls |= bit;
            } else {
                board.vertical_walls &= !bit;
            }
        }
    }
}

pub fn has_wall(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    let js_row = row + 1;
    match orientation {
        WallOrientation::Horizontal => has_horizontal(board, js_row, col),
        WallOrientation::Vertical => has_vertical(board, js_row, col),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::board::{Board, WallOrientation};

    #[test]
    fn vertical_d8v_blocks_black_left_from_e9() {
        let mut board = Board::new();
        set_wall(
            &mut board,
            7,
            3,
            WallOrientation::Vertical,
            true,
        );
        board.side_to_move = crate::board::Player::Two;
        // P2 at e9 (internal 8,4) — left to d9 must be blocked by d8v.
        assert!(!can_step(&board, 8, 4, 0, -1));
        assert!(can_step(&board, 8, 4, 0, 1));
    }
}
