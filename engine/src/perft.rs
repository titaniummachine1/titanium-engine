//! Perft (divide) — correctness + performance harness for move generation.

use crate::board::{Board, Move};
use crate::moves::generate_legal_moves;
use std::collections::BTreeMap;

pub fn perft(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut nodes = 0u64;
    for mv in generate_legal_moves(board) {
        let mut next = board.clone();
        next.apply_move(mv);
        nodes += perft(&next, depth - 1);
    }
    nodes
}

pub fn perft_divide(board: &Board, depth: u32) -> (u64, BTreeMap<String, u64>) {
    let mut total = 0u64;
    let mut lines = BTreeMap::new();
    for mv in generate_legal_moves(board) {
        let label = format_move(mv);
        let mut next = board.clone();
        next.apply_move(mv);
        let nodes = perft(&next, depth.saturating_sub(1));
        lines.insert(label, nodes);
        total += nodes;
    }
    (total, lines)
}

pub fn format_move(mv: Move) -> String {
    match mv {
        Move::Pawn { row, col } => Board::format_square(row, col),
        Move::Wall {
            row,
            col,
            orientation,
        } => {
            let suffix = match orientation {
                crate::board::WallOrientation::Horizontal => 'h',
                crate::board::WallOrientation::Vertical => 'v',
            };
            format!("{}{}{}", Board::column_char(col), row + 1, suffix)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perft_depth1_start() {
        let board = Board::new();
        assert_eq!(perft(&board, 1), generate_legal_moves(&board).len() as u64);
    }

    #[test]
    fn perft_depth0_is_one() {
        let board = Board::new();
        assert_eq!(perft(&board, 0), 1);
    }
}
