//! Read-only `Board` view of a titanium `GameState` (for CAT viz / tests).

use crate::core::board::Board;
use crate::titanium::game::GameState;
use crate::titanium::move_id_to_board;

/// Replay `g` into a fresh `Board` without mutating `g` or any search session.
pub fn board_from_game(g: &GameState) -> Board {
    let mut board = Board::new();
    for i in 0..g.hist_len {
        let _ = board.make_move(move_id_to_board(g.hist_m[i]));
    }
    board
}
