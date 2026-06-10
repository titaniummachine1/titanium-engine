//! Titanium move pipeline — book PV hints + iterative-deepening negamax.

use crate::core::board::Board;
use crate::opening::book::{self, BOOK_MAX_PLY};
use crate::path::BfsScratch;
use crate::search::alphabeta::genmove_algebraic as minimax_algebraic;
use crate::search::genmove::{GenmoveConfig, GenmoveEngine};
use crate::search::greedy::choose_greedy_move;
use crate::util::perft::format_move;

pub fn walls_placed(board: &Board) -> u8 {
    20u8.saturating_sub(board.walls_remaining[0].saturating_sub(board.walls_remaining[1]))
}

/// Inputs for adaptive LMR stage classification (`LmrProfile::stage_t`).
#[derive(Debug, Clone, Copy)]
pub struct LmrStageInputs {
    pub walls: u8,
    pub ply: u32,
    pub our_dist: u8,
    pub opp_dist: u8,
}

pub fn lmr_stage_inputs(board: &Board, bfs: &mut BfsScratch) -> LmrStageInputs {
    let us = board.side();
    LmrStageInputs {
        walls: walls_placed(board),
        ply: book::ply_number(board),
        our_dist: bfs.shortest_distance(board, us).unwrap_or(u8::MAX),
        opp_dist: bfs
            .shortest_distance(board, us.opposite())
            .unwrap_or(u8::MAX),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchPhase {
    /// Ply ≤ book window — book hints steer search ordering.
    Book,
    /// All other positions — negamax + adaptive LMR.
    Minimax,
}

pub fn search_phase(board: &Board) -> SearchPhase {
    if book::ply_number(board) <= BOOK_MAX_PLY {
        SearchPhase::Book
    } else {
        SearchPhase::Minimax
    }
}

/// Select the best move for the current position (CLI / web entry).
pub fn select_move(board: &mut Board, config: GenmoveConfig) -> Option<String> {
    let book_hint = book::book_hint(board);

    match config.engine {
        GenmoveEngine::Mcts | GenmoveEngine::Minimax => {
            let mut minimax_cfg = config.minimax;
            minimax_cfg.book_hint = book_hint;
            minimax_algebraic(board, minimax_cfg)
        }
        GenmoveEngine::Greedy => {
            let mut scratch = BfsScratch::new();
            choose_greedy_move(board, &mut scratch).map(format_move)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opening::book as opening;
    use crate::search::alphabeta::SearchConfig;
    use crate::search::genmove::{GenmoveConfig, GenmoveEngine};

    fn replay(moves: &[&str]) -> Board {
        let mut board = Board::new();
        for mv in moves {
            board.apply_algebraic(mv);
        }
        board
    }

    fn config(engine: GenmoveEngine) -> GenmoveConfig {
        GenmoveConfig {
            engine,
            mcts: Default::default(),
            minimax: SearchConfig {
                time_ms: 1,
                max_nodes: 1,
                log: false,
                book_hint: None,
                ..SearchConfig::default()
            },
        }
    }

    #[test]
    fn phase_book_at_start() {
        let board = Board::new();
        assert_eq!(search_phase(&board), SearchPhase::Book);
    }

    #[test]
    fn phase_minimax_after_book_window() {
        let board = replay(&["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "d3h", "f4"]);
        assert_eq!(opening::ply_number(&board), 11);
        assert_eq!(search_phase(&board), SearchPhase::Minimax);
    }

    #[test]
    fn walls_placed_counts_concrete_topology() {
        let mut board = Board::new();
        for _ in 0..4 {
            board.apply_algebraic("d2h");
            board.apply_algebraic("d8h");
        }
        assert!(walls_placed(&board) >= 4);
        // Few plies — still inside book window for ordering hints.
        assert_eq!(search_phase(&board), SearchPhase::Book);
    }

    #[test]
    fn hybrid_prefers_edge_wall_at_ply7() {
        let mut board = replay(&["e2", "e8", "e3", "e7", "e4", "e6"]);
        let hint = opening::book_hint(&mut board).expect("book hint");
        let mv = format_move(hint.mv);
        assert_eq!(mv, "e3h", "fair-10v10 mined ply-7 mainline");
    }

    #[test]
    fn pipeline_always_negamax() {
        let mut board = replay(&["e2", "e8", "e3", "e7", "e4", "e6"]);
        let mut cfg = config(GenmoveEngine::Minimax);
        cfg.minimax.time_ms = 50;
        cfg.minimax.max_nodes = 50_000;
        cfg.minimax.log = true;
        let mv = select_move(&mut board, cfg).expect("move");
        assert!(!mv.is_empty());
    }

    #[test]
    fn mcts_engine_routes_to_negamax() {
        let mut board = Board::new();
        let mut cfg = config(GenmoveEngine::Mcts);
        cfg.minimax.time_ms = 50;
        cfg.minimax.max_nodes = 50_000;
        assert!(select_move(&mut board, cfg).is_some());
    }
}
