//! Titanium move pipeline — book PV hints, phase routing, engine selection.
//!
//! `genmove` is only the public name for this; routing lives here so search and
//! movegen stay single-purpose.

use crate::core::board::Board;
use crate::opening::book::{self, BOOK_MAX_PLY};
use crate::path::BfsScratch;
use crate::search::alphabeta::genmove_algebraic as minimax_algebraic;
use crate::search::genmove::{GenmoveConfig, GenmoveEngine};
use crate::search::greedy::choose_greedy_move;
use crate::search::mcts::{genmove_algebraic as mcts_algebraic, MctsConfig};
use crate::util::perft::format_move;

/// Walls on board before minimax is preferred over MCTS (topology is concrete).
const BRIDGE_WALL_THRESHOLD: u8 = 4;
/// Last ply where CAT-MCTS bridge may run (after book window).
const BRIDGE_MAX_PLY: u32 = 20;
/// Once any wall is on the board, hybrid uses minimax — MCTS rollouts misread structure.
const OPENING_MINIMAX_WALLS: u8 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SearchPhase {
    /// Ply ≤ 10 — book hints steer search ordering.
    Book,
    /// Ply 11–20, open board — CAT-guided MCTS.
    Bridge,
    /// Ply > 20 or enough walls — minimax + corridor attention.
    Minimax,
}

pub fn walls_placed(board: &Board) -> u8 {
    20u8.saturating_sub(board.walls_remaining[0].saturating_add(board.walls_remaining[1]))
}

pub fn search_phase(board: &Board) -> SearchPhase {
    if walls_placed(board) >= BRIDGE_WALL_THRESHOLD {
        return SearchPhase::Minimax;
    }
    let ply = book::ply_number(board);
    if ply <= BOOK_MAX_PLY {
        SearchPhase::Book
    } else if std::env::var("TITANIUM_BRIDGE").is_ok_and(|v| v == "1") && ply <= BRIDGE_MAX_PLY {
        SearchPhase::Bridge
    } else {
        SearchPhase::Minimax
    }
}

fn use_bridge(board: &Board) -> bool {
    matches!(search_phase(board), SearchPhase::Bridge)
}

/// True when STM has a longer shortest-path race than the opponent.
fn losing_race(board: &Board, bfs: &mut BfsScratch) -> bool {
    let us = board.side();
    let our = bfs.shortest_distance(board, us).unwrap_or(u8::MAX);
    let opp = bfs.shortest_distance(board, us.opposite()).unwrap_or(u8::MAX);
    our > opp
}

fn log_phase(phase: SearchPhase) {
    if !std::env::var("TITANIUM_LOG").is_ok() {
        return;
    }
    let label = match phase {
        SearchPhase::Book => "book",
        SearchPhase::Bridge => "bridge",
        SearchPhase::Minimax => "minimax",
    };
    eprintln!("info phase {label}");
}

/// Select the best move for the current position (CLI / web entry).
pub fn select_move(board: &mut Board, config: GenmoveConfig) -> Option<String> {
    let phase = search_phase(board);
    log_phase(phase);

    // Book lines are PV hints for move ordering and aspiration bias — search always runs.
    let book_hint = book::book_hint(board);
    let mut race_bfs = BfsScratch::new();
    let behind_in_race = losing_race(board, &mut race_bfs);

    match config.engine {
        GenmoveEngine::Mcts => {
            let mut mcts_cfg = config.mcts;
            mcts_cfg.book_hint = book_hint;
            mcts_algebraic(board, mcts_cfg)
        }
        GenmoveEngine::Minimax => {
            let mut minimax_cfg = config.minimax;
            minimax_cfg.book_hint = book_hint;
            if phase == SearchPhase::Book {
                // MCTS rollouts sprint pawns; once losing the race, switch to AB for walls.
                if walls_placed(board) >= OPENING_MINIMAX_WALLS || behind_in_race {
                    minimax_algebraic(board, minimax_cfg)
                } else {
                    let mut opening = config.mcts;
                    opening.time_ms = config.minimax.time_ms;
                    opening.log = config.minimax.log;
                    opening.book_hint = book_hint;
                    mcts_algebraic(board, opening)
                }
            } else if use_bridge(board) {
                let bridge = MctsConfig {
                    time_ms: config.minimax.time_ms,
                    max_simulations: config.mcts.max_simulations,
                    uct: config.mcts.uct,
                    log: config.minimax.log,
                    use_cat_guidance: true,
                    book_hint,
                };
                mcts_algebraic(board, bridge)
            } else {
                minimax_algebraic(board, minimax_cfg)
            }
        }
        GenmoveEngine::Greedy => {
            let mut scratch = crate::path::BfsScratch::new();
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
    use crate::search::mcts::MctsConfig;

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
            mcts: MctsConfig {
                time_ms: 1,
                max_simulations: 1,
                log: false,
                ..MctsConfig::default()
            },
            minimax: SearchConfig {
                time_ms: 1,
                max_nodes: 1,
                log: false,
                book_hint: None,
            },
        }
    }

    #[test]
    fn phase_book_at_start() {
        let board = Board::new();
        assert_eq!(search_phase(&board), SearchPhase::Book);
    }

    #[test]
    fn phase_bridge_after_book_window() {
        let board = replay(&["e2", "e8", "e3", "e7", "e4", "e6", "e5", "e4", "d3h", "f4"]);
        assert_eq!(opening::ply_number(&board), 11);
        assert_eq!(search_phase(&board), SearchPhase::Minimax);
    }

    #[test]
    fn phase_minimax_when_walls_concrete() {
        let mut board = Board::new();
        for _ in 0..4 {
            board.apply_algebraic("d2h");
            board.apply_algebraic("d8h");
        }
        assert!(walls_placed(&board) >= 4);
        assert_eq!(search_phase(&board), SearchPhase::Minimax);
    }

    #[test]
    fn hybrid_prefers_edge_wall_at_ply7() {
        let mut board = replay(&["e2", "e8", "e3", "e7", "e4", "e6"]);
        let hint = opening::book_hint(&mut board).expect("book hint");
        let mv = format_move(hint.mv);
        assert!(
            matches!(mv.as_str(), "h3h" | "a3h"),
            "expected anti-Gorisanson edge wall PV, got {mv}"
        );
    }

    #[test]
    fn hybrid_blocks_when_losing_sprint_race() {
        let mut board = replay(&[
            "e2", "e8", "d2", "e7", "d3", "e6", "d4", "e5", "c4", "e4",
        ]);
        let cfg = GenmoveConfig {
            engine: GenmoveEngine::Minimax,
            mcts: MctsConfig {
                time_ms: 2000,
                max_simulations: 50_000,
                log: false,
                ..MctsConfig::default()
            },
            minimax: SearchConfig {
                time_ms: 2000,
                max_nodes: 500_000,
                log: false,
                book_hint: None,
            },
        };
        let mv = select_move(&mut board, cfg).expect("move");
        assert!(
            mv.ends_with('h') || mv.ends_with('v'),
            "expected a wall to slow black's e-file sprint, got {mv}"
        );
    }

    #[test]
    fn hybrid_opening_uses_mcts_not_minimax() {
        let mut board = replay(&["e2", "e8", "e3", "e7", "e4", "e6"]);
        let mut cfg = config(GenmoveEngine::Minimax);
        cfg.minimax.time_ms = 1;
        cfg.mcts.max_simulations = 1;
        assert!(select_move(&mut board, cfg).is_some());
    }
}
