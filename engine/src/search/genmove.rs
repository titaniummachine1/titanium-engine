//! Public `genmove` API — types and a one-line delegate to `pipeline`.
//!
//! Move generation is `moves`; pruning is `prune`; search is `search`; routing is `pipeline`.

use crate::core::board::Board;
use crate::search::mcts::{MctsConfig, DEFAULT_TIME_MS};
use crate::search::alphabeta::{SearchConfig, DEFAULT_MAX_NODES};

pub use crate::search::pipeline::{search_phase, walls_placed, SearchPhase};
pub use crate::search::mcts::DEFAULT_MAX_SIMULATIONS as MCTS_DEFAULT_MAX_SIMULATIONS;
pub use crate::search::mcts::DEFAULT_UCT as MCTS_DEFAULT_UCT;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenmoveEngine {
    Mcts,
    Minimax,
    Greedy,
}

impl Default for GenmoveEngine {
    fn default() -> Self {
        Self::Mcts
    }
}

#[derive(Debug, Clone)]
pub struct GenmoveConfig {
    pub engine: GenmoveEngine,
    pub mcts: MctsConfig,
    pub minimax: SearchConfig,
}

impl Default for GenmoveConfig {
    fn default() -> Self {
        Self {
            engine: GenmoveEngine::Mcts,
            mcts: MctsConfig::default(),
            minimax: SearchConfig {
                time_ms: DEFAULT_TIME_MS,
                max_nodes: DEFAULT_MAX_NODES,
                log: false,
                book_hint: None,
            },
        }
    }
}

/// CLI / web entry — delegates to `pipeline::select_move`.
pub fn genmove_algebraic(board: &mut Board, config: GenmoveConfig) -> Option<String> {
    crate::search::pipeline::select_move(board, config)
}
