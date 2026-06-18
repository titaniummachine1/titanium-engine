//! Public `genmove` API — types and a one-line delegate to `pipeline`.
//!
//! Move generation is `moves`; pruning is `prune`; search is `search`; routing is `pipeline`.

use crate::core::board::Board;
use crate::search::alphabeta::SearchConfig;
use crate::search::deprecated::mcts::MctsConfig;

pub use crate::search::deprecated::mcts::DEFAULT_MAX_SIMULATIONS as MCTS_DEFAULT_MAX_SIMULATIONS;
pub use crate::search::deprecated::mcts::DEFAULT_UCT as MCTS_DEFAULT_UCT;
pub use crate::search::pipeline::{
    lmr_stage_inputs, search_phase, walls_placed, LmrStageInputs, SearchPhase,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenmoveEngine {
    #[deprecated(since = "0.2.0", note = "MCTS is inactive; routes silently to negamax")]
    Mcts,
    Minimax,
    Greedy,
}

impl Default for GenmoveEngine {
    fn default() -> Self {
        Self::Minimax
    }
}

#[derive(Debug, Clone)]
pub struct GenmoveConfig {
    pub engine: GenmoveEngine,
    #[allow(deprecated)]
    pub mcts: MctsConfig,
    pub minimax: SearchConfig,
}

impl Default for GenmoveConfig {
    fn default() -> Self {
        Self {
            engine: GenmoveEngine::Minimax,
            mcts: MctsConfig::default(),
            minimax: SearchConfig::default(),
        }
    }
}

/// CLI / web entry — delegates to `pipeline::select_move`.
pub fn genmove_algebraic(board: &mut Board, config: GenmoveConfig) -> Option<String> {
    crate::search::pipeline::select_move(board, config)
}
