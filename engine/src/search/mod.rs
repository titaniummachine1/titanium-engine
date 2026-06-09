//! Search — αβ, MCTS, TT, pipeline, genmove entry.

pub mod alphabeta;
pub mod context;
pub mod genmove;
pub mod greedy;
pub mod mcts;
pub mod pipeline;
pub mod runtime;
pub mod tt;

pub use alphabeta::{
    search_best_move, SearchConfig, SearchReport, DEFAULT_MAX_NODES, DEFAULT_TIME_MS,
};
pub use context::{EngineLimits, SharedState, ThreadBenchResult, WorkerContext};
pub use genmove::{
    genmove_algebraic, GenmoveConfig, GenmoveEngine, MCTS_DEFAULT_MAX_SIMULATIONS, MCTS_DEFAULT_UCT,
};
pub use mcts::{search_mcts, MctsConfig, MctsReport};
pub use pipeline::{search_phase, walls_placed, SearchPhase};
pub use runtime::Engine;
pub use tt::TranspositionTable;
