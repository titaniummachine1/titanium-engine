//! Titanium Engine — Quoridor search core.
//!
//! ```text
//! core/     board, zobrist
//! util/     grid, perft
//! movegen/  legal moves only
//! path/     BFS reachability
//! cat/      Corridor Attention Table v3 + pruning + viz
//! eval/     static evaluation (see search::alphabeta)
//! search/   αβ, MCTS, TT, pipeline, genmove
//! opening/  book
//! ```

pub mod cat;
pub mod core;
pub mod eval;
pub mod movegen;
pub mod opening;
pub mod oracle;
pub mod path;
pub mod search;
pub mod util;

#[cfg(test)]
mod test_replay;

// ── Public API (stable re-exports) ───────────────────────────────────────────

pub use cat::{
    cat_snapshot_json, collect_search_moves, move_corridor_attention, wall_net_race,
    wall_should_search, CorridorAttention, CAT_COLD_CM, CAT_HOT_CM,
};
pub use core::board::{Board, Column, Move, Player, Row, Undo, WallOrientation};
pub use movegen::{
    generate_legal_moves, generate_legal_moves_into, generate_legal_moves_slice, MAX_LEGAL_MOVES,
};
pub use opening::{ply_number, BOOK_MAX_PLY};
pub use path::{both_players_reach_goals, can_reach_goal, shortest_distance, BfsScratch};
pub use search::greedy::choose_greedy_move;
pub use search::{
    genmove_algebraic, search_best_move, search_mcts, search_phase, walls_placed, Engine,
    GenmoveConfig, GenmoveEngine, MctsConfig, MctsReport, SearchConfig, SearchPhase, SearchReport,
    TranspositionTable, DEFAULT_MAX_NODES, DEFAULT_TIME_MS, EngineLimits, MCTS_DEFAULT_MAX_SIMULATIONS,
    MCTS_DEFAULT_UCT, SharedState, ThreadBenchResult, WorkerContext,
};
pub use util::perft::{
    format_move, perft, perft_divide, perft_fast, perft_fast_ctx, perft_iterative, perft_naive,
    perft_parallel_root, PerftContext, PERFT3_STARTPOS, PERFT4_STARTPOS,
};
