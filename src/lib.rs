//! Titanium Engine — Quoridor search core.
//!
//! ```text
//! core/     board, zobrist
//! util/     grid, perft
//! movegen/  legal moves only
//! path/     BFS reachability
//! cat/      Corridor Attention Table v3 + pruning + viz
//! eval/     static evaluation (see search::alphabeta)
//! search/   αβ negamax, TT, pipeline, genmove
//! opening/  book
//! ```

pub mod ace;
pub mod cat;
pub mod core;
pub mod eval;
pub mod movegen;
pub mod opening;
pub mod oracle;
pub mod path;
pub mod search;
pub mod util;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(test)]
mod test_replay;

// ── Public API (stable re-exports) ───────────────────────────────────────────

pub use cat::{
    cat_snapshot_json, collect_search_moves, move_corridor_attention, wall_net_race,
    wall_should_search, CorridorAttention, CAT_COLD_CM, CAT_HOT_CM,
};
pub use core::board::{Board, Column, Move, Player, Row, Undo, WallOrientation};
pub use movegen::{
    generate_legal_moves, generate_legal_moves_into, generate_legal_moves_slice,
    generate_legal_moves_slice_mode, PawnGenMode, MAX_LEGAL_MOVES,
};
pub use opening::{ply_number, BOOK_MAX_PLY};
pub use path::{both_players_reach_goals, can_reach_goal, shortest_distance, BfsScratch};
pub use search::greedy::choose_greedy_move;
#[allow(deprecated)]
pub use search::lmr_viz::lmr_snapshot_json;
pub use search::session_stdio::run_session_stdio;
pub use search::uci::run_uci_stdio;
pub use search::{
    genmove_algebraic, run_search, search_best_move, search_mcts, search_phase, walls_placed,
    Engine, EngineLimits, GameSearchSession, GenmoveConfig, GenmoveEngine, MctsConfig, MctsReport,
    SearchConfig, SearchPhase, SearchReport, SharedState, ThreadBenchResult, TranspositionTable,
    WorkerContext, DEFAULT_MAX_NODES, DEFAULT_TIME_MS, MCTS_DEFAULT_MAX_SIMULATIONS,
    MCTS_DEFAULT_UCT,
};
pub use util::perft::{
    format_move, perft, perft_divide, perft_fast, perft_fast_ctx, perft_fast_mode,
    perft_fast_mode_ctx, perft_iterative, perft_naive,
    perft_no_tt_mode, PerftContext, PERFT3_STARTPOS, PERFT4_STARTPOS,
};
#[cfg(feature = "parallel")]
pub use util::perft::perft_parallel_root;
