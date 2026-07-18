//! Titanium Engine — Quoridor search core.
//!
//! Architecture v1.0 (see `docs/architecture.md`):
//! ```text
//! Live under src/:
//!   core/ movegen/ pathfinding/ cat/   Layer 0 — infrastructure
//!   titanium/position/                 Layer 1
//!   titanium/{eval,endgame}/           Layer 2
//!   titanium/search/                   Layer 3 — play search
//!   titanium/uci/ + validation/ + weights/  Layer 4 + assets
//!
//! Historical (not under src/):
//!   engine/legacy/{search,opening}/    αβ/CLI + crate-root opening book
//! ```
//! Training lives at repo-root `training/` — outside this crate.
//! Do not put new play-engine code under `engine/legacy/`.

pub mod bench_instr;
pub mod cat;
pub mod core;
#[path = "../legacy/search/mod.rs"]
pub mod legacy_search;
pub mod movegen;
#[path = "../legacy/opening/mod.rs"]
pub mod opening;
pub mod pathfinding;
pub mod titanium;
pub mod util;
pub mod validation;

#[cfg(feature = "wasm")]
pub mod wasm;

#[cfg(all(feature = "wasm-threads", target_arch = "wasm32"))]
pub use wasm_bindgen_rayon::init_thread_pool;

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
pub use pathfinding::{both_players_reach_goals, can_reach_goal, shortest_distance, BfsScratch};
pub use legacy_search::greedy::choose_greedy_move;
#[allow(deprecated)]
pub use legacy_search::lmr_viz::lmr_snapshot_json;
pub use legacy_search::session_stdio::run_session_stdio;
pub use legacy_search::uci::run_uci_stdio;
pub use legacy_search::{
    genmove_algebraic, run_search, search_best_move, search_mcts, search_phase, walls_placed,
    Engine, EngineLimits, GameSearchSession, GenmoveConfig, GenmoveEngine, MctsConfig, MctsReport,
    SearchConfig, SearchPhase, SearchReport, SharedState, ThreadBenchResult, TranspositionTable,
    WorkerContext, DEFAULT_MAX_NODES, DEFAULT_TIME_MS, MCTS_DEFAULT_MAX_SIMULATIONS,
    MCTS_DEFAULT_UCT,
};
#[cfg(not(target_arch = "wasm32"))]
pub use util::perft::perft_fast_timed;
#[cfg(feature = "parallel")]
pub use util::perft::perft_parallel_root;
pub use util::perft::{
    format_move, perft, perft_divide, perft_fast, perft_fast_anchor_baseline, perft_fast_ctx,
    perft_fast_mode, perft_fast_mode_ctx, perft_iterative, perft_naive,
    perft_no_tt_anchor_baseline, perft_no_tt_mode, perft_pawn_only_mode, PerftContext,
    PERFT3_STARTPOS, PERFT4_STARTPOS, PERFT5_STARTPOS, PERFT5_TIMEOUT_SECS,
};

// Re-export for sibling engines (e.g. `engines/ace`) that need ACE-row goal bits.
pub use titanium::dist;

// Titanium v15 production API (formerly `acev13` module path).
pub use titanium::fields_viz;
#[cfg(not(target_arch = "wasm32"))]
pub use titanium::opening_book;
#[cfg(not(target_arch = "wasm32"))]
pub use titanium::reduction_shadow_probe;
pub use titanium::{
    algebraic_to_move_id, board_move_to_move_id, decode_packed_state, move_id_to_algebraic,
    move_id_to_board, pack_state, reduction_counterfactual_probe, run_titanium_session_stdio,
    titanium_game_from_packed, titanium_genmove, GameState, TitaniumParams, TitaniumSearch,
    FEATURE_SCHEMA, PACKED_STATE_LEN, POSITION_SCHEMA_VERSION, TITANIUM_NO_MOVE,
};
