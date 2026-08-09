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
pub mod movegen;
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
pub use pathfinding::{both_players_reach_goals, can_reach_goal, shortest_distance, BfsScratch};
#[cfg(not(target_arch = "wasm32"))]
pub use util::perft::perft_fast_timed;
#[cfg(feature = "parallel")]
pub use util::perft::perft_parallel_root;
pub use util::perft_engine::{Engine, EngineLimits, ThreadBenchResult};
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
pub use titanium::{
    algebraic_to_move_id, board_move_to_move_id, decode_packed_state, move_id_to_algebraic,
    move_id_to_board, pack_state, run_titanium_session_stdio,
    titanium_game_from_packed,
    titanium_genmove, GameState, TitaniumParams, TitaniumSearch,
    FEATURE_SCHEMA, PACKED_STATE_LEN, POSITION_SCHEMA_VERSION, TITANIUM_NO_MOVE,
};
