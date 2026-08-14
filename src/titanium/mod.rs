//! **Titanium** — the production engine, and the only search in this crate.
//!
//! Version is `titanium-v{CARGO_PKG_VERSION}` (see `uci::session::ENGINE_VERSION`);
//! there is no second engine, no reference port, and no flag that selects one.
//! An engine flag this binary does not recognise is rejected rather than routed,
//! because a weaker search silently reachable by a stale flag has cost real Elo
//! here before.
//!
//! Titanium grafts the O1 pawn-LUT movegen, an adaptive TT, the win-certificate
//! solver (`certify.rs`), and an incremental HalfPW accumulator onto the search
//! core it originally inherited from the ACE v13 JavaScript engine. That
//! ancestry survives only as history: the ACE crate and its faithful-port
//! variants are deleted, and nothing in the tree is required to match their
//! behaviour. Comments elsewhere that pin behaviour to "what the JS did" are
//! constraints on a reference implementation that no longer exists — treat them
//! as suspect, not as invariants.
//!
//! Session entry point: `run_titanium_session_stdio` (warm TT, `go TIME_SEC`).
//!
//! ## Coordinate mapping (ACE row 0 = top, Titanium row 0 = bottom)
//!   pawn  m = (8 - row) * 9 + col
//!   wall  m = dense base + (7 - row) * 8 + col   (81 = h, 145 = v)

// Layered façades (architecture v1.0). Compatibility aliases keep `titanium::race`
// etc. stable — public API ≠ ownership.
pub mod endgame;
pub mod eval;
pub mod opening;
pub mod position;
pub mod research;
pub mod search;
pub mod timeman;
pub mod uci;

pub mod ka_policy;
pub mod lazy_seal;
pub mod perft;
pub mod policy;

// Compatibility module paths (ownership lives under façades above).
pub use endgame::cert_bridge;
pub use endgame::certify;
pub use endgame::exact_dp;
pub use endgame::race;
pub use eval::dist;
pub use eval::field_planes;
pub use eval::fields_viz;
pub use eval::nnue as net;
pub use opening::opening_book;
pub use opening::opening_book_embedded;
pub use position::board_bridge;
pub use position::dataset_state;
pub use position::game;
pub use position::packed_state;
pub use research::wall_ignore_cert;
pub use research::wall_ignore_cert_tests;
pub use research::wall_ignore_corridor;
pub use timeman::time_alloc;
pub use uci::session;

pub use game::GameState;
pub use packed_state::{
    ace_pawn_cell_to_python, decode_packed_state, pack_state, pack_state_dag,
    titanium_game_from_packed, FEATURE_SCHEMA, PACKED_STATE_LEN, POSITION_SCHEMA_VERSION,
};
pub use perft::{
    default_timeout, oracle_nodes, perft_titanium_native_timed,
    perft_titanium_ti_timed, perft_titanium_timed, TimedPerftResult, TITANIUM_PERFT4_STARTPOS,
};
pub use race::RaceOutcomeStats;
pub use search::{
    board_move_to_move_id, format_root_defense_diag_json, RootDefenseDiag, ThinkResult,
    TitaniumSearch,
};
pub use session::run_titanium_session_stdio;

/// Sentinel — pawn move id `0` is legal (cell a9); do not use `0` for "no move".
pub const TITANIUM_NO_MOVE: i16 = -1;
pub const MOVE_HW_BASE: i16 = 81;
pub const MOVE_VW_BASE: i16 = 145;
pub const MOVE_ID_MAX: i16 = 208;

#[inline]
pub fn is_pawn_move(m: i16) -> bool {
    (0..=80).contains(&m)
}

#[inline]
pub fn is_hwall_move(m: i16) -> bool {
    (81..=144).contains(&m)
}

#[inline]
pub fn is_vwall_move(m: i16) -> bool {
    (145..=208).contains(&m)
}

#[inline]
pub fn is_wall_move(m: i16) -> bool {
    (81..=208).contains(&m)
}

#[inline]
pub fn wall_slot(m: i16) -> usize {
    debug_assert!(is_wall_move(m));
    if is_hwall_move(m) {
        (m - MOVE_HW_BASE) as usize
    } else {
        (m - MOVE_VW_BASE) as usize
    }
}

use crate::core::board::{Move as BoardMove, WallOrientation};

/// Dense move encoding → Titanium board move (row flip between coordinate systems).
pub fn move_id_to_board(m: i16) -> BoardMove {
    if is_pawn_move(m) {
        BoardMove::Pawn {
            row: 8 - (m / 9) as u8,
            col: (m % 9) as u8,
        }
    } else {
        let orientation = if is_hwall_move(m) {
            WallOrientation::Horizontal
        } else {
            WallOrientation::Vertical
        };
        let slot = wall_slot(m) as i16;
        BoardMove::Wall {
            row: 7 - (slot / 8) as u8,
            col: (slot % 8) as u8,
            orientation,
        }
    }
}

/// Algebraic ("e2", "e3h") → dense move encoding.
pub fn algebraic_to_move_id(text: &str) -> i16 {
    let b = text.as_bytes();
    let col = (b[0] - b'a') as i16;
    let row = (b[1] - b'1') as i16;
    if b.len() > 2 {
        let slot = (7 - row) * 8 + col;
        match b[2] {
            b'h' => MOVE_HW_BASE + slot,
            b'v' => MOVE_VW_BASE + slot,
            _ => panic!("bad wall suffix in {text}"),
        }
    } else {
        (8 - row) * 9 + col
    }
}

/// Dense move encoding → algebraic.
pub fn move_id_to_algebraic(m: i16) -> String {
    if is_pawn_move(m) {
        let r = m / 9;
        let c = m % 9;
        format!("{}{}", (b'a' + c as u8) as char, 9 - r)
    } else {
        let (suffix, slot) = if is_hwall_move(m) {
            ('h', m - MOVE_HW_BASE)
        } else {
            ('v', m - MOVE_VW_BASE)
        };
        let r = slot / 8;
        let c = slot % 8;
        format!("{}{}{}", (b'a' + c as u8) as char, 8 - r, suffix)
    }
}

#[derive(Debug, Clone)]
pub struct TitaniumParams {
    pub time_ms: u64,
    pub max_depth: i32,
    pub threads: usize,
    /// Disable the easy-move early stop (search the full time budget).
    pub full: bool,
    /// Hybrid: CAT-filter wall moves at inner nodes.
    pub cat: bool,
    /// Titanium `movegen` on mirrored board (fast full-legal generation).
    pub ti_movegen: bool,
    /// Stream iterative-deepening progress on stderr (`info json`).
    pub log: bool,
    /// Top-N root lines to expose in progress JSON (`multiPv`).
    pub multipv: usize,
    /// Emit ranked `rootMoves` in progress JSON.
    pub root_scores: bool,
    /// Early Move Extensions on ordered wall moves (mirror of graduated LMR).
    pub eme: bool,
    /// Root opening book mode (`off` | `order` | `play`).
    pub book: crate::titanium::opening_book::OpeningBookMode,
    /// Optional path to opening DAG SQLite (default: training/data/opening_book/...).
    pub book_db: Option<String>,
}

impl Default for TitaniumParams {
    fn default() -> Self {
        Self {
            time_ms: 4000,
            max_depth: 128,
            threads: 1,
            full: false,
            cat: false,
            ti_movegen: false,
            log: false,
            multipv: 1,
            root_scores: true,
            eme: false,
            book: crate::titanium::opening_book::OpeningBookMode::Off,
            book_db: None,
        }
    }
}

/// CLI entry — plays `moves` (algebraic) from startpos, thinks, returns best move.
pub fn titanium_genmove(
    moves: &[String],
    params: TitaniumParams,
    engine_label: &str,
) -> Option<(String, ThinkResult)> {
    let mut g = GameState::new();
    for text in moves {
        g.make_move(algebraic_to_move_id(text));
    }
    if g.winner() >= 0 {
        return None;
    }
    // One engine. The label and the ti_movegen/cat/eme params are accepted for
    // wire compatibility but no longer select a configuration -- that two-layer
    // setup let the site, the bench and the match harness diverge.
    let mut search = TitaniumSearch::production(g, None);
    search.set_multipv(params.multipv as u32);
    search.set_root_scores(params.root_scores);
    #[cfg(not(target_arch = "wasm32"))]
    {
        use std::path::PathBuf;
        let db = params.book_db.as_deref().map(PathBuf::from);
        search.set_opening_book(params.book, db);
    }
    #[cfg(target_arch = "wasm32")]
    {
        search.set_opening_book(params.book, None);
    }
    #[cfg(not(target_arch = "wasm32"))]
    let result = search.think_with_threads(
        params.time_ms,
        params.max_depth,
        params.full,
        params.log,
        engine_label,
        params.threads,
    );
    #[cfg(target_arch = "wasm32")]
    let result = search.think(
        params.time_ms,
        params.max_depth,
        params.full,
        params.log,
        engine_label,
    );
    if result.mv == TITANIUM_NO_MOVE {
        return None;
    }
    if result.mv == 0 && search.g.winner() >= 0 {
        return None;
    }
    Some((move_id_to_algebraic(result.mv), result))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn move_translation_round_trips() {
        // pawn: e1 = our (0,4) = ACE cell 76
        assert_eq!(algebraic_to_move_id("e1"), 76);
        assert_eq!(move_id_to_algebraic(76), "e1");
        // pawn: e9 = our (8,4) = ACE cell 4
        assert_eq!(algebraic_to_move_id("e9"), 4);
        assert_eq!(move_id_to_algebraic(4), "e9");
        // wall: d8v = our wall (7,3) = dense vw slot 3
        assert_eq!(algebraic_to_move_id("d8v"), 148);
        assert_eq!(move_id_to_algebraic(148), "d8v");
        // wall: a1h = our wall (0,0) = dense hw slot 56
        assert_eq!(algebraic_to_move_id("a1h"), 137);
        assert_eq!(move_id_to_algebraic(137), "a1h");
    }

    #[test]
    fn startpos_has_pawn_and_wall_moves() {
        let mut g = GameState::new();
        let mut buf = [0i16; 160];
        let n = g.gen_pawn_moves(&mut buf, 0);
        assert_eq!(n, 3);
        let mut walls = 0;
        for slot in 0..64 {
            if g.wall_legal(0, slot) {
                walls += 1;
            }
            if g.wall_legal(1, slot) {
                walls += 1;
            }
        }
        assert_eq!(walls, 128);
    }

    #[test]
    fn a8_goal_pawn_encodes_as_zero_not_no_move() {
        assert_eq!(algebraic_to_move_id("a9"), 0);
        assert_eq!(move_id_to_algebraic(0), "a9");
        let moves: Vec<String> = "e2 e8 e3 e7 e4 e6 d3h d6h f3h f6h d5v h3v e4h h6h h1h e3v d4 c4h b3h f6 c4 g6 f1h g5 b4 h5 d1h h4 b5 b6h c5h h3 a5 g3 b7v f3 a6 g5v a7 b2v a8 f2"
            .split_whitespace()
            .map(String::from)
            .collect();
        let params = TitaniumParams {
            time_ms: 500,
            max_depth: 4,
            threads: 1,
            full: false,
            cat: false,
            ti_movegen: true,
            log: false,
            multipv: 1,
            root_scores: true,
            eme: false,
            book: crate::titanium::opening_book::OpeningBookMode::Off,
            book_db: None,
        };
        let (alg, result) = titanium_genmove(&moves, params, "ace-v13-ti").expect("best move");
        assert_eq!(alg, "a9");
        assert_eq!(result.mv, 0);
        assert_ne!(result.mv, TITANIUM_NO_MOVE);
    }

    #[test]
    fn h6h_legal_after_a2h_line() {
        use crate::core::board::Board;
        use crate::movegen::generate_legal_moves;
        use crate::util::perft::format_move;

        let moves = [
            "e2", "e8", "e3", "e7", "e4", "e6", "d3h", "d6h", "f3h", "f6h", "b3h", "b6h", "h3h",
            "d4v", "a2h",
        ];
        let mut g = GameState::new();
        let mut board = Board::new();
        for m in moves {
            g.make_move(algebraic_to_move_id(m));
            board.apply_algebraic(m);
        }
        let slot = crate::titanium::wall_slot(algebraic_to_move_id("h6h"));
        assert!(
            g.wall_legal(0, slot),
            "ACE must accept h6h (off-topology fast path)"
        );
        let ti_legal: Vec<_> = generate_legal_moves(&board)
            .iter()
            .map(|mv| format_move(*mv))
            .collect();
        assert!(
            ti_legal.iter().any(|m| m == "h6h"),
            "Titanium oracle must accept h6h after onB edge fix"
        );
    }

    #[test]
    fn a6h_path_parity_after_h3v_line() {
        use crate::core::board::Board;
        use crate::core::board::WallOrientation;
        use crate::movegen::generate_legal_moves;
        use crate::movegen::legal::can_wall_block_topology;
        use crate::util::perft::format_move;

        let moves = [
            "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e6h", "c3h", "c6h", "g3h", "g6h", "a3h",
            "e4v", "h3v",
        ];
        let mut g = GameState::new();
        let mut board = Board::new();
        for m in moves {
            g.make_move(algebraic_to_move_id(m));
            board.apply_algebraic(m);
        }
        let slot = crate::titanium::wall_slot(algebraic_to_move_id("a6h"));
        let row = 7 - (slot / 8) as u8;
        let col = (slot % 8) as u8;
        let ti_legal: Vec<_> = generate_legal_moves(&board)
            .iter()
            .map(|mv| format_move(*mv))
            .collect();
        let ace_ok = g.wall_legal(0, slot);
        let can_block = can_wall_block_topology(&board, row, col, WallOrientation::Horizontal);
        // a6h keeps both goal paths open here (naive BFS confirms); the old
        // rejection was V10's partial-component false negative. ACE and the
        // Titanium oracle must agree on acceptance.
        assert!(can_block, "a6h touches topology — path flood must run");
        assert!(ace_ok, "ACE must accept a6h when both goal paths survive");
        assert!(
            ti_legal.iter().any(|m| m == "a6h"),
            "Titanium oracle must accept a6h on h3v line"
        );
    }

}
