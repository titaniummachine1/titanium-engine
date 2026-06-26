//! wasm-bindgen bindings for the website (GitHub Pages + static hosting).
//!
//! Build (from repo root):
//!   cd site/web && npm run build:wasm
//!
//! `WasmEngine` — warm titanium-v15 grafted session (NNUE + O1 movegen).
//! `WasmAceEngine` — one-shot ACE genmove for ACE tier sliders.

use wasm_bindgen::prelude::*;

use crate::cat::cat_snapshot_json;
use crate::core::board::Board;
use crate::titanium::search::{think_result_progress_json, SearchProfile};
use crate::titanium::{
    algebraic_to_move_id, build_search_for_engine_flag, move_id_to_algebraic, GameState,
    ThinkResult, TitaniumParams, TitaniumSearch, TITANIUM_NO_MOVE,
};

fn titanium_v15_params_from_mode(
    engine_mode: &str,
    movetime_ms: u32,
    max_depth: i32,
) -> TitaniumParams {
    let ti_movegen = engine_mode.contains("-ti") || engine_mode == "ace-v13";
    let eme = engine_mode.contains("pmc");
    TitaniumParams {
        time_ms: (movetime_ms as u64).max(1),
        max_depth: if max_depth > 0 { max_depth } else { 30 },
        max_nodes: 0,
        full: false,
        cat: false,
        ti_movegen,
        log: false,
        eme,
        threads: 1,
    }
}

fn ace_params_from_mode(
    engine_mode: &str,
    movetime_ms: u32,
    max_depth: i32,
) -> crate::ace::AceParams {
    let ti_movegen = engine_mode.contains("-ti");
    let eme = engine_mode.contains("pmc");
    crate::ace::AceParams {
        time_ms: (movetime_ms as u64).max(1),
        max_depth: if max_depth > 0 { max_depth } else { 30 },
        full: false,
        cat: false,
        ti_movegen,
        log: false,
        eme,
    }
}

fn is_titanium_v15_mode(engine_mode: &str) -> bool {
    engine_mode.starts_with("ace-v13") || engine_mode == "ace-v13"
}

fn build_titanium_search(
    g: GameState,
    params: TitaniumParams,
    engine_label: &str,
) -> TitaniumSearch {
    if engine_label.starts_with("ace-v13")
        || engine_label.starts_with("titanium-v15")
        || engine_label.starts_with("titanium-v14")
        || engine_label.contains("grafted")
    {
        return *build_search_for_engine_flag(engine_label, g);
    }
    let mut search = match engine_label {
        _ if params.ti_movegen && params.cat => *TitaniumSearch::with_ti_movegen_and_cat(g),
        _ if params.ti_movegen => *TitaniumSearch::with_ti_movegen(g),
        _ if params.cat => *TitaniumSearch::with_cat(g),
        _ => *TitaniumSearch::new(g),
    };
    if params.eme {
        search.enable_eme();
    }
    search
}

fn titanium_genmove_with_progress(
    moves: &str,
    params: TitaniumParams,
    engine_label: &str,
    on_progress: Option<js_sys::Function>,
) -> Option<(String, ThinkResult)> {
    let g = replay_moves(moves).ok()?;
    if g.winner() >= 0 {
        return None;
    }
    let mut search = build_titanium_search(g, params, engine_label);
    search.set_wasm_progress(on_progress.clone());
    let stream = on_progress.is_some();
    let result = search.think(
        params.time_ms,
        params.max_depth,
        params.max_nodes,
        params.full,
        stream,
        engine_label,
    );
    if result.mv == TITANIUM_NO_MOVE {
        return None;
    }
    if result.mv == 0 && search.g.winner() >= 0 {
        return None;
    }
    // Guarantee the browser receives final depth/nodes even if streaming throttled.
    if let Some(f) = on_progress.as_ref() {
        let json = think_result_progress_json(engine_label, &result);
        let _ = f.call1(&JsValue::NULL, &JsValue::from_str(&json));
    }
    Some((move_id_to_algebraic(result.mv), result))
}

fn replay_moves(moves: &str) -> Result<GameState, JsError> {
    let mut g = GameState::new();
    for text in moves.split_whitespace().filter(|s| !s.is_empty()) {
        if g.winner() >= 0 {
            return Err(JsError::new(&format!(
                "illegal replay past terminal: {text}"
            )));
        }
        g.make_move(algebraic_to_move_id(text));
    }
    Ok(g)
}

fn replay_board(moves: &str) -> Result<Board, JsError> {
    let mut board = Board::new();
    for text in moves.split_whitespace().filter(|s| !s.is_empty()) {
        if board.is_terminal().is_some() {
            return Err(JsError::new(&format!(
                "illegal replay past terminal: {text}"
            )));
        }
        board.apply_algebraic(text);
    }
    Ok(board)
}

/// Engine-backed CAT v3 snapshot for static/WASM hosting.
#[wasm_bindgen]
pub fn cat_snapshot(moves: &str) -> Result<String, JsError> {
    let mut board = replay_board(moves)?;
    Ok(cat_snapshot_json(&mut board))
}

/// ACE Rust port in WASM — one-shot genmove from a move list (GitHub Pages; no native binary).
#[wasm_bindgen]
pub struct WasmAceEngine;

#[wasm_bindgen]
impl WasmAceEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> WasmAceEngine {
        WasmAceEngine
    }

    /// Space-separated algebraic moves from startpos; returns best move or "(none)".
    /// `on_progress` — optional JS callback receiving `info json` during iterative deepening.
    pub fn genmove(
        &self,
        moves: &str,
        movetime_ms: u32,
        max_depth: i32,
        engine_mode: &str,
        on_progress: Option<js_sys::Function>,
    ) -> String {
        let list: Vec<String> = moves
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        if is_titanium_v15_mode(engine_mode) {
            let params = titanium_v15_params_from_mode(engine_mode, movetime_ms, max_depth);
            return match titanium_genmove_with_progress(moves, params, engine_mode, on_progress) {
                Some((alg, _)) => alg,
                None => "(none)".to_string(),
            };
        }
        let params = ace_params_from_mode(engine_mode, movetime_ms, max_depth);
        match crate::ace::ace_genmove(&list, params, engine_mode) {
            Some((alg, _)) => alg,
            None => "(none)".to_string(),
        }
    }
}

/// Warm titanium-v15 session — TT / history persist between plies (GitHub Pages Titanium).
#[wasm_bindgen]
pub struct WasmEngine {
    search: TitaniumSearch,
    engine_label: String,
    last_depth: i32,
    last_nodes: u64,
}

#[wasm_bindgen]
impl WasmEngine {
    /// `tier`: 0 = frozen (Easy), 1 = medium (previous live), 2 = hard (latest live).
    #[wasm_bindgen(constructor)]
    pub fn new(tier: u8) -> WasmEngine {
        let g = GameState::new();
        let (search, engine_label) = match tier {
            0 => (
                *TitaniumSearch::grafted_frozen(g, None),
                "titanium-v15-frozen".to_string(),
            ),
            1 => (
                *TitaniumSearch::grafted_medium(g, None),
                "titanium-v15-medium".to_string(),
            ),
            _ => (
                *TitaniumSearch::grafted(g, None),
                "titanium-v15".to_string(),
            ),
        };
        WasmEngine {
            search,
            engine_label,
            last_depth: 0,
            last_nodes: 0,
        }
    }

    /// Reset to startpos (clears TT/killers/history).
    pub fn reset(&mut self) {
        self.search.set_position(GameState::new());
    }

    /// Set position from startpos via space-separated algebraic moves.
    pub fn position(&mut self, moves: &str) -> Result<usize, JsError> {
        let g = replay_moves(moves)?;
        let n = moves.split_whitespace().filter(|s| !s.is_empty()).count();
        self.search.set_position(g);
        Ok(n)
    }

    /// Apply one algebraic move. Returns false if illegal/terminal.
    pub fn make_move(&mut self, mv: &str) -> bool {
        if self.search.g.winner() >= 0 {
            return false;
        }
        self.search.apply_move(algebraic_to_move_id(mv));
        true
    }

    /// Search; returns best move in algebraic notation, or "(none)".
    /// `max_nodes` is ignored — v15 uses wall-clock only (matches native session).
    /// `on_progress` — optional JS callback receiving `info json` payloads during search.
    pub fn go(
        &mut self,
        movetime_ms: u32,
        max_nodes: u32,
        on_progress: Option<js_sys::Function>,
    ) -> String {
        self.go_with_profile(movetime_ms, max_nodes, 0, 0, 0, on_progress)
    }

    /// Search with a per-worker `SearchProfile` (main vs helper lanes for multi-worker LazySMP).
    /// WASM cannot spawn OS threads; each Web Worker calls this with its own `WasmEngine` instance.
    pub fn go_with_profile(
        &mut self,
        movetime_ms: u32,
        max_nodes: u32,
        worker_id: u32,
        late_wall_skip_pct: u32,
        lmr_bias: i32,
        on_progress: Option<js_sys::Function>,
    ) -> String {
        let profile = SearchProfile {
            worker_id: worker_id as usize,
            late_wall_skip_pct: late_wall_skip_pct.min(100) as u8,
            lmr_bias,
        };
        self.search.set_search_profile(profile);
        self.search.set_wasm_progress(on_progress.clone());
        let stream = on_progress.is_some() && profile.worker_id == 0;
        if self.search.g.winner() >= 0 {
            self.last_depth = 0;
            self.last_nodes = 0;
            return "(none)".to_string();
        }
        let result = self.search.think(
            (movetime_ms as u64).max(1),
            30,
            max_nodes as u64,
            false,
            stream,
            &self.engine_label,
        );
        self.search.set_wasm_progress(None);
        self.last_depth = result.depth;
        self.last_nodes = result.nodes;
        if let Some(f) = on_progress.as_ref() {
            if profile.worker_id == 0 {
                let json = think_result_progress_json(&self.engine_label, &result);
                let _ = f.call1(&JsValue::NULL, &JsValue::from_str(&json));
            }
        }
        if result.mv == TITANIUM_NO_MOVE {
            "(none)".to_string()
        } else {
            move_id_to_algebraic(result.mv)
        }
    }

    /// Depth reached on the last `go` / `go_with_profile` call.
    pub fn last_search_depth(&self) -> i32 {
        self.last_depth
    }

    /// Nodes searched on the last `go` / `go_with_profile` call.
    pub fn last_search_nodes(&self) -> u64 {
        self.last_nodes
    }

    /// Space-separated legal moves (not used by the site worker; kept for API compat).
    pub fn legal_moves(&self) -> String {
        String::new()
    }

    /// -1 = ongoing; 0/1 = winner player index.
    pub fn winner(&self) -> i32 {
        let w = self.search.g.winner();
        if w < 0 {
            -1
        } else {
            w
        }
    }
}
