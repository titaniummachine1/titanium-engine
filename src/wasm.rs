//! wasm-bindgen bindings for the website (GitHub Pages + static hosting).
//!
//! Build (from repo root):
//!   cd site/web && npm run build:wasm
//!
//! `WasmEngine` — warm titanium-v15 grafted session (NNUE + O1 movegen).
//! `WasmAceEngine` — one-shot ACE genmove for ACE tier sliders.

use wasm_bindgen::prelude::*;

use crate::acev13::{
    ace_to_algebraic, algebraic_to_ace, AceGame, AceParams, AceSearch, ThinkResult, ACE_NO_MOVE,
};

fn acev13_params_from_mode(engine_mode: &str, movetime_ms: u32, max_depth: i32) -> AceParams {
    let ti_movegen = engine_mode.contains("-ti") || engine_mode == "ace-v13";
    let eme = engine_mode.contains("pmc");
    AceParams {
        time_ms: (movetime_ms as u64).max(1),
        max_depth: if max_depth > 0 { max_depth } else { 30 },
        full: false,
        cat: false,
        ti_movegen,
        log: false,
        eme,
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

fn is_acev13_mode(engine_mode: &str) -> bool {
    engine_mode.starts_with("ace-v13") || engine_mode == "ace-v13"
}

fn build_acev13_search(g: AceGame, params: AceParams, engine_label: &str) -> AceSearch {
    let mut search = match engine_label {
        "titanium-v15" | "titanium-v14" | "ace-v13-grafted" => *AceSearch::grafted(g, None),
        "titanium-v15-frozen" => *AceSearch::grafted_frozen(g, None),
        "titanium-v15-no-raceproof" | "ace-v13-grafted-no-raceproof" => {
            *AceSearch::grafted_no_raceproof(g, None)
        }
        "ace-v13-ti-pure" => *AceSearch::with_ti_movegen_pure(g),
        _ if params.ti_movegen && params.cat => *AceSearch::with_ti_movegen_and_cat(g),
        _ if params.ti_movegen => *AceSearch::with_ti_movegen(g),
        _ if params.cat => *AceSearch::with_cat(g),
        _ => *AceSearch::new(g),
    };
    if params.eme {
        search.enable_eme();
    }
    search
}

fn acev13_genmove_with_progress(
    moves: &str,
    params: AceParams,
    engine_label: &str,
    on_progress: Option<js_sys::Function>,
) -> Option<(String, ThinkResult)> {
    let g = replay_moves(moves).ok()?;
    if g.winner() >= 0 {
        return None;
    }
    let mut search = build_acev13_search(g, params, engine_label);
    search.set_wasm_progress(on_progress.clone());
    let stream = on_progress.is_some();
    let result = search.think(
        params.time_ms,
        params.max_depth,
        params.full,
        stream,
        engine_label,
    );
    if result.mv == ACE_NO_MOVE {
        return None;
    }
    if result.mv == 0 && search.g.winner() >= 0 {
        return None;
    }
    Some((ace_to_algebraic(result.mv), result))
}

fn replay_moves(moves: &str) -> Result<AceGame, JsError> {
    let mut g = AceGame::new();
    for text in moves.split_whitespace().filter(|s| !s.is_empty()) {
        if g.winner() >= 0 {
            return Err(JsError::new(&format!(
                "illegal replay past terminal: {text}"
            )));
        }
        g.make_move(algebraic_to_ace(text));
    }
    Ok(g)
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
        if is_acev13_mode(engine_mode) {
            let params = acev13_params_from_mode(engine_mode, movetime_ms, max_depth);
            return match acev13_genmove_with_progress(moves, params, engine_mode, on_progress) {
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
    search: AceSearch,
    engine_label: String,
}

#[wasm_bindgen]
impl WasmEngine {
    /// `frozen = true` → pinned pre-train NNUE (`titanium-v15-frozen`).
    #[wasm_bindgen(constructor)]
    pub fn new(frozen: bool) -> WasmEngine {
        let g = AceGame::new();
        let (search, engine_label) = if frozen {
            (
                *AceSearch::grafted_frozen(g, None),
                "titanium-v15-frozen".to_string(),
            )
        } else {
            (*AceSearch::grafted(g, None), "titanium-v15".to_string())
        };
        WasmEngine {
            search,
            engine_label,
        }
    }

    /// Reset to startpos (clears TT/killers/history).
    pub fn reset(&mut self) {
        self.search.set_position(AceGame::new());
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
        self.search.apply_move(algebraic_to_ace(mv));
        true
    }

    /// Search; returns best move in algebraic notation, or "(none)".
    /// `max_nodes` is ignored — v15 uses wall-clock only (matches native session).
    /// `on_progress` — optional JS callback receiving `info json` payloads during search.
    pub fn go(
        &mut self,
        movetime_ms: u32,
        _max_nodes: u32,
        on_progress: Option<js_sys::Function>,
    ) -> String {
        self.search.set_wasm_progress(on_progress.clone());
        let stream = on_progress.is_some();
        if self.search.g.winner() >= 0 {
            return "(none)".to_string();
        }
        let result = self.search.think(
            (movetime_ms as u64).max(1),
            30,
            false,
            stream,
            &self.engine_label,
        );
        self.search.set_wasm_progress(None);
        if result.mv == ACE_NO_MOVE {
            "(none)".to_string()
        } else {
            ace_to_algebraic(result.mv)
        }
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
