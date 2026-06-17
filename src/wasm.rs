//! wasm-bindgen bindings for the website (GitHub Pages + static hosting).
//!
//! Build (from repo root):
//!   cd site/web && npm run build:wasm
//!
//! `WasmEngine` — warm titanium-v15 grafted session (NNUE + O1 movegen).
//! `WasmAceEngine` — one-shot ACE genmove for ACE tier sliders.

use wasm_bindgen::prelude::*;

use crate::acev13::{
    ace_genmove, ace_to_algebraic, algebraic_to_ace, AceGame, AceParams, AceSearch, ACE_NO_MOVE,
};

fn acev13_params_from_mode(
    engine_mode: &str,
    movetime_ms: u32,
    max_depth: i32,
) -> AceParams {
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

fn ace_params_from_mode(engine_mode: &str, movetime_ms: u32, max_depth: i32) -> crate::ace::AceParams {
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

fn replay_moves(moves: &str) -> Result<AceGame, JsError> {
    let mut g = AceGame::new();
    for text in moves.split_whitespace().filter(|s| !s.is_empty()) {
        if g.winner() >= 0 {
            return Err(JsError::new(&format!("illegal replay past terminal: {text}")));
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
    pub fn genmove(
        &self,
        moves: &str,
        movetime_ms: u32,
        max_depth: i32,
        engine_mode: &str,
    ) -> String {
        let list: Vec<String> = moves
            .split_whitespace()
            .filter(|s| !s.is_empty())
            .map(|s| s.to_string())
            .collect();
        let result = if is_acev13_mode(engine_mode) {
            let params = acev13_params_from_mode(engine_mode, movetime_ms, max_depth);
            ace_genmove(&list, params, engine_mode).map(|(alg, _)| alg)
        } else {
            let params = ace_params_from_mode(engine_mode, movetime_ms, max_depth);
            crate::ace::ace_genmove(&list, params, engine_mode).map(|(alg, _)| alg)
        };
        match result {
            Some(alg) => alg,
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
            (
                *AceSearch::grafted(g, None),
                "titanium-v15".to_string(),
            )
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
        if w < 0 { -1 } else { w }
    }
}
