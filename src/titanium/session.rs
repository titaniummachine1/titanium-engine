//! ACE v13 reference-engine session REPL (ace-v13-*, ace-v13-ti-pure, …).
//!
//! Wire protocol: `reset` / `position [MOVES]` / `makemove MOVE` /
//! `go TIME_SEC` / `quit`.  Holds one warm `TitaniumSearch` per process so the
//! TT, killers, history, and countermove tables persist between plies.
//!
//! `titanium-v15` uses this session (grafted build).  `session_v15` infinite
//! search exists but is not routed — see main.rs.

use std::io::{self, BufRead, Write};

use super::{algebraic_to_move_id, move_id_to_algebraic, GameState, TitaniumSearch};

fn reply_ready(stdout: &mut io::Stdout) {
    let _ = writeln!(stdout, "ready");
    let _ = stdout.flush();
}

fn reply_error(stdout: &mut io::Stdout, message: &str) {
    let _ = writeln!(stdout, "error {}", message);
    let _ = stdout.flush();
}

fn is_v16_graft(engine_flag: &str) -> bool {
    matches!(
        engine_flag,
        "titanium-v16"
            | "titanium-v16-sfhist"
            | "titanium-v17"
            | "titanium-v17-cat-path-lmr"
            | "titanium-v17-route-touch"
            | "titanium-v17-qsearch"
            | "titanium-v17-route-touch-qsearch"
            | "titanium-v17-lazy-topn"
            | "titanium-v17-rfp-ace"
            | "titanium-v17-lmp-ace"
            | "titanium-v17-no-partial-iter"
            | "titanium-v17-no-predict-stop"
            | "titanium-v17-no-partial-no-predict"
    )
}

fn configure_session_experiments(search: &mut TitaniumSearch, engine_flag: &str) {
    let is_v17 = matches!(
        engine_flag,
        "titanium-v17"
            | "titanium-v17-lmp-ace"
            | "titanium-v17-no-partial-iter"
            | "titanium-v17-no-predict-stop"
            | "titanium-v17-no-partial-no-predict"
    );
    let enable_sf_history = matches!(
        engine_flag,
        "titanium-v16-sfhist"
            | "titanium-v17"
            | "titanium-v17-cat-path-lmr"
            | "titanium-v17-route-touch"
            | "titanium-v17-qsearch"
            | "titanium-v17-route-touch-qsearch"
            | "titanium-v17-lazy-topn"
            | "titanium-v17-rfp-ace"
            | "titanium-v17-lmp-ace"
    ) || is_v17;
    if enable_sf_history {
        search.set_sf_history(true);
    }
    if engine_flag.contains("route-touch") {
        search.enable_route_touch_ordering();
    }
    if engine_flag == "titanium-v17"
        || engine_flag == "titanium-v17-cat-path-lmr"
        || engine_flag.contains("qsearch")
        || engine_flag == "titanium-v17-lazy-topn"
        || engine_flag == "titanium-v17-rfp-ace"
        || engine_flag == "titanium-v17-lmp-ace"
        || is_v17
    {
        search.enable_q_search();
    }
    if engine_flag == "titanium-v17"
        || engine_flag == "titanium-v17-cat-path-lmr"
        || engine_flag == "titanium-v17-lazy-topn"
        || engine_flag == "titanium-v17-rfp-ace"
        || engine_flag == "titanium-v17-lmp-ace"
        || is_v17
    {
        search.enable_cat_path_lmr();
    }
    if engine_flag == "titanium-v17"
        || engine_flag == "titanium-v17-lazy-topn"
        || engine_flag == "titanium-v17-rfp-ace"
        || engine_flag == "titanium-v17-lmp-ace"
    {
        search.enable_lazy_topn();
    }
    if engine_flag == "titanium-v17" || engine_flag == "titanium-v17-lmp-ace" {
        search.set_ace_lmp(true);
    }
    if engine_flag == "titanium-v17-rfp-ace" {
        search.set_ace_rfp(true);
    }
    if engine_flag == "titanium-v17-no-partial-iter"
        || engine_flag == "titanium-v17-no-partial-no-predict"
    {
        search.set_partial_iter(false);
    }
    if engine_flag == "titanium-v17-no-predict-stop"
        || engine_flag == "titanium-v17-no-partial-no-predict"
    {
        search.set_predict_stop(false);
    }
    if engine_flag.contains("pmc") {
        search.enable_eme();
    }
}

fn build_search(engine_flag: &str, g: GameState) -> Box<TitaniumSearch> {
    // titanium-v15 = production grafted build. ace-v13-ti-pure = JS baseline yardstick.
    let mut search = match engine_flag {
        "ace-v13-pure" => TitaniumSearch::new(g),
        "ace-v13-ti-pure" => TitaniumSearch::with_ti_movegen_pure(g),
        "titanium-v15-medium" => TitaniumSearch::grafted_medium(g, None),
        "titanium-v15-frozen" => TitaniumSearch::grafted_frozen(g, None),
        flag if is_v16_graft(flag) => TitaniumSearch::grafted_v16(g, None),
        "titanium-v15-no-raceproof" | "ace-v13-grafted-no-raceproof" => {
            TitaniumSearch::grafted_no_raceproof(g, None)
        }
        "ace-v13-grafted" | "titanium-v14" | "titanium-v15" => TitaniumSearch::grafted(g, None),
        _ => TitaniumSearch::with_ti_movegen(g),
    };
    configure_session_experiments(&mut search, engine_flag);
    search
}

fn replay(moves: &[String]) -> Result<GameState, String> {
    let mut g = GameState::new();
    for text in moves {
        if g.winner() >= 0 {
            return Err(format!("move {text} past terminal position"));
        }
        g.make_move(algebraic_to_move_id(text));
    }
    Ok(g)
}

/// Blocking REPL holding one warm `TitaniumSearch` for the process lifetime.
pub fn run_titanium_session_stdio(engine_flag: &str, threads: usize) {
    let mut search = build_search(engine_flag, GameState::new());
    let mut applied: Vec<String> = Vec::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(e) => {
                reply_error(&mut stdout, &e.to_string());
                break;
            }
        };
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        match parts[0] {
            "reset" => {
                search.set_position(GameState::new());
                applied.clear();
                reply_ready(&mut stdout);
            }
            "position" => {
                let moves: Vec<String> = parts[1..].iter().map(|s| (*s).to_string()).collect();
                let extends = !applied.is_empty()
                    && moves.len() >= applied.len()
                    && moves.iter().zip(applied.iter()).all(|(a, b)| a == b);
                if extends {
                    // common case: game advanced — push only the new plies,
                    // the search state stays fully warm.
                    let mut err = None;
                    for text in &moves[applied.len()..] {
                        if search.g.winner() >= 0 {
                            err = Some(format!("move {text} past terminal position"));
                            break;
                        }
                        search.apply_move(algebraic_to_move_id(text));
                    }
                    if let Some(msg) = err {
                        reply_error(&mut stdout, &msg);
                        continue;
                    }
                } else {
                    // undo / divergence — rebuild the board, keep the TT.
                    match replay(&moves) {
                        Ok(g) => search.set_position(g),
                        Err(msg) => {
                            reply_error(&mut stdout, &msg);
                            continue;
                        }
                    }
                }
                applied = moves;
                let _ = writeln!(stdout, "ready {}", applied.len());
                let _ = stdout.flush();
            }
            "makemove" => {
                let Some(mv) = parts.get(1) else {
                    reply_error(&mut stdout, "makemove requires a move");
                    continue;
                };
                if search.g.winner() >= 0 {
                    reply_error(&mut stdout, "terminal position");
                    continue;
                }
                search.apply_move(algebraic_to_move_id(mv));
                applied.push((*mv).to_string());
                reply_ready(&mut stdout);
            }
            "go" => {
                if search.g.winner() >= 0 {
                    reply_error(&mut stdout, "terminal position");
                    continue;
                }
                let time_sec: f64 = parts.get(1).and_then(|s| s.parse().ok()).unwrap_or(4.0);
                let time_ms = (time_sec * 1000.0).max(1.0) as u64;
                #[cfg(not(target_arch = "wasm32"))]
                let result =
                    search.think_with_threads(time_ms, 128, false, true, engine_flag, threads);
                #[cfg(target_arch = "wasm32")]
                let result = search.think(time_ms, 128, false, true, engine_flag);
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let helper_nodes = result
                        .helper_nodes
                        .iter()
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    let helper_depths = result
                        .helper_completed_depths
                        .iter()
                        .map(|d| d.to_string())
                        .collect::<Vec<_>>()
                        .join(",");
                    let _ = writeln!(
                        stdout,
                        "info json {{\"engine\":\"{}\",\"stoppedBy\":\"{}\",\"searchDepth\":{},\"nodes\":{},\"mainThreadNodes\":{},\"helperNodes\":[{}],\"totalNodes\":{},\"mainCompletedDepth\":{},\"helperCompletedDepths\":[{}]}}",
                        engine_flag,
                        result.stop_reason,
                        result.depth,
                        result.nodes,
                        result.main_thread_nodes,
                        helper_nodes,
                        result.total_nodes,
                        result.main_completed_depth,
                        helper_depths,
                    );
                    let _ = stdout.flush();
                }
                if result.mv == super::TITANIUM_NO_MOVE {
                    let _ = writeln!(stdout, "bestmove (none)");
                } else {
                    let _ = writeln!(stdout, "bestmove {}", move_id_to_algebraic(result.mv));
                }
                let _ = stdout.flush();
            }
            "quit" => break,
            _ => reply_error(&mut stdout, "unknown command"),
        }
    }
}

#[cfg(test)]
mod session_tests {
    use super::*;
    use crate::titanium::game::GameState;

    #[test]
    fn v17_route_touch_session_enables_experiments() {
        let search = build_search("titanium-v17-route-touch", GameState::new());
        assert!(search.route_touch_ordering_enabled());
        assert!(!search.q_search_enabled());
    }

    #[test]
    fn v17_session_enables_qsearch_without_route_touch() {
        let search = build_search("titanium-v17", GameState::new());
        assert!(search.q_search_enabled());
        assert!(!search.route_touch_ordering_enabled());
    }

    #[test]
    fn v17_qsearch_session_enables_experiments() {
        let search = build_search("titanium-v17-qsearch", GameState::new());
        assert!(search.q_search_enabled());
        assert!(!search.route_touch_ordering_enabled());
    }

    #[test]
    fn v17_cat_path_lmr_inherits_v17_and_only_enables_path_flag() {
        let search = build_search("titanium-v17-cat-path-lmr", GameState::new());
        assert!(search.q_search_enabled());
        assert!(search.cat_path_lmr_enabled());
        assert!(!search.route_touch_ordering_enabled());
        assert!(search.sf_history_enabled());
    }

    #[test]
    fn default_v17_enables_cat_path_lmr() {
        let search = build_search("titanium-v17", GameState::new());
        assert!(search.cat_path_lmr_enabled());
    }

    #[test]
    fn v17_lazy_topn_inherits_v17_and_only_enables_lazy_topn() {
        let search = build_search("titanium-v17-lazy-topn", GameState::new());
        assert!(search.sf_history_enabled());
        assert!(search.q_search_enabled());
        assert!(search.cat_path_lmr_enabled());
        assert!(search.lazy_topn_enabled());
        assert!(!search.route_touch_ordering_enabled());
    }

    #[test]
    fn default_v17_enables_lazy_topn() {
        let search = build_search("titanium-v17", GameState::new());
        assert!(search.lazy_topn_enabled());
    }

    #[test]
    fn v17_defaults_to_ace_lmp_and_compatibility_label_matches() {
        let candidate = build_search("titanium-v17-lmp-ace", GameState::new());
        let default = build_search("titanium-v17", GameState::new());
        assert!(candidate.sf_history_enabled());
        assert!(candidate.q_search_enabled());
        assert!(candidate.cat_path_lmr_enabled());
        assert!(candidate.lazy_topn_enabled());
        assert!(candidate.ace_lmp_enabled());
        assert!(default.ace_lmp_enabled());
    }

    #[test]
    fn v17_no_partial_iter_disables_only_partial_iteration() {
        let candidate = build_search("titanium-v17-no-partial-iter", GameState::new());
        let default = build_search("titanium-v17", GameState::new());
        assert!(!candidate.partial_iter_enabled());
        assert!(candidate.predict_stop_enabled());
        assert!(default.partial_iter_enabled());
        assert!(default.predict_stop_enabled());
    }

    #[test]
    fn v17_no_predict_stop_disables_only_predictive_stop() {
        let candidate = build_search("titanium-v17-no-predict-stop", GameState::new());
        assert!(candidate.partial_iter_enabled());
        assert!(!candidate.predict_stop_enabled());
    }

    #[test]
    fn v17_no_partial_no_predict_disables_both_controls() {
        let candidate = build_search("titanium-v17-no-partial-no-predict", GameState::new());
        assert!(!candidate.partial_iter_enabled());
        assert!(!candidate.predict_stop_enabled());
    }

    #[test]
    fn v17_rfp_ace_inherits_v17_and_only_changes_rfp() {
        let candidate = build_search("titanium-v17-rfp-ace", GameState::new());
        let default = build_search("titanium-v17", GameState::new());
        assert!(candidate.sf_history_enabled());
        assert!(candidate.q_search_enabled());
        assert!(candidate.cat_path_lmr_enabled());
        assert!(candidate.lazy_topn_enabled());
        assert!(candidate.ace_rfp_enabled());
        assert!(!default.ace_rfp_enabled());
    }
}
