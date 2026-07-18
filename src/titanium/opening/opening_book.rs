//! Root opening book backed by the non-Titanium DAG SQLite database.

#[cfg(target_arch = "wasm32")]
use std::path::Path;
#[cfg(not(target_arch = "wasm32"))]
use std::path::{Path, PathBuf};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::Mutex;

#[cfg(not(target_arch = "wasm32"))]
use rusqlite::{Connection, OpenFlags};

use crate::titanium::dataset_state::DatasetState;
use crate::titanium::game::GameState;
use crate::titanium::opening_book_embedded::embedded_opening_book;
use crate::titanium::packed_state::pack_state_dag;
use crate::titanium::{algebraic_to_move_id, move_id_to_algebraic};

/// Opening-book horizon (order + play): stay in book the full 15 plies.
pub const OPENING_BOOK_MAX_PLIES: usize = 15;
/// Hard ceiling — book is never consulted at or past this ply.
pub const OPENING_BOOK_EXTENDED_MAX_PLIES: usize = 15;
/// Minimum raw win rate (decided games) to keep book active past [`OPENING_BOOK_MAX_PLIES`].
pub const OPENING_BOOK_EXTENDED_MIN_WIN_RATE: f64 = 0.55;
pub const PLAY_MIN_VISITS: u32 = 12;
pub const PLAY_MIN_SHARE: f64 = 0.60;
pub const PLAY_WILSON_GAP: f64 = 0.02;

/// Plies 1–6: sacred center trunk (`e2 e8 e3 e7 e4 e6`) — always forced, never
/// overridden. e4/e6 is the shared prefix of every mined book line in this file
/// (Ishtar lines, DAG denials, all ply-7 continuations below) — it was already
/// the de facto mainline, just not force-extended past ply 4 before.
pub const OPENING_SACRED_MAX_PLY: usize = 6;
/// Sacred plies only — plies 7+ use search (book ordering bias, never forced).
pub const OPENING_FORCE_MAX_PLY: usize = 6;
/// Root move-order bonus scale: win-rate fraction × this (+ Ishtar tier on top).
pub const BOOK_ATTENTION_WINRATE_SCALE: i32 = 1000;
pub const BOOK_ATTENTION_ISHTAR_BONUS: i32 = 1000;

/// Highest win-rate main line (non-Titanium DAG) — do not change.
const SACRED_CENTER_LINE: &[&str] = &["e2", "e8", "e3", "e7", "e4", "e6"];

/// Hard-forced Black (Ishtar) replies mined from real Ka-vs-Ishtar games run via
/// `site/ka_vs_ishtar_match.js` on 2026-07-03. Only Black's moves are forced —
/// White's moves in these games were Ka's, and Ka LOST (Ka-short vs
/// Ishtar-short, 3/3 games) or is presumed losing (Ka-intuition vs
/// Ishtar-medium, game capped at ply 12 before decision) in every source
/// game. Forcing White's side too would make our own White-side play
/// deliberately repeat the losing side's moves — so only the proven/assumed
/// winning (Black) replies are forced here; White's moves stay free search.
///
/// Line A (a3h -> a6h): Ka-short vs Ishtar-short, replicated identically in
/// 3/3 games, Ishtar (Black) won all 3.
/// Line B (a3h -> e3v): Ka-intuition vs Ishtar-medium, 1 game, undecided at
/// the ply-12 cutoff but Ishtar-medium is vastly stronger than Ka-intuition.
/// Both share ply 7 (White's a3h) as context; only one of a6h/e3v is ever
/// actually forced per game since ply 8 is our own choice when we're Black —
/// a6h is primary (proven wins); e3v's continuation is kept for completeness
/// in case ply 8 is ever reached via e3v through some other path.
const MINED_BLACK_MAINLINE: &[(&[&str], &str)] = &[
    // Line A: e2 e8 e3 e7 e4 e6 a3h a6h e3h c3v e5 e6h
    (&["e2", "e8", "e3", "e7", "e4", "e6", "a3h"], "a6h"),
    (
        &["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "a6h", "e3h"],
        "c3v",
    ),
    (
        &[
            "e2", "e8", "e3", "e7", "e4", "e6", "a3h", "a6h", "e3h", "c3v", "e5",
        ],
        "e6h",
    ),
    // Line B: e2 e8 e3 e7 e4 e6 a3h e3v e5 e4 d3h d4h
    (
        &["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "e3v", "e5"],
        "e4",
    ),
    (
        &[
            "e2", "e8", "e3", "e7", "e4", "e6", "a3h", "e3v", "e5", "e4", "d3h",
        ],
        "d4h",
    ),
];

/// Force a mined Black reply when the exact prefix matches (see
/// `MINED_BLACK_MAINLINE`). Only ever fires on Black's own ply (even
/// `hist_len`), consistent with only forcing the proven/assumed-winning side.
pub fn mined_black_mainline_direct_play(g: &GameState, legal_moves: &[i16]) -> Option<i16> {
    let played = history_algebraic(g);
    for (prefix, reply) in MINED_BLACK_MAINLINE {
        if played.len() == prefix.len() && played.iter().zip(prefix.iter()).all(|(a, b)| a == b) {
            let mv = algebraic_to_move_id(reply);
            if legal_moves.contains(&mv) {
                return Some(mv);
            }
        }
    }
    None
}

/// Hand-mined Ishtar answers — extra search attention past the force window.
const ISHTAR_BOOK_LINES: &[(&[&str], &str)] = &[
    (&["h2h"], "e8"),
    (&["h2h", "e8"], "e2"),
    (&["h2h", "e8", "e2", "e7"], "e3"),
    (&["e2", "e8", "e3", "e7", "e4", "e6", "a3h"], "e6h"),
    (&["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "e6h"], "c3h"),
    (
        &["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "e6h", "c3h"],
        "e3v",
    ),
];

/// Refuted wall-fest for **White only** — strip from DAG order/play, never forced.
/// Black may still book `e3h` etc. in other positions (winning for Black there).
/// Line: e2 e8 e3 e7 e4 e6 h3h e6h e3h …
const DENIED_WHITE_BOOK_MOVES: &[(&[&str], &str)] = &[
    (&["e2", "e8", "e3", "e7", "e4", "e6"], "h3h"),
    (&["e2", "e8", "e3", "e7", "e4", "e6", "h3h", "e6h"], "e3h"),
    (
        &[
            "e2", "e8", "e3", "e7", "e4", "e6", "h3h", "e6h", "e3h", "c6h",
        ],
        "g2h",
    ),
];

/// Refuted continuation for **Black** — strip from DAG order/play after exact prefix.
/// Line: e2 e8 e3 e7 e4 e6 e3h f6h c3h d6h g3h f6 — f6 is a losing tempo.
const DENIED_BLACK_BOOK_MOVES: &[(&[&str], &str)] = &[(
    &[
        "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "f6h", "c3h", "d6h", "g3h",
    ],
    "f6",
)];

fn history_matches_prefix(g: &GameState, prefix: &[&str]) -> bool {
    g.hist_len == prefix.len()
        && prefix
            .iter()
            .enumerate()
            .all(|(i, expected)| move_id_to_algebraic(g.hist_m[i]) == *expected)
}

/// White must not take this book move at this exact prefix (refuted wall-fest).
pub fn opening_white_book_move_denied(g: &GameState, next: &str) -> bool {
    if g.turn != 0 {
        return false;
    }
    DENIED_WHITE_BOOK_MOVES
        .iter()
        .any(|(prefix, denied)| history_matches_prefix(g, prefix) && next == *denied)
}

/// Black must not take this book move at this exact prefix (refuted continuation).
pub fn opening_black_book_move_denied(g: &GameState, next: &str) -> bool {
    if g.turn != 1 {
        return false;
    }
    DENIED_BLACK_BOOK_MOVES
        .iter()
        .any(|(prefix, denied)| history_matches_prefix(g, prefix) && next == *denied)
}

/// Root search / book consult: side-aware deny list.
pub fn opening_move_would_be_denied(g: &GameState, next: &str) -> bool {
    opening_white_book_move_denied(g, next) || opening_black_book_move_denied(g, next)
}

/// Remove denied opening moves from a legal-move slice (root search / book consult).
pub fn filter_denied_opening_legal_moves(g: &GameState, moves: &mut [i16], n: usize) -> usize {
    let mut write = 0usize;
    for i in 0..n {
        let alg = move_id_to_algebraic(moves[i]);
        if opening_move_would_be_denied(g, &alg) {
            continue;
        }
        moves[write] = moves[i];
        write += 1;
    }
    write
}

fn history_algebraic(g: &GameState) -> Vec<String> {
    (0..g.hist_len)
        .map(|i| move_id_to_algebraic(g.hist_m[i]))
        .collect()
}

fn on_sacred_center_trunk(g: &GameState) -> bool {
    for i in 0..g.hist_len.min(OPENING_SACRED_MAX_PLY) {
        if move_id_to_algebraic(g.hist_m[i]) != SACRED_CENTER_LINE[i] {
            return false;
        }
    }
    true
}

/// Force the canonical center PV for plies 1–6 when still on trunk.
pub fn sacred_center_direct_play(g: &GameState, legal_moves: &[i16]) -> Option<i16> {
    if !on_sacred_center_trunk(g) || g.hist_len >= OPENING_SACRED_MAX_PLY {
        return None;
    }
    let expected = SACRED_CENTER_LINE[g.hist_len];
    let mv = algebraic_to_move_id(expected);
    legal_moves.contains(&mv).then_some(mv)
}

pub fn is_ishtar_tier_move(g: &GameState, reply: &str) -> bool {
    let played = history_algebraic(g);
    ISHTAR_BOOK_LINES.iter().any(|(prefix, expected)| {
        played.len() == prefix.len()
            && played.iter().zip(prefix.iter()).all(|(a, b)| a == b)
            && reply == *expected
    })
}

pub fn candidate_attention_boost(g: &GameState, c: &BookCandidate) -> i32 {
    let wr = (c.raw_win_rate * f64::from(BOOK_ATTENTION_WINRATE_SCALE)) as i32;
    let ishtar = is_ishtar_tier_move(g, &c.algebraic);
    wr + if ishtar {
        BOOK_ATTENTION_ISHTAR_BONUS
    } else {
        0
    }
}

fn should_force_direct_play(ply_from_start: usize, mode: OpeningBookMode) -> bool {
    mode == OpeningBookMode::Play && ply_from_start < OPENING_FORCE_MAX_PLY
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum OpeningBookMode {
    #[default]
    Off,
    Order,
    Play,
}

impl OpeningBookMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "off" | "none" | "0" => Some(Self::Off),
            "order" | "sort" => Some(Self::Order),
            "play" | "direct" => Some(Self::Play),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Order => "order",
            Self::Play => "play",
        }
    }
}

#[derive(Debug, Clone)]
pub struct BookCandidate {
    pub move_code_u8: u8,
    pub algebraic: String,
    pub move_id: i16,
    pub visits: u32,
    pub wins_stm: u32,
    pub losses_stm: u32,
    pub draws: u32,
    pub raw_win_rate: f64,
    pub wilson_lower: f64,
}

#[derive(Debug, Clone, Default)]
pub struct OpeningBookDiagnostics {
    pub mode: OpeningBookMode,
    pub ply_from_start: usize,
    pub position_hit: bool,
    pub effective_mode: OpeningBookMode,
    pub played_directly: bool,
    pub ordered_only: bool,
    pub candidates: Vec<BookCandidate>,
    pub selected_move: Option<i16>,
    pub db_path: String,
}

#[derive(Clone, Copy)]
pub struct BookEdgeRow {
    pub code: u8,
    pub visits: u32,
    pub wins: u32,
    pub losses: u32,
    pub draws: u32,
}

enum OpeningBookBackend {
    #[cfg(not(target_arch = "wasm32"))]
    Sqlite {
        conn: Mutex<Connection>,
        path: PathBuf,
    },
    Embedded,
}

pub struct OpeningBook {
    backend: OpeningBookBackend,
}

impl OpeningBook {
    pub fn open(path: Option<&Path>) -> Result<Arc<Self>, String> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            let path = path
                .map(Path::to_path_buf)
                .unwrap_or_else(Self::default_path);
            if path.is_file() {
                let conn = Connection::open_with_flags(&path, OpenFlags::SQLITE_OPEN_READ_ONLY)
                    .map_err(|e| format!("opening book open failed: {e}"))?;
                return Ok(Arc::new(Self {
                    backend: OpeningBookBackend::Sqlite {
                        conn: Mutex::new(conn),
                        path,
                    },
                }));
            }
        }
        let _ = path;
        Ok(Arc::new(Self {
            backend: OpeningBookBackend::Embedded,
        }))
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn default_path() -> PathBuf {
        if let Ok(raw) = std::env::var("TITANIUM_BOOK_DB") {
            return PathBuf::from(raw);
        }
        PathBuf::from("training/data/opening_book/non_titanium_opening_dag.db")
    }

    fn db_path_label(&self) -> String {
        match &self.backend {
            #[cfg(not(target_arch = "wasm32"))]
            OpeningBookBackend::Sqlite { path, .. } => path.display().to_string(),
            OpeningBookBackend::Embedded => "embedded:non_titanium_opening_dag.bin".into(),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn lookup_position_id(&self, packed: &[u8; 24]) -> Option<i64> {
        let OpeningBookBackend::Sqlite { conn, .. } = &self.backend else {
            return None;
        };
        let conn = conn.lock().ok()?;
        conn.query_row(
            "SELECT position_id FROM positions WHERE packed_state = ?1",
            rusqlite::params![packed.as_slice()],
            |row| row.get(0),
        )
        .ok()
    }

    pub fn consult(
        &self,
        g: &GameState,
        mode: OpeningBookMode,
        legal_moves: &[i16],
    ) -> OpeningBookConsult {
        if mode == OpeningBookMode::Play {
            if let Some(mv) = sacred_center_direct_play(g, legal_moves) {
                let attention = BOOK_ATTENTION_WINRATE_SCALE * 2 + BOOK_ATTENTION_ISHTAR_BONUS;
                let diag = OpeningBookDiagnostics {
                    mode,
                    ply_from_start: g.hist_len,
                    position_hit: true,
                    effective_mode: OpeningBookMode::Play,
                    played_directly: true,
                    ordered_only: false,
                    selected_move: Some(mv),
                    db_path: self.db_path_label(),
                    ..Default::default()
                };
                return OpeningBookConsult {
                    diagnostics: diag,
                    order: vec![mv],
                    order_attention: vec![attention],
                    direct_play: Some(mv),
                };
            }
            if let Some(mv) = mined_black_mainline_direct_play(g, legal_moves) {
                let attention = BOOK_ATTENTION_WINRATE_SCALE * 2 + BOOK_ATTENTION_ISHTAR_BONUS;
                let diag = OpeningBookDiagnostics {
                    mode,
                    ply_from_start: g.hist_len,
                    position_hit: true,
                    effective_mode: OpeningBookMode::Play,
                    played_directly: true,
                    ordered_only: false,
                    selected_move: Some(mv),
                    db_path: self.db_path_label(),
                    ..Default::default()
                };
                return OpeningBookConsult {
                    diagnostics: diag,
                    order: vec![mv],
                    order_attention: vec![attention],
                    direct_play: Some(mv),
                };
            }
        }
        match &self.backend {
            OpeningBookBackend::Embedded => embedded_opening_book().consult(g, mode, legal_moves),
            #[cfg(not(target_arch = "wasm32"))]
            OpeningBookBackend::Sqlite { .. } => self.consult_sqlite(g, mode, legal_moves),
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn consult_sqlite(
        &self,
        g: &GameState,
        mode: OpeningBookMode,
        legal_moves: &[i16],
    ) -> OpeningBookConsult {
        let mut diag = OpeningBookDiagnostics {
            mode,
            ply_from_start: g.hist_len,
            db_path: self.db_path_label(),
            ..Default::default()
        };
        if mode == OpeningBookMode::Off {
            return OpeningBookConsult {
                diagnostics: diag,
                order: Vec::new(),
                order_attention: Vec::new(),
                direct_play: None,
            };
        }
        if g.hist_len >= OPENING_BOOK_EXTENDED_MAX_PLIES {
            diag.effective_mode = OpeningBookMode::Off;
            return OpeningBookConsult {
                diagnostics: diag,
                order: Vec::new(),
                order_attention: Vec::new(),
                direct_play: None,
            };
        }

        let packed = pack_state_dag(g);
        let Some(position_id) = self.lookup_position_id(&packed) else {
            diag.effective_mode = OpeningBookMode::Off;
            return OpeningBookConsult {
                diagnostics: diag,
                order: Vec::new(),
                order_attention: Vec::new(),
                direct_play: None,
            };
        };
        diag.position_hit = true;

        let conn = match self.sqlite_conn() {
            Some(c) => c,
            None => {
                diag.effective_mode = OpeningBookMode::Off;
                return OpeningBookConsult {
                    diagnostics: diag,
                    order: Vec::new(),
                    order_attention: Vec::new(),
                    direct_play: None,
                };
            }
        };
        let mut stmt = match conn.prepare(
            "SELECT move_code_u8, visit_count, wins_stm, losses_stm, draws \
             FROM edges WHERE parent_position_id = ?1",
        ) {
            Ok(s) => s,
            Err(_) => {
                diag.effective_mode = OpeningBookMode::Off;
                return OpeningBookConsult {
                    diagnostics: diag,
                    order: Vec::new(),
                    order_attention: Vec::new(),
                    direct_play: None,
                };
            }
        };
        let rows = stmt.query_map([position_id], |row| {
            Ok(BookEdgeRow {
                code: row.get::<_, i64>(0)? as u8,
                visits: row.get::<_, i64>(1)? as u32,
                wins: row.get::<_, i64>(2)? as u32,
                losses: row.get::<_, i64>(3)? as u32,
                draws: row.get::<_, i64>(4)? as u32,
            })
        });
        let Ok(rows) = rows else {
            diag.effective_mode = OpeningBookMode::Off;
            return OpeningBookConsult {
                diagnostics: diag,
                order: Vec::new(),
                order_attention: Vec::new(),
                direct_play: None,
            };
        };
        let edge_rows: Vec<BookEdgeRow> = rows.flatten().collect();
        consult_from_edge_rows(&mut diag, mode, g, legal_moves, &edge_rows)
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn sqlite_conn(&self) -> Option<std::sync::MutexGuard<'_, Connection>> {
        let OpeningBookBackend::Sqlite { conn, .. } = &self.backend else {
            return None;
        };
        conn.lock().ok()
    }
}

pub fn consult_from_edge_rows(
    diag: &mut OpeningBookDiagnostics,
    mode: OpeningBookMode,
    g: &GameState,
    legal_moves: &[i16],
    edge_rows: &[BookEdgeRow],
) -> OpeningBookConsult {
    let packed = pack_state_dag(g);
    let state = match DatasetState::from_packed(&packed) {
        Ok(s) => s,
        Err(_) => {
            diag.effective_mode = OpeningBookMode::Off;
            return OpeningBookConsult {
                diagnostics: diag.clone(),
                order: Vec::new(),
                order_attention: Vec::new(),
                direct_play: None,
            };
        }
    };

    let legal_set: std::collections::HashSet<i16> = legal_moves.iter().copied().collect();
    let mut candidates = Vec::new();
    for row in edge_rows {
        let Ok(alg) = state.decode_move_code(row.code) else {
            continue;
        };
        let mv = algebraic_to_move_id(&alg);
        if !legal_set.contains(&mv) {
            continue;
        }
        if opening_white_book_move_denied(g, &alg) || opening_black_book_move_denied(g, &alg) {
            continue;
        }
        let raw_wr = raw_win_rate(row.wins, row.losses);
        candidates.push(BookCandidate {
            move_code_u8: row.code,
            algebraic: alg,
            move_id: mv,
            visits: row.visits,
            wins_stm: row.wins,
            losses_stm: row.losses,
            draws: row.draws,
            raw_win_rate: raw_wr,
            wilson_lower: wilson_lower_bound(row.wins, row.losses),
        });
    }
    candidates.sort_by(|a, b| {
        b.wilson_lower
            .partial_cmp(&a.wilson_lower)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| b.visits.cmp(&a.visits))
            .then_with(|| a.algebraic.cmp(&b.algebraic))
    });
    diag.candidates = candidates.clone();

    if candidates.is_empty() {
        diag.effective_mode = OpeningBookMode::Off;
        return OpeningBookConsult {
            diagnostics: diag.clone(),
            order: Vec::new(),
            order_attention: Vec::new(),
            direct_play: None,
        };
    }

    if diag.ply_from_start >= OPENING_BOOK_MAX_PLIES && !qualifies_for_extended_opening(&candidates)
    {
        diag.effective_mode = OpeningBookMode::Off;
        return OpeningBookConsult {
            diagnostics: diag.clone(),
            order: Vec::new(),
            order_attention: Vec::new(),
            direct_play: None,
        };
    }

    let mut order: Vec<i16> = Vec::new();
    let mut order_attention: Vec<i32> = Vec::new();
    for c in &candidates {
        order.push(c.move_id);
        order_attention.push(candidate_attention_boost(g, c));
    }
    diag.effective_mode = mode;

    // Play mode: force through ply 6 only; ply 7+ is order/attention bias for search.
    let direct_play = if should_force_direct_play(diag.ply_from_start, mode) {
        diag.played_directly = true;
        diag.selected_move = Some(candidates[0].move_id);
        Some(candidates[0].move_id)
    } else {
        diag.ordered_only = true;
        None
    };

    OpeningBookConsult {
        diagnostics: diag.clone(),
        order,
        order_attention,
        direct_play,
    }
}

#[derive(Debug, Clone)]
pub struct OpeningBookConsult {
    pub diagnostics: OpeningBookDiagnostics,
    pub order: Vec<i16>,
    /// Parallel to `order`: search root attention bonus (win rate + Ishtar tier).
    pub order_attention: Vec<i32>,
    pub direct_play: Option<i16>,
}

pub fn raw_win_rate(wins: u32, losses: u32) -> f64 {
    let decided = wins + losses;
    if decided == 0 {
        0.5
    } else {
        wins as f64 / decided as f64
    }
}

/// Wilson score lower confidence bound (95%, z = 1.96).
pub fn wilson_lower_bound(wins: u32, losses: u32) -> f64 {
    wilson_lower_bound_z(wins, losses, 1.96)
}

pub fn wilson_lower_bound_z(wins: u32, losses: u32, z: f64) -> f64 {
    let n = (wins + losses) as f64;
    if n <= 0.0 {
        return 0.5;
    }
    let p = wins as f64 / n;
    let z2 = z * z;
    let denom = 1.0 + z2 / n;
    let center = p + z2 / (2.0 * n);
    let margin = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).max(0.0)).sqrt();
    ((center - margin) / denom).clamp(0.0, 1.0)
}

/// Past the base ply cap, keep book order/play only when the top line is statistically strong.
pub fn qualifies_for_extended_opening(candidates: &[BookCandidate]) -> bool {
    let Some(top) = candidates.first() else {
        return false;
    };
    if top.visits < PLAY_MIN_VISITS {
        return false;
    }
    if top.raw_win_rate >= OPENING_BOOK_EXTENDED_MIN_WIN_RATE {
        return true;
    }
    should_play_direct(candidates)
}

pub fn should_play_direct(candidates: &[BookCandidate]) -> bool {
    let Some(top) = candidates.first() else {
        return false;
    };
    if top.visits < PLAY_MIN_VISITS {
        return false;
    }
    let total: u32 = candidates.iter().map(|c| c.visits).sum();
    if total > 0 && top.visits as f64 / total as f64 >= PLAY_MIN_SHARE {
        return true;
    }
    if let Some(second) = candidates.get(1) {
        return top.wilson_lower > second.wilson_lower + PLAY_WILSON_GAP;
    }
    true
}

pub fn diagnostics_json(diag: &OpeningBookDiagnostics) -> String {
    let mut out = String::from("{\"book\":{");
    out.push_str(&format!(
        "\"mode\":\"{}\",\"effectiveMode\":\"{}\",\"positionHit\":{},\"ply\":{},\"playedDirectly\":{},\"orderedOnly\":{},\"db\":",
        diag.mode.as_str(),
        diag.effective_mode.as_str(),
        diag.position_hit,
        diag.ply_from_start,
        diag.played_directly,
        diag.ordered_only,
    ));
    out.push_str(&json_str(&diag.db_path));
    out.push_str(",\"candidates\":[");
    for (i, c) in diag.candidates.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"move\":\"{}\",\"u8\":{},\"visits\":{},\"wins\":{},\"losses\":{},\"draws\":{},\"rawWinRate\":{:.4},\"wilsonLower\":{:.4}}}",
            json_escape(&c.algebraic),
            c.move_code_u8,
            c.visits,
            c.wins_stm,
            c.losses_stm,
            c.draws,
            c.raw_win_rate,
            c.wilson_lower,
        ));
    }
    out.push_str("],\"selectedMove\":");
    match diag.selected_move {
        Some(mv) => out.push_str(&format!("\"{}\"", json_escape(&move_id_to_algebraic(mv)))),
        None => out.push_str("null"),
    }
    out.push_str("}}");
    out
}

fn json_escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn json_str(s: &str) -> String {
    format!("\"{}\"", json_escape(s))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::titanium::game::GameState;
    use std::path::PathBuf;

    fn book_path() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("training")
            .join("data")
            .join("opening_book")
            .join("non_titanium_opening_dag.db")
    }

    #[test]
    fn root_packed_state_lookup() {
        let book = match OpeningBook::open(Some(&book_path())) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip: {e}");
                return;
            }
        };
        let g = GameState::new();
        let _packed = crate::titanium::pack_state(&g);
        let consult = book.consult(
            &g,
            OpeningBookMode::Order,
            &[algebraic_to_move_id("e2"), algebraic_to_move_id("d2")],
        );
        assert!(consult.diagnostics.position_hit);
    }

    #[test]
    fn wilson_ranks_e2_above_f1_at_root() {
        let book = match OpeningBook::open(Some(&book_path())) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip: {e}");
                return;
            }
        };
        let g = GameState::new();
        let legal = vec![
            algebraic_to_move_id("e2"),
            algebraic_to_move_id("d2"),
            algebraic_to_move_id("f2"),
            algebraic_to_move_id("f1"),
            algebraic_to_move_id("d1"),
        ];
        let consult = book.consult(&g, OpeningBookMode::Order, &legal);
        assert!(consult.diagnostics.position_hit);
        let e2 = consult
            .diagnostics
            .candidates
            .iter()
            .find(|c| c.algebraic == "e2")
            .expect("e2 in book");
        let f1 = consult
            .diagnostics
            .candidates
            .iter()
            .find(|c| c.algebraic == "f1")
            .expect("f1 in book");
        assert!(e2.visits > f1.visits);
        assert_eq!(consult.order[0], e2.move_id);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn book_off_matches_default_search_on_startpos() {
        use crate::titanium::opening_book::OpeningBookMode;
        use crate::titanium::TitaniumSearch;
        use std::path::PathBuf;

        let db = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("training")
            .join("data")
            .join("opening_book")
            .join("non_titanium_opening_dag.db");
        if !db.is_file() {
            eprintln!("skip: opening dag missing");
            return;
        }
        let mut off = TitaniumSearch::grafted(GameState::new(), Some(18));
        off.set_opening_book(OpeningBookMode::Off, Some(db.clone()));
        let r_off = off.think(50, 8, true, false, "book-test-off");

        let mut on = TitaniumSearch::grafted(GameState::new(), Some(18));
        on.set_opening_book(OpeningBookMode::Off, None);
        let r_on = on.think(50, 8, true, false, "book-test-off");

        assert_eq!(r_off.mv, r_on.mv);
        // Node counts may differ when one path opens the sqlite book handle even in Off mode.
    }

    #[test]
    fn missing_position_is_miss() {
        let book = match OpeningBook::open(Some(&book_path())) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip: {e}");
                return;
            }
        };
        let mut g = GameState::new();
        for mv in ["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "d4v", "e5", "f6"] {
            g.make_move(algebraic_to_move_id(mv));
        }
        let consult = book.consult(&g, OpeningBookMode::Order, &[algebraic_to_move_id("e7")]);
        assert!(!consult.diagnostics.position_hit);
        assert!(consult.order.is_empty());
    }

    #[test]
    fn illegal_db_move_rejected() {
        let book = match OpeningBook::open(Some(&book_path())) {
            Ok(b) => b,
            Err(e) => {
                eprintln!("skip: {e}");
                return;
            }
        };
        let g = GameState::new();
        // only d2 legal in this fake set — e2 from book should be dropped
        let consult = book.consult(&g, OpeningBookMode::Order, &[algebraic_to_move_id("d2")]);
        assert!(consult
            .diagnostics
            .candidates
            .iter()
            .all(|c| c.move_id == algebraic_to_move_id("d2")));
    }

    #[test]
    fn sacred_center_forces_first_six_plies() {
        let book = OpeningBook::open(None).expect("embedded book");
        let mut g = GameState::new();
        for expected in ["e2", "e8", "e3", "e7", "e4"] {
            let legal = [algebraic_to_move_id(expected)];
            let consult = book.consult(&g, OpeningBookMode::Play, &legal);
            assert_eq!(
                consult.direct_play,
                Some(algebraic_to_move_id(expected)),
                "sacred ply {}",
                g.hist_len + 1
            );
            g.make_move(algebraic_to_move_id(expected));
        }
        // Ply 6 (e6) is still sacred-forced too — the extended trunk.
        let legal = [algebraic_to_move_id("e6")];
        let consult = book.consult(&g, OpeningBookMode::Play, &legal);
        assert_eq!(
            consult.direct_play,
            Some(algebraic_to_move_id("e6")),
            "sacred ply 6 (e6) must still be forced"
        );
    }

    #[test]
    fn mined_mainline_forces_black_a6h_after_a3h() {
        let book = OpeningBook::open(None).expect("embedded book");
        let mut g = GameState::new();
        for mv in ["e2", "e8", "e3", "e7", "e4", "e6", "a3h"] {
            g.make_move(algebraic_to_move_id(mv));
        }
        let legal = [algebraic_to_move_id("a6h")];
        let consult = book.consult(&g, OpeningBookMode::Play, &legal);
        assert_eq!(
            consult.direct_play,
            Some(algebraic_to_move_id("a6h")),
            "Black's a6h must be forced after White's a3h (proven 3/3 wins)"
        );
    }

    #[test]
    fn mined_mainline_does_not_force_white_a3h() {
        // White's a3h itself must NOT be forced -- Ka (White) lost every
        // source game playing it. Only Black's replies are mined-forced.
        let book = OpeningBook::open(None).expect("embedded book");
        let mut g = GameState::new();
        for mv in ["e2", "e8", "e3", "e7", "e4", "e6"] {
            g.make_move(algebraic_to_move_id(mv));
        }
        let legal = [algebraic_to_move_id("a3h"), algebraic_to_move_id("h3h")];
        let consult = book.consult(&g, OpeningBookMode::Play, &legal);
        assert_ne!(
            consult.direct_play,
            Some(algebraic_to_move_id("a3h")),
            "White's a3h must stay free search, not forced"
        );
    }

    #[test]
    fn mined_mainline_forces_full_line_a_through_ply_12() {
        let book = OpeningBook::open(None).expect("embedded book");
        // White's moves (a3h, e3h, e5) are asserted as-played but not
        // book-forced; Black's moves (a6h, c3v, e6h) must be forced.
        let mut g = GameState::new();
        let white_moves = ["e2", "e3", "e4", "a3h", "e3h", "e5"];
        let black_forced = ["e8", "e7", "e6", "a6h", "c3v", "e6h"];
        for i in 0..6 {
            g.make_move(algebraic_to_move_id(white_moves[i]));
            let legal = [algebraic_to_move_id(black_forced[i])];
            let consult = book.consult(&g, OpeningBookMode::Play, &legal);
            assert_eq!(
                consult.direct_play,
                Some(algebraic_to_move_id(black_forced[i])),
                "Black ply {} ({}) must be forced",
                g.hist_len + 1,
                black_forced[i]
            );
            g.make_move(algebraic_to_move_id(black_forced[i]));
        }
    }

    #[test]
    fn mined_mainline_forces_line_b_ply_10_and_12_once_on_branch() {
        // Ply 8 in this branch (e3v) is NOT book-forced -- a6h is the primary
        // forced ply-8 reply (proven 3/3 wins), so e3v is only ever reached
        // manually/hypothetically here, exactly like this test constructs it.
        // Once on that branch, plies 10 (e4) and 12 (d4h) ARE forced.
        let book = OpeningBook::open(None).expect("embedded book");
        let mut g = GameState::new();
        for mv in ["e2", "e8", "e3", "e7", "e4", "e6", "a3h", "e3v", "e5"] {
            g.make_move(algebraic_to_move_id(mv));
        }
        let legal = [algebraic_to_move_id("e4")];
        let consult = book.consult(&g, OpeningBookMode::Play, &legal);
        assert_eq!(
            consult.direct_play,
            Some(algebraic_to_move_id("e4")),
            "Black ply 10 (e4) must be forced once on the e3v branch"
        );
        g.make_move(algebraic_to_move_id("e4"));
        g.make_move(algebraic_to_move_id("d3h"));
        let legal = [algebraic_to_move_id("d4h")];
        let consult = book.consult(&g, OpeningBookMode::Play, &legal);
        assert_eq!(
            consult.direct_play,
            Some(algebraic_to_move_id("d4h")),
            "Black ply 12 (d4h) must be forced once on the e3v branch"
        );
    }

    #[test]
    fn play_mode_forces_through_ply_6_only() {
        let mut diag = OpeningBookDiagnostics {
            mode: OpeningBookMode::Play,
            ply_from_start: 3,
            ..Default::default()
        };
        let g = GameState::new();
        let rows = vec![BookEdgeRow {
            code: 128,
            visits: 3,
            wins: 1,
            losses: 2,
            draws: 0,
        }];
        let consult = consult_from_edge_rows(
            &mut diag,
            OpeningBookMode::Play,
            &g,
            &[algebraic_to_move_id("e2")],
            &rows,
        );
        assert_eq!(consult.direct_play, Some(algebraic_to_move_id("e2")));
        assert!(diag.played_directly);

        diag.ply_from_start = OPENING_FORCE_MAX_PLY;
        let consult = consult_from_edge_rows(
            &mut diag,
            OpeningBookMode::Play,
            &g,
            &[algebraic_to_move_id("e2")],
            &rows,
        );
        assert!(consult.direct_play.is_none());
        assert!(!consult.order.is_empty());
        assert!(!consult.order_attention.is_empty());
    }

    #[test]
    fn white_e3h_blocked_after_h3h_e6h() {
        let book = OpeningBook::open(None).expect("embedded book");
        let mut g = GameState::new();
        for mv in ["e2", "e8", "e3", "e7", "e4", "e6", "h3h", "e6h"] {
            g.make_move(algebraic_to_move_id(mv));
        }
        assert_eq!(g.turn, 0, "White to move");
        let mut buf = [0i16; 160];
        let n = g.gen_legal_moves(&mut buf);
        let consult = book.consult(&g, OpeningBookMode::Play, &buf[..n]);
        assert!(
            !consult
                .order
                .iter()
                .any(|&m| m == algebraic_to_move_id("e3h")),
            "e3h is losing for White on refuted h3h line — must not be in book"
        );
        assert_ne!(
            consult.direct_play,
            Some(algebraic_to_move_id("e3h")),
            "e3h must not be forced for White"
        );
    }

    #[test]
    fn white_g2h_blocked_on_refuted_wall_fest() {
        let book = OpeningBook::open(None).expect("embedded book");
        let mut g = GameState::new();
        for mv in [
            "e2", "e8", "e3", "e7", "e4", "e6", "h3h", "e6h", "e3h", "c6h",
        ] {
            g.make_move(algebraic_to_move_id(mv));
        }
        let mut buf = [0i16; 160];
        let n = g.gen_legal_moves(&mut buf);
        let consult = book.consult(&g, OpeningBookMode::Play, &buf[..n]);
        assert_ne!(
            consult.direct_play,
            Some(algebraic_to_move_id("g2h")),
            "g2h must not be forced from book"
        );
        assert!(
            !consult
                .order
                .iter()
                .any(|&m| m == algebraic_to_move_id("g2h")),
            "g2h must not appear in opening book order for refuted wall-fest"
        );
    }

    #[test]
    fn h3h_wall_fest_excluded_from_book_after_e4_e6() {
        let book = OpeningBook::open(None).expect("embedded book");
        let mut g = GameState::new();
        for mv in ["e2", "e8", "e3", "e7", "e4", "e6"] {
            g.make_move(algebraic_to_move_id(mv));
        }
        let mut buf = [0i16; 160];
        let n = g.gen_legal_moves(&mut buf);
        let consult = book.consult(&g, OpeningBookMode::Play, &buf[..n]);
        assert!(
            !consult
                .order
                .iter()
                .any(|&m| m == algebraic_to_move_id("h3h")),
            "h3h refuted wall-fest must not appear in book order"
        );
        assert!(
            consult.direct_play != Some(algebraic_to_move_id("h3h")),
            "h3h must never be direct-play"
        );
    }

    #[test]
    fn black_f6_blocked_after_e3h_g3h_line() {
        let book = OpeningBook::open(None).expect("embedded book");
        let mut g = GameState::new();
        for mv in [
            "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "f6h", "c3h", "d6h", "g3h",
        ] {
            g.make_move(algebraic_to_move_id(mv));
        }
        assert_eq!(g.turn, 1, "Black to move");
        let mut buf = [0i16; 160];
        let n = g.gen_legal_moves(&mut buf);
        let consult = book.consult(&g, OpeningBookMode::Play, &buf[..n]);
        assert_ne!(
            consult.direct_play,
            Some(algebraic_to_move_id("f6")),
            "f6 must not be forced from book after g3h"
        );
        assert!(
            !consult
                .order
                .iter()
                .any(|&m| m == algebraic_to_move_id("f6")),
            "f6 must not appear in opening book order on refuted e3h line"
        );
    }

    #[test]
    fn black_book_still_active_after_white_wall_fest_g2h() {
        let book = OpeningBook::open(None).expect("embedded book");
        let mut g = GameState::new();
        for mv in [
            "e2", "e8", "e3", "e7", "e4", "e6", "h3h", "e6h", "e3h", "c6h", "g2h",
        ] {
            g.make_move(algebraic_to_move_id(mv));
        }
        assert_eq!(g.turn, 1, "Black to move — book not globally disabled");
        let mut buf = [0i16; 160];
        let n = g.gen_legal_moves(&mut buf);
        let consult = book.consult(&g, OpeningBookMode::Order, &buf[..n]);
        // Black may still use DAG hints here; denylist is White-only.
        assert_ne!(consult.diagnostics.effective_mode, OpeningBookMode::Off);
    }

    #[test]
    fn extended_opening_requires_strong_top_line() {
        let strong = BookCandidate {
            move_code_u8: 128,
            algebraic: "e2".into(),
            move_id: algebraic_to_move_id("e2"),
            visits: 20,
            wins_stm: 12,
            losses_stm: 8,
            draws: 0,
            raw_win_rate: 0.6,
            wilson_lower: wilson_lower_bound(12, 8),
        };
        assert!(qualifies_for_extended_opening(&[strong.clone()]));
        let weak_top = BookCandidate {
            visits: 20,
            wins_stm: 8,
            losses_stm: 12,
            raw_win_rate: 0.4,
            wilson_lower: wilson_lower_bound(8, 12),
            ..strong.clone()
        };
        let alt = BookCandidate {
            move_code_u8: 130,
            algebraic: "f1".into(),
            move_id: algebraic_to_move_id("f1"),
            visits: 18,
            wins_stm: 9,
            losses_stm: 9,
            draws: 0,
            raw_win_rate: 0.5,
            wilson_lower: wilson_lower_bound(9, 9),
        };
        assert!(!qualifies_for_extended_opening(&[weak_top, alt]));
    }

    #[test]
    fn consult_from_edge_rows_blocks_ply_12_without_strong_line() {
        let mut diag = OpeningBookDiagnostics {
            mode: OpeningBookMode::Order,
            ply_from_start: OPENING_BOOK_MAX_PLIES,
            ..Default::default()
        };
        let g = GameState::new();
        let rows = vec![BookEdgeRow {
            code: 128,
            visits: 8,
            wins: 3,
            losses: 5,
            draws: 0,
        }];
        let consult = consult_from_edge_rows(
            &mut diag,
            OpeningBookMode::Order,
            &g,
            &[algebraic_to_move_id("e2")],
            &rows,
        );
        assert!(consult.order.is_empty());
        assert_eq!(diag.effective_mode, OpeningBookMode::Off);
    }
}
