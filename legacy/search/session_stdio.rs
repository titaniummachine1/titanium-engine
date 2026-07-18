//! Long-lived stdin/stdout session — one process per web UI engine seat.
//!
//! Commands (one per line):
//!   `reset`
//!   `position MOVE [MOVE ...]`
//!   `makemove MOVE`
//!   `go TIME_SEC [MAX_NODES]`
//!   `quit`
//!
//! Responses on stdout: `ready` | `bestmove MOVE` | `error MESSAGE`
//! Search progress on stderr (`info json …`), same as `genmove --log`.

use std::io::{self, BufRead, Write};

use crate::legacy_search::alphabeta::{run_search, SearchConfig, DEFAULT_MAX_NODES, DEFAULT_TIME_MS};
use crate::legacy_search::session::GameSearchSession;
use crate::util::perft::format_move;

fn reply_ready(stdout: &mut io::Stdout) {
    let _ = writeln!(stdout, "ready");
    let _ = stdout.flush();
}

fn reply_error(stdout: &mut io::Stdout, message: &str) {
    let _ = writeln!(stdout, "error {}", message);
    let _ = stdout.flush();
}

fn run_go(session: &mut GameSearchSession, parts: &[&str], stdout: &mut io::Stdout) {
    let time_sec: f64 = parts
        .get(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_TIME_MS as f64 / 1000.0);
    let max_nodes: u64 = parts
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_NODES);
    let time_ms = (time_sec * 1000.0).max(1.0) as u64;

    let config = SearchConfig {
        time_ms,
        max_nodes,
        log: true,
        book_hint: None,
        max_id_depth: crate::legacy_search::alphabeta::DEFAULT_MAX_ID_DEPTH,
        cert_enabled: None,
    };

    match run_search(session, config) {
        Some(report) => {
            let _ = writeln!(stdout, "bestmove {}", format_move(report.best_move));
        }
        None => {
            let _ = writeln!(stdout, "bestmove (none)");
        }
    }
    let _ = stdout.flush();
}

/// Blocking REPL — intended for a single long-lived child process per engine seat.
pub fn run_session_stdio() {
    let mut session = GameSearchSession::new();
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
                session.reset();
                reply_ready(&mut stdout);
            }
            "position" => {
                let moves: Vec<String> = parts[1..].iter().map(|s| (*s).to_string()).collect();
                match session.set_position(&moves) {
                    Ok(applied) => {
                        let _ = writeln!(stdout, "ready {applied}");
                        let _ = stdout.flush();
                    }
                    Err(msg) => reply_error(&mut stdout, &msg),
                }
            }
            "makemove" => {
                let Some(mv) = parts.get(1) else {
                    reply_error(&mut stdout, "makemove requires a move");
                    continue;
                };
                if !session.apply_algebraic(mv) {
                    reply_error(&mut stdout, "illegal or terminal position");
                    continue;
                }
                reply_ready(&mut stdout);
            }
            "go" => {
                if session.board.is_terminal().is_some() {
                    reply_error(&mut stdout, "terminal position");
                    continue;
                }
                run_go(&mut session, &parts, &mut stdout);
            }
            "quit" => break,
            _ => reply_error(&mut stdout, "unknown command"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::Board;

    #[test]
    fn session_reset_startpos() {
        let mut session = GameSearchSession::new();
        assert_eq!(session.board, Board::new());
        session
            .set_position(&["e2".to_string(), "e8".to_string()])
            .expect("e2 e8");
        assert_ne!(session.board, Board::new());
        session.reset();
        assert_eq!(session.board, Board::new());
    }
}
