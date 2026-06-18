//! Perft — ACE v7 native movegen vs Titanium `movegen` / `perft_fast`.

use crate::util::clock::{Duration, Instant};
use std::sync::mpsc;

use crate::acev13::game::AceGame;
use crate::acev13::search::board_move_to_ace;
use crate::core::board::Board;
use crate::core::board::Undo;
use crate::movegen::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::path::BfsScratch;
use crate::search::runtime::Engine;
use crate::util::perft::{perft_fast, PERFT3_STARTPOS, PERFT4_STARTPOS, PERFT4_TEST_TIMEOUT_SECS};

pub const ACE_PERFT4_STARTPOS: u64 = PERFT4_STARTPOS;

#[derive(Debug, Clone)]
pub struct TimedPerftResult {
    pub depth: u32,
    pub nodes: Option<u64>,
    pub elapsed_ms: u64,
    pub timed_out: bool,
    pub label: &'static str,
}

impl AceGame {
    pub fn gen_legal_moves(&mut self, out: &mut [i16; 160]) -> usize {
        let mut n = self.gen_pawn_moves(out, 0);
        if self.wl[self.turn] > 0 {
            for slot in 0..64 {
                if self.wall_legal(0, slot) {
                    out[n] = 100 + slot as i16;
                    n += 1;
                }
                if self.wall_legal(1, slot) {
                    out[n] = 200 + slot as i16;
                    n += 1;
                }
            }
        }
        n
    }
}

fn perft_ace_native(g: &mut AceGame, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut buf = [0i16; 160];
    let n = g.gen_legal_moves(&mut buf);
    let mut nodes = 0u64;
    for i in 0..n {
        g.make_move(buf[i]);
        nodes += perft_ace_native(g, depth - 1);
        g.unmake_move();
    }
    nodes
}

#[allow(dead_code)]
fn perft_ace_ti_gen(
    g: &mut AceGame,
    depth: u32,
    board: &mut Board,
    bfs: &mut BfsScratch,
    undo_stack: &mut Vec<Undo>,
) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut ti_buf = [crate::core::board::Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = generate_legal_moves_slice(board, &mut ti_buf, bfs);
    let mut nodes = 0u64;
    for i in 0..n {
        let ace_m = board_move_to_ace(ti_buf[i]);
        let undo = board.make_move(ti_buf[i]);
        undo_stack.push(undo);
        g.make_move(ace_m);
        nodes += perft_ace_ti_gen(g, depth - 1, board, bfs, undo_stack);
        g.unmake_move();
        if let Some(u) = undo_stack.pop() {
            board.unmake_move(u);
        }
    }
    nodes
}

fn run_timed<F>(label: &'static str, depth: u32, timeout: Duration, work: F) -> TimedPerftResult
where
    F: FnOnce() -> u64 + Send + 'static,
{
    let (done_tx, done_rx) = mpsc::channel::<(u64, Duration)>();
    let wall_start = Instant::now();

    let handle = std::thread::Builder::new()
        .name(format!("perft-{label}-d{depth}"))
        .spawn(move || {
            let t0 = Instant::now();
            let nodes = work();
            let _ = done_tx.send((nodes, t0.elapsed()));
        })
        .expect("spawn perft worker");

    match done_rx.recv_timeout(timeout) {
        Ok((nodes, worker_elapsed)) => {
            handle.join().ok();
            TimedPerftResult {
                depth,
                nodes: Some(nodes),
                elapsed_ms: worker_elapsed.as_millis() as u64,
                timed_out: false,
                label,
            }
        }
        Err(mpsc::RecvTimeoutError::Timeout) | Err(mpsc::RecvTimeoutError::Disconnected) => {
            std::mem::forget(handle);
            TimedPerftResult {
                depth,
                nodes: None,
                elapsed_ms: wall_start.elapsed().as_millis() as u64,
                timed_out: true,
                label,
            }
        }
    }
}

pub fn perft_ace_timed(depth: u32, timeout: Duration) -> TimedPerftResult {
    run_timed("ace-v7-native", depth, timeout, move || {
        let mut g = AceGame::new();
        perft_ace_native(&mut g, depth)
    })
}

/// Titanium movegen on the ACE ruleset — same nodes as `perft_fast` (verified at depth 3).
pub fn perft_ace_ti_timed(depth: u32, timeout: Duration) -> TimedPerftResult {
    perft_titanium_timed_with_label("ace-ti-movegen", depth, timeout)
}

fn perft_titanium_timed_with_label(
    label: &'static str,
    depth: u32,
    timeout: Duration,
) -> TimedPerftResult {
    run_timed(label, depth, timeout, move || {
        let mut board = Board::new();
        perft_fast(&mut board, depth)
    })
}

pub fn perft_titanium_timed(depth: u32, timeout: Duration) -> TimedPerftResult {
    perft_titanium_timed_with_label("titanium-perft_fast", depth, timeout)
}

pub fn perft_engine_timed(depth: u32, timeout: Duration) -> TimedPerftResult {
    run_timed("engine", depth, timeout, move || {
        let board = Board::new();
        Engine::new().perft(&board, depth)
    })
}

pub fn oracle_nodes(depth: u32) -> Option<u64> {
    match depth {
        0 => Some(1),
        1 => Some(131),
        2 => Some(16_677),
        3 => Some(PERFT3_STARTPOS),
        4 => Some(PERFT4_STARTPOS),
        _ => None,
    }
}

pub fn default_timeout(depth: u32) -> Duration {
    if depth >= 4 {
        Duration::from_secs(PERFT4_TEST_TIMEOUT_SECS)
    } else if depth == 3 {
        Duration::from_secs(5)
    } else {
        Duration::from_secs(2)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perft3_native_matches_titanium_and_ti_bridge() {
        let mut g = AceGame::new();
        let native = perft_ace_native(&mut g, 3);
        let mut board = Board::new();
        let mut bfs = BfsScratch::new();
        let mut undo = Vec::new();
        let ti_gen = perft_ace_ti_gen(&mut g, 3, &mut board, &mut bfs, &mut undo);
        let mut board2 = Board::new();
        let fast = perft_fast(&mut board2, 3);
        assert_eq!(native, ti_gen);
        assert_eq!(native, fast);
        assert_eq!(native, PERFT3_STARTPOS);
    }

    /// Heavy (247M nodes): confirms acev13 make/unmake stays in lockstep with
    /// the independently-validated Titanium legal-move list to depth 4.
    /// Run with `cargo test --release acev13 -- --ignored`.
    #[test]
    #[ignore]
    fn perft4_ti_bridge_matches_oracle() {
        let mut g = AceGame::new();
        let mut board = Board::new();
        let mut bfs = BfsScratch::new();
        let mut undo = Vec::new();
        let ti_gen = perft_ace_ti_gen(&mut g, 4, &mut board, &mut bfs, &mut undo);
        assert_eq!(ti_gen, PERFT4_STARTPOS);
    }
}
