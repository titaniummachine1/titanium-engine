//! Perft (divide) — correctness + fast make/unmake + TT + iterative deepening driver.
//!
//! **Standard correctness depth:** 3 from startpos → **2_062_264** nodes.

/// Startpos perft(3) — agreed by scraped JS, gorisanson, and Titanium.
pub const PERFT3_STARTPOS: u64 = 2_062_264;

/// Startpos perft(4) — Ishtar / Canta oracle (2025).
pub const PERFT4_STARTPOS: u64 = 247_569_030;

use crate::core::board::{Board, Move};
use crate::movegen::{generate_legal_moves_into, generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::path::BfsScratch;
use crate::search::context::{SharedState, WorkerContext};
use std::collections::BTreeMap;

/// Back-compat name — prefer [`WorkerContext`] + [`SharedState`] or [`crate::search::context::Engine`].
pub type PerftContext = WorkerContext;

pub fn perft_fast_ctx(
    board: &mut Board,
    depth: u32,
    mut shared: Option<&mut SharedState>,
    worker: &mut WorkerContext,
) -> u64 {
    if depth == 0 {
        return 1;
    }

    if let Some(shared) = shared.as_mut() {
        if let Some(nodes) = shared.tt.probe(board.hash, depth as u8) {
            return nodes;
        }
    }

    let mut move_buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let move_count = generate_legal_moves_slice(board, &mut move_buf, &mut worker.bfs);
    let mut nodes = 0u64;

    for i in 0..move_count {
        let mv = move_buf[i];
        let undo = board.make_move(mv);
        nodes += perft_fast_ctx(board, depth - 1, shared.as_deref_mut(), worker);
        board.unmake_move(undo);
    }

    if let Some(shared) = shared {
        shared.tt.store(board.hash, depth as u8, nodes);
    }
    nodes
}

/// Fast perft — single-threaded via a fresh [`SharedState`].
pub fn perft_fast(board: &mut Board, depth: u32) -> u64 {
    let mut shared = SharedState::new();
    let mut worker = WorkerContext::new();
    perft_fast_ctx(board, depth, Some(&mut shared), &mut worker)
}

/// Root-split parallel perft — experimental bench path when `threads > 1`.
/// Each root move runs in its own subtree with private TT (embarrassingly parallel).
pub fn perft_parallel_root(board: &Board, depth: u32, pool: &rayon::ThreadPool) -> u64 {
    if depth == 0 {
        return 1;
    }

    let mut probe = board.clone();
    let mut worker = WorkerContext::new();
    let mut move_buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let move_count = generate_legal_moves_slice(&mut probe, &mut move_buf, &mut worker.bfs);
    let moves = &move_buf[..move_count];

    pool.install(|| {
        use rayon::prelude::*;
        moves
            .par_iter()
            .map(|&mv| {
                let mut child = board.clone();
                let mut worker = WorkerContext::new();
                let _undo = child.make_move(mv);
                // No TT per worker — avoids 131× heap alloc; subtrees are independent.
                perft_fast_ctx(&mut child, depth - 1, None, &mut worker)
            })
            .sum()
    })
}

pub fn perft_iterative(
    board: &mut Board,
    max_depth: u32,
    shared: &mut SharedState,
) -> Vec<(u32, u64)> {
    let mut out = Vec::with_capacity(max_depth as usize + 1);
    let mut worker = WorkerContext::new();
    for depth in 0..=max_depth {
        shared.clear_tt();
        let nodes = if depth == 0 {
            1
        } else {
            perft_fast_ctx(board, depth, Some(shared), &mut worker)
        };
        out.push((depth, nodes));
    }
    out
}

/// Legacy naive perft (clone) — kept for differential testing.
pub fn perft_naive(board: &Board, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let mut probe = board.clone();
    let mut moves = Vec::new();
    let mut scratch = BfsScratch::new();
    generate_legal_moves_into(&mut probe, &mut moves, &mut scratch);
    let mut nodes = 0u64;
    for &mv in &moves {
        let mut next = board.clone();
        next.apply_move(mv);
        nodes += perft_naive(&next, depth - 1);
    }
    nodes
}

/// Default perft entry — single-thread [`crate::search::runtime::Engine`].
pub fn perft(board: &Board, depth: u32) -> u64 {
    crate::search::runtime::Engine::new().perft(board, depth)
}

pub fn perft_divide(board: &Board, depth: u32) -> (u64, BTreeMap<String, u64>) {
    let mut lines = BTreeMap::new();
    let mut moves = Vec::new();
    let mut copy = board.clone();
    let mut scratch = BfsScratch::new();
    generate_legal_moves_into(&mut copy, &mut moves, &mut scratch);
    let mut total = 0u64;

    for &mv in &moves {
        let label = format_move(mv);
        let undo = copy.make_move(mv);
        let nodes = perft(&copy, depth.saturating_sub(1));
        copy.unmake_move(undo);
        lines.insert(label, nodes);
        total += nodes;
    }
    (total, lines)
}

pub fn format_move(mv: Move) -> String {
    match mv {
        Move::Pawn { row, col } => Board::format_square(row, col),
        Move::Wall {
            row,
            col,
            orientation,
        } => {
            let suffix = match orientation {
                crate::core::board::WallOrientation::Horizontal => 'h',
                crate::core::board::WallOrientation::Vertical => 'v',
            };
            format!("{}{}{}", Board::column_char(col), row + 1, suffix)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::movegen::generate_legal_moves;
    use crate::search::runtime::Engine;

    #[test]
    fn perft_depth1_start() {
        let board = Board::new();
        assert_eq!(perft(&board, 1), generate_legal_moves(&board).len() as u64);
    }

    #[test]
    fn perft_depth0_is_one() {
        let board = Board::new();
        assert_eq!(perft(&board, 0), 1);
    }

    #[test]
    fn perft_depth2_smoke() {
        let board = Board::new();
        assert_eq!(perft(&board, 2), 16_677);
    }

    #[test]
    fn perft_depth3_matches_js_oracle() {
        let board = Board::new();
        assert_eq!(perft(&board, 3), PERFT3_STARTPOS);
    }

    #[test]
    fn fast_matches_naive_depth3() {
        let board = Board::new();
        let naive = perft_naive(&board, 3);
        let mut fast_board = board.clone();
        let fast = perft_fast(&mut fast_board, 3);
        assert_eq!(naive, fast);
        assert_eq!(fast, PERFT3_STARTPOS);
    }

    #[test]
    fn iterative_depth3_last() {
        let mut board = Board::new();
        let mut shared = SharedState::new();
        let lines = perft_iterative(&mut board, 3, &mut shared);
        assert_eq!(lines.last().map(|x| x.1), Some(PERFT3_STARTPOS));
    }

    #[test]
    fn parallel_root_depth3() {
        let board = Board::new();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        assert_eq!(perft_parallel_root(&board, 3, &pool), PERFT3_STARTPOS);
    }

    #[test]
    fn engine_iterative_depth3() {
        let mut board = Board::new();
        let mut engine = Engine::new();
        let lines = engine.perft_iterative(&mut board, 3);
        assert_eq!(lines.last().map(|x| x.1), Some(PERFT3_STARTPOS));
    }

    /// Full-tree regression — run with `cargo test --release perft_depth4 -- --ignored`.
    #[test]
    #[ignore = "slow in debug; run: cargo test --release perft_depth4 -- --ignored"]
    fn perft_depth4_matches_oracle() {
        let board = Board::new();
        let mut fast_board = board.clone();
        assert_eq!(perft_fast(&mut fast_board, 4), PERFT4_STARTPOS);
    }
}
