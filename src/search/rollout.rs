//! EXPERIMENTAL — simulation-guided move evaluation (v14.1 MCTS ideas in a
//! minimax frame).
//!
//! The v14.1 AlphaZero engine ranks root moves with a PUCT tree whose leaf
//! values come from a neural net. Titanium has no net, but it has a strong
//! hand-crafted static eval (`alphabeta::eval_stm`). This module asks the
//! research question:
//!
//!   *Can cheap eval-guided rollouts (a "soft minimax" playout) predict the
//!    deep alpha-beta move ranking well enough to seed move ordering?*
//!
//! Two ingredients are lifted directly from v14.1:
//!   * `rprior = sigmoid(eval / 400)` — the eval→win-probability prior
//!     (`stubValue` / DELTA-7 `rprior` in `AceV14.1.html`).
//!   * a value/visit accumulation per root move (the `W`/`N` arrays of `TNode`),
//!     here filled by playouts instead of a net-guided tree descent.
//!
//! Each rollout is a *soft minimax* playout: at every ply, the side to move
//! samples a move via a softmax over the one-ply static eval from its own
//! perspective (temperature `TEMP`). Greedy in the limit `TEMP → 0`, uniform as
//! `TEMP → ∞`. The playout ends at a goal (value 1/0 for the hero) or after
//! `max_plies` (value = `sigmoid(eval)`).
//!
//! This file does NOT touch the production search. It is a measurement harness
//! reached only via the `titanium rollout` CLI subcommand.
//!
//! ── VERDICT (measured 2026-06-14, native build) ──────────────────────────────
//! Eval-guided rollouts are a DEAD END for seeding Titanium's root ordering:
//!   * Speed: ~9 s (24 sims) … 56 s (128 sims) to rank one position, vs the
//!     deep αβ search it would seed taking 0.6 s. ~15–80× slower — a non-starter
//!     as a *seed* for an already-sub-second search.
//!   * Correlation: Spearman ρ ≈ 0.11–0.20 vs the depth-8 ranking, and it does
//!     NOT improve with more sims (0.19 @24 → 0.11 @128) — the error is BIAS,
//!     not variance. The soft-eval policy just walks shortest paths and never
//!     finds the tactical wall sequences αβ values.
//!   * Confound: at quiet positions (startpos) deep search returns many moves
//!     with IDENTICAL scores (all reasonable openings transpose), so there is no
//!     well-ordered ground truth to correlate against there anyway.
//! Root cause: v14.1's PUCT works because a trained NET supplies strong priors
//! and leaf values. With only the static eval (near-flat across walls: prior
//! ≈ 0.493 for every startpos wall), playouts are noise. The portable v14.1
//! ideas for a netless engine are NOT rollouts but proof propagation (`pin` —
//! Titanium already has this via mate scores + the v13 certify_win solver) and
//! the `mhat` BFS race-margin ordering bonus. Kept as a harness for re-testing
//! if a learned prior is ever added; NOT wired into production search.

use crate::core::board::{Board, Move, Player};
use crate::movegen::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::path::BfsScratch;
use crate::search::alphabeta::eval_stm;

/// Eval→win-probability scale (centipawn-ish). Matches v14.1 `stubValue`/`rprior`
/// which used `1/(1+exp(-sc/400))`.
const EVAL_SCALE: f64 = 400.0;
/// Softmax temperature for move sampling inside a rollout. Lower = greedier.
const TEMP: f64 = 250.0;

/// Sigmoid of a centipawn eval → win probability in (0, 1).
#[inline]
pub fn sigmoid_winprob(eval_cp: i32) -> f64 {
    1.0 / (1.0 + (-(eval_cp as f64) / EVAL_SCALE).exp())
}

/// Tiny deterministic PRNG (xorshift64*) — reproducible rollouts, no `rand` dep,
/// no wasm/`getrandom` friction. Seeded per call so runs are repeatable.
struct Rng(u64);
impl Rng {
    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    /// Uniform f64 in [0, 1).
    #[inline]
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// One soft-minimax playout from `board` (caller's position is preserved via
/// make/unmake). Returns the win probability for `hero`.
fn rollout_once(
    board: &mut Board,
    hero: Player,
    max_plies: u32,
    bfs: &mut BfsScratch,
    rng: &mut Rng,
) -> f64 {
    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let mut undos = Vec::with_capacity(max_plies as usize);

    let mut value = 0.5;
    for _ in 0..max_plies {
        if let Some(winner) = board.is_terminal() {
            value = if winner == hero { 1.0 } else { 0.0 };
            break;
        }
        let n = generate_legal_moves_slice(board, &mut buf, bfs);
        if n == 0 {
            value = sigmoid_winprob(eval_stm(board, hero, bfs));
            break;
        }
        let side = board.side();

        // Score each candidate by the mover's own one-ply static eval.
        let mut evals = [0i32; MAX_LEGAL_MOVES];
        let mut best = i32::MIN;
        for i in 0..n {
            let undo = board.make_move(buf[i]);
            let e = eval_stm(board, side, bfs);
            board.unmake_move(undo);
            evals[i] = e;
            if e > best {
                best = e;
            }
        }

        // Softmax sample (numerically stabilised by subtracting `best`).
        let mut sum = 0.0;
        let mut weights = [0.0f64; MAX_LEGAL_MOVES];
        for i in 0..n {
            let w = (((evals[i] - best) as f64) / TEMP).exp();
            weights[i] = w;
            sum += w;
        }
        let mut pick = 0usize;
        let target = rng.next_f64() * sum;
        let mut acc = 0.0;
        for i in 0..n {
            acc += weights[i];
            if acc >= target {
                pick = i;
                break;
            }
        }

        undos.push(board.make_move(buf[pick]));

        // Leaf fallback evaluated lazily only if the loop exits by exhaustion.
        value = sigmoid_winprob(eval_stm(board, hero, bfs));
    }

    // Unwind to the caller's position.
    while let Some(undo) = undos.pop() {
        board.unmake_move(undo);
    }
    value
}

/// One ranked root move: rollout value `q`, static prior, and visit count.
#[derive(Debug, Clone)]
pub struct RolloutRank {
    pub mv: Move,
    /// Mean rollout win probability for the root side (the "simulated" value).
    pub q: f64,
    /// `sigmoid(eval)` prior right after the move (v14.1 `rprior`).
    pub prior: f64,
    pub sims: u32,
}

/// Rank every legal root move by eval-guided rollouts. The root side is the
/// hero. Returns moves sorted by `q` (then `prior`) descending.
pub fn rollout_rank(board: &mut Board, sims: u32, max_plies: u32, seed: u64) -> Vec<RolloutRank> {
    let mut bfs = BfsScratch::default();
    let hero = board.side();
    let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let n = generate_legal_moves_slice(board, &mut buf, &mut bfs);

    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let mv = buf[i];
        let undo = board.make_move(mv);
        let prior = sigmoid_winprob(eval_stm(board, hero, &mut bfs));
        // Distinct seed per root move so playouts don't all share a stream.
        let mut rng = Rng(seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x1234_5678);
        let mut acc = 0.0;
        for _ in 0..sims {
            acc += rollout_once(board, hero, max_plies, &mut bfs, &mut rng);
        }
        board.unmake_move(undo);
        out.push(RolloutRank {
            mv,
            q: if sims > 0 { acc / sims as f64 } else { prior },
            prior,
            sims,
        });
    }

    out.sort_by(|a, b| {
        b.q.partial_cmp(&a.q)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(
                b.prior
                    .partial_cmp(&a.prior)
                    .unwrap_or(std::cmp::Ordering::Equal),
            )
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_is_monotone_and_centered() {
        assert!((sigmoid_winprob(0) - 0.5).abs() < 1e-9);
        assert!(sigmoid_winprob(1000) > sigmoid_winprob(0));
        assert!(sigmoid_winprob(-1000) < sigmoid_winprob(0));
    }

    #[test]
    fn rollout_restores_board_and_ranks_all_moves() {
        let mut board = Board::new();
        let before = board.hash;
        let ranks = rollout_rank(&mut board, 8, 20, 42);
        assert_eq!(board.hash, before, "rollout must restore the root position");
        // Startpos has 3 pawn moves + 128 walls = 131 legal moves.
        assert_eq!(ranks.len(), 131);
        for r in &ranks {
            assert!(r.q >= 0.0 && r.q <= 1.0);
        }
    }

    #[test]
    fn pawn_advance_is_top_valued_at_startpos() {
        // Soft-minimax rollouts should value advancing toward the goal highest.
        // The forward pawn push e2 is the unique move that shifts the static
        // eval at startpos (walls barely move it), so it should hold the top `q`
        // — but only when sims are high enough to beat tail noise, hence we
        // assert on the *value* (q), not the exact sorted slot.
        let mut board = Board::new();
        let ranks = rollout_rank(&mut board, 64, 24, 1);
        let pawn_q = ranks
            .iter()
            .filter(|r| matches!(r.mv, Move::Pawn { .. }))
            .map(|r| r.q)
            .fold(f64::MIN, f64::max);
        let wall_q = ranks
            .iter()
            .filter(|r| matches!(r.mv, Move::Wall { .. }))
            .map(|r| r.q)
            .fold(f64::MIN, f64::max);
        assert!(
            pawn_q >= wall_q,
            "best pawn q ({pawn_q:.3}) should be >= best wall q ({wall_q:.3})"
        );
    }
}
