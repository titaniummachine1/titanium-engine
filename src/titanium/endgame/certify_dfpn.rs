//! Unrestricted AND/OR win certificate — the completeness half of a proof solver.
//!
//! Why this exists
//! ---------------
//! [`super::certify`] is a 1:1 port of `certify_win.js`. It proves "side S wins"
//! inside a **restricted subgame**: S may only step strictly toward goal (plus
//! one equal-distance re-commit right after each opponent wall). That
//! restriction is what makes it sound — it only removes options from the
//! maximising side, so a PROVEN verdict transfers to the real game — but it is
//! also a hard completeness ceiling: **any win that needs a sideways or
//! backward step is unprovable by construction.**
//!
//! That ceiling is not hypothetical. In arena game 1922 the two good moves in a
//! lost-looking race were `b3` (toward goal) and `a4` (**sideways**). The legacy
//! certifier cannot place `a4` in a winning strategy at all.
//!
//! This module searches the **real legal move set** (`gen_legal_moves`), so it
//! is complete with respect to the node cap rather than with respect to a move
//! restriction.
//!
//! Semantics (matching the barrier-race proof spec)
//! ------------------------------------------------
//! Target predicate: "`side` can force reaching its own goal". Nodes alternate
//! OR (`side` to move — one winning child suffices) and AND (opponent to move —
//! every child must win).
//!
//! A repetition on the **active path** means the win target was not achieved, so
//! it is DISPROVEN *for the prover* — it is **not** a symmetric draw and must
//! never be scored as one. This asymmetry matters: the search's main repetition
//! rule returns a symmetric `0`, which is a different question ("is this
//! position equal?") from the one a certificate asks ("can S force a win?").
//!
//! GHI safety
//! ----------
//! Graph History Interaction is the classic soundness bug for proof search on a
//! loopy graph: a result derived from an ancestor repetition is only valid on
//! the path that produced it, and caching it corrupts other paths. This is a
//! tree search with ancestor-path repetition detection, and the memo stores
//! **only grounded results** — those whose subtree never consumed a repetition
//! verdict. `grounded` is propagated up alongside the value; an ungrounded
//! result is returned to the caller but never cached.
//!
//! Soundness argument
//! ------------------
//! PROVEN means: at every OR state on the returned strategy `side` has a move,
//! and for every opponent reply the same holds, terminating in `side` at goal
//! without any repetition. That is a winning strategy in the real game.
//! DISPROVEN within the cap means only "not proven under these semantics" —
//! treating a repetition as failure makes the prover pessimistic, never
//! optimistic, so a DISPROVEN verdict is *not* a proof of loss.
//!
//! Not yet done: proof-number ordering (df-pn / EWS most-proving-node
//! selection). That is an efficiency multiplier, not a correctness change —
//! this orders by distance progress instead. See the module docs in
//! `barrier-race/engine/search/proof/` for the target design.

use std::collections::HashMap;

use crate::titanium::game::GameState;
use crate::util::clock::Instant;

/// Node cap or wall-clock deadline hit; the verdict is UNKNOWN, not DISPROVEN.
#[derive(Debug, Clone, Copy)]
pub struct Abort;

type NodeResult = Result<(bool, bool), Abort>; // (side_wins, grounded)

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofVal {
    /// `side` has a forced win. Sound: transfers to the real game.
    Proven,
    /// Not provable under these semantics within the cap. NOT a proof of loss.
    Disproven,
    /// Ran out of nodes or time.
    Unknown,
}

#[derive(Debug, Clone)]
pub struct DfpnOpts {
    /// Hard node cap. `certify`'s in-search budget is 1200; offline labelling
    /// can afford orders of magnitude more.
    pub node_cap: u64,
    pub deadline: Option<Instant>,
    /// Side whose win is being certified.
    pub side: usize,
    /// Record the winning move at each OR state on the proven strategy.
    pub want_certificate: bool,
}

impl Default for DfpnOpts {
    fn default() -> Self {
        Self {
            node_cap: 200_000,
            deadline: None,
            side: 0,
            want_certificate: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DfpnReport {
    pub value_proven: bool,
    pub capped: bool,
    pub nodes: u64,
    pub memo_hits: u64,
    /// (hash_lo, hash_hi, winning move) at each OR state on the strategy.
    /// Checkable independently — replay it and confirm every branch.
    pub certificate: Vec<(u32, u32, i16)>,
}

impl DfpnReport {
    pub fn value(&self) -> ProofVal {
        if self.value_proven {
            ProofVal::Proven
        } else if self.capped {
            ProofVal::Unknown
        } else {
            ProofVal::Disproven
        }
    }
}

/// Node budget as a function of the *relationship* between the two wall
/// stocks. Returns 0 to mean "do not call the solver at all".
///
/// Cost here is set by tree width, and tree width is set by how many wall moves
/// each side has. A side holding no walls answers with ~5 pawn moves instead of
/// ~130 wall placements, so half the tree collapses. That makes the WEAKER
/// stock — not the total — the thing that decides whether a proof is reachable.
///
/// Measured over 1,000 corpus positions at a 20k cap (solved% / median nodes):
///
/// ```text
///   split 0-4   100%       34      split 2-2    18%   40,000
///   split 0-5   100%       68      split 2-3    24%   40,000
///   split 1-6    80%    1,410      split 2-4    11%   40,000
///   split 1-5    86%    2,934      split 3-3    13%   40,000
///   split 1-4    55%    4,258      split 3-4    21%   40,000
///   split 1-3    46%   40,000
/// ```
///
/// A 1-6 split (seven walls on the board) resolves in 1,410 nodes; a 2-2 split
/// (four walls) burns the entire cap for an 18% hit rate. Gating on the total
/// would keep the wrong half: it skips 0-7 — the class containing the arena
/// game 1922 loss, provable in 1,364 nodes — while paying full price for 2-2.
///
/// The gap alone is not the signal either; the weaker side's absolute level
/// dominates it. At a constant gap of 5:
///
/// ```text
///   0-5   100% solved       68 nodes
///   1-6    80% solved    1,410 nodes
///   2-7    33% solved   40,000 nodes
/// ```
///
/// So `weaker` selects the band and the gap only refines it inside `weaker == 1`,
/// where the richer side does move the needle (richer 1-3: 33-46% at the full
/// cap; richer 4-6: 55-86% in 1.4-4.3k nodes).
///
/// `weaker >= 2` is refused outright: every such bucket caps out at a hit rate
/// below the legacy prover's own coverage elsewhere, so the nodes buy nothing.
/// Distance to goal is the second axis, and at the margin it separates
/// solvable from hopeless better than the wall stocks do. Proof depth is
/// bounded by how far the winner still has to walk, so a pawn near home has a
/// shallow proof tree regardless of how many walls are in hand.
///
/// Measured hit rate by (weaker stock, distance of the nearer pawn), n=3,000:
///
/// ```text
///            dmin 0-3   4-7    8-11   12+
///   weaker=0     97%    85%     64%    62%
///   weaker=1     91%    55%      2%     0%
///   weaker=2     76%     0%      0%     0%
///   weaker=3     38%     0%      0%     0%
/// ```
///
/// Two things fall out. `weaker >= 2` is NOT uniformly hopeless — close to
/// goal it still lands 38-76%, and a walls-only gate discards all of that.
/// Conversely `weaker = 1` at `dmin >= 8` is a dead loss: 0-2% hit rate for
/// ~222 ms per position. Distance is what tells those apart.
///
/// Cells were fitted on half the sample (keep a cell at >=25% hit rate, n>=15)
/// and scored on the held-out half:
///
/// ```text
///   always run                505/505 proofs   100.0% time
///   this table                499/505 proofs    15.7% time
///   walls-only gate           352/505 proofs     5.9% time
///   oracle (perfect)          505/505 proofs     5.5% time
/// ```
///
/// 99% of the reachable proofs for a 6.4x time cut. The residual gap to the
/// oracle is time, not coverage — see the note on learned predictors below.
///
/// CAVEAT: the table is fitted at a 10k node budget. "Solvable" is a function
/// of the budget, so refit if `base` changes materially.
/// Budget is per-cell, not one flat number. Raising the cap 5x (2k -> 10k) buys
/// +24.6% more proofs for 4.1x the time — but the gain is concentrated in three
/// cells, and everywhere else the small cap is already as good:
///
/// ```text
///   cell              2k     10k    gain
///   w=0 dmin 4-7      69%    86%   +18pp   <- pay
///   w=1 dmin 4-7      10%    55%   +45pp   <- pay
///   w=3 dmin 0-3      27%    41%   +15pp   <- pay
///   w=0 dmin 0-3      95%    98%    +3pp
///   w=1 dmin 0-3      85%    92%    +6pp
///   w=2 dmin 0-3      67%    72%    +5pp
///   everything else    0%     0%     0pp   <- skip
/// ```
///
/// So the deep budget goes only to cells that convert it. Elsewhere a fifth of
/// it reaches the same hit rate, and the remaining cells get nothing at all.
///
/// Two features that look promising and are NOT used, both measured:
/// walls-on-board is identical between solved and unsolved in every cell; and
/// `|d0 - d1|` separates strongly only until the 255 "unreachable" sentinels
/// are removed, after which it flattens (e.g. w=0 dmin 8-11 goes from 16.5 vs
/// 4.7 to 4.7 vs 4.7). Neither earns its place, which also bounds how much a
/// learned budget predictor could add over this table.
pub fn gated_budget(wl0: i32, wl1: i32, d0: i32, d1: i32, base: u64) -> u64 {
    let weaker = wl0.min(wl1);
    let dmin = d0.min(d1);
    // Per-cell budgets were tried and are NOT the default. Giving the shallow
    // cells base/5 and the three depth-converting cells the full cap measured:
    //
    //   gate, single budget      996 proofs (98.4%)   67.8s
    //   gate, per-cell budget    919 proofs (90.8%)   52.3s
    //
    // i.e. 77 proofs surrendered for 23% less time. The +3..+6pp cells look
    // negligible per-cell but sum to real proofs across thousands of positions.
    // Trading coverage for time is only correct once certify is known to be a
    // large share of search time, which has not been measured. Single budget
    // until it is.
    match (weaker, dmin) {
        (0, _) => base,
        (1, 0..=7) => base,
        (2 | 3, 0..=3) => base,
        // Nothing here justifies the nodes at any depth.
        _ => 0,
    }
}

/// Memo key. Wall stocks are part of the position: the same board with
/// different walls-in-hand is a different game.
type MemoKey = (u32, u32, u8, u8, u8);

struct Solver<'a> {
    g: &'a mut GameState,
    side: usize,
    nodes: u64,
    cap: u64,
    deadline: Option<Instant>,
    /// Ancestor hashes, indexed by ply — the repetition frontier.
    path: Vec<(u32, u32)>,
    /// Grounded decisive results only. Never holds a repetition-derived verdict.
    memo: HashMap<MemoKey, bool>,
    memo_hits: u64,
    want_cert: bool,
    cert: Vec<(u32, u32, i16)>,
}

impl<'a> Solver<'a> {
    #[inline]
    fn key(&self) -> MemoKey {
        (
            self.g.hash_lo,
            self.g.hash_hi,
            self.g.turn as u8,
            self.g.wl[0] as u8,
            self.g.wl[1] as u8,
        )
    }

    #[inline]
    fn tick(&mut self) -> Result<(), Abort> {
        self.nodes += 1;
        if self.nodes >= self.cap {
            return Err(Abort);
        }
        // Check the clock rarely — Instant::now() is not free.
        if self.nodes & 0x3FF == 0 {
            if let Some(dl) = self.deadline {
                if Instant::now() >= dl {
                    return Err(Abort);
                }
            }
        }
        Ok(())
    }

    /// Order moves cheaply: pawn moves before walls, and pawn moves by the
    /// distance-to-goal of their destination. At an OR node that tries genuine
    /// progress first; at an AND node it tries the fastest refutation first.
    /// A pawn move's encoding IS its destination cell (< 81), so the ordering
    /// key is a table lookup with no make/unmake.
    fn order(&mut self, moves: &mut [i16], mover: usize) {
        let mut dist = [0u8; 81];
        self.g.compute_dist(mover, &mut dist);
        let n = moves.len();
        let mut keys = vec![0u16; n];
        for i in 0..n {
            let mv = moves[i];
            keys[i] = if (0..81).contains(&mv) {
                dist[mv as usize] as u16
            } else {
                // Walls last: they never shorten the mover's own path.
                1000
            };
        }
        for i in 1..n {
            let (k, m) = (keys[i], moves[i]);
            let mut j = i;
            while j > 0 && keys[j - 1] > k {
                keys[j] = keys[j - 1];
                moves[j] = moves[j - 1];
                j -= 1;
            }
            keys[j] = k;
            moves[j] = m;
        }
    }

    fn solve(&mut self, ply: usize) -> NodeResult {
        self.tick()?;

        // Terminal: decisive and history-independent.
        let w = self.g.winner();
        if w == self.side as i32 {
            return Ok((true, true));
        }
        if w >= 0 {
            return Ok((false, true));
        }

        // Ancestor repetition: the win target was not achieved on this path.
        // Disproven for the prover, and NOT grounded — never cache it.
        let h = (self.g.hash_lo, self.g.hash_hi);
        if self.path[..ply].contains(&h) {
            return Ok((false, false));
        }

        let key = self.key();
        if let Some(&v) = self.memo.get(&key) {
            self.memo_hits += 1;
            return Ok((v, true));
        }

        if ply + 1 >= self.path.len() {
            // Depth guard: treat as unprovable rather than growing without bound.
            return Ok((false, false));
        }
        self.path[ply] = h;

        let mut buf = [0i16; 160];
        let n = self.g.gen_legal_moves(&mut buf);
        if n == 0 {
            return Ok((false, true));
        }
        let mover = self.g.turn;
        let mut moves: Vec<i16> = buf[..n].to_vec();
        self.order(&mut moves, mover);

        let or_node = mover == self.side;
        // OR: one winning child proves it. AND: one losing child refutes it.
        let mut value = !or_node;
        let mut grounded = true;
        let mut decisive_move: i16 = -1;

        for &mv in &moves {
            self.g.make_move(mv);
            let res = self.solve(ply + 1);
            self.g.unmake_move();
            let (child, child_grounded) = res?;

            if or_node && child {
                // Proven by this child: only this child's grounding matters.
                value = true;
                grounded = child_grounded;
                decisive_move = mv;
                break;
            }
            if !or_node && !child {
                // Refuted by this child: only this child's grounding matters.
                value = false;
                grounded = child_grounded;
                break;
            }
            // Inconclusive child: the verdict now leans on all of them.
            grounded &= child_grounded;
        }

        if grounded {
            self.memo.insert(key, value);
        }
        if or_node && value && self.want_cert && decisive_move >= 0 {
            self.cert.push((h.0, h.1, decisive_move));
        }
        Ok((value, grounded))
    }
}

/// Prove that `opts.side` can force a win from `game`.
///
/// The board is restored on every path, including on abort.
pub fn certify_dfpn(game: &mut GameState, opts: &DfpnOpts) -> DfpnReport {
    const MAX_DEPTH: usize = 256;
    let mut solver = Solver {
        g: game,
        side: opts.side,
        nodes: 0,
        cap: opts.node_cap,
        deadline: opts.deadline,
        path: vec![(0, 0); MAX_DEPTH],
        memo: HashMap::new(),
        memo_hits: 0,
        want_cert: opts.want_certificate,
        cert: Vec::new(),
    };
    let outcome = solver.solve(0);
    let (proven, capped) = match outcome {
        Ok((v, _)) => (v, false),
        Err(Abort) => (false, true),
    };
    DfpnReport {
        value_proven: proven,
        capped,
        nodes: solver.nodes,
        memo_hits: solver.memo_hits,
        certificate: solver.cert,
    }
}
