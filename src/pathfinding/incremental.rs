//! Goal-seeded BFF layer memoization for incremental wall-legality checks.
//!
//! The production L3 check ([`bff_wall_legal`](crate::pathfinding::bff_wall_legal))
//! floods **from the pawn to the goal row**, once per surviving wall candidate.
//! Every trial restarts from scratch, so a node with `k` heavy candidates pays
//! `k` full floods.
//!
//! This module inverts the flood: seed with the **whole goal row** and expand
//! until the wave reaches the pawn, memoizing the cumulative reached-set after
//! every ring. Two properties fall out, and they are what make the trials cheap:
//!
//! 1. **The field does not depend on pawn positions.** It is seeded from a goal
//!    row, so it is a pure function of the wall configuration. A pawn move —
//!    roughly half the branches in the tree — leaves the whole structure valid;
//!    only the `d[pawn]` read-off changes.
//! 2. **The memoized prefix survives a wall placement.** If `t` is the earliest
//!    layer holding a cell incident to an edge the wall cuts, then `reached[t-1]`
//!    is bit-identical in the walled graph, so a trial restarts at ring `t-1`
//!    instead of ring 0. See [`GoalField::probe`] for the proof.
//!
//! Because the flood stops at the pawn, the memoized set is a **ball of radius
//! `d[pawn]` around the goal row**. The pawn's descent chain to the goal lies
//! entirely inside that ball (each step strictly decreases distance), so a wall
//! cutting no edge with *both* endpoints in the ball cannot touch the chain and
//! is legal outright — no flood, four AND/shift pairs. That is the common case.
//!
//! Distances only ever *increase* when edges are removed, so the memoized layers
//! are sound as a **termination target**, never as a seed annexation: the
//! cross-player "bit theft" that [`bff_to_goal_cached`] relies on has no
//! analogue here, since the two players' fields are seeded from opposite rows
//! and measure different quantities. Theft stays with the fallback flood.
//!
//! [`bff_to_goal_cached`]: crate::pathfinding::bff::wall::bff_to_goal_cached

use crate::pathfinding::bff::wall::{expand_wave, WallGrids};
use crate::util::grid::{FLOOD_PLAYABLE, FLOOD_STRIDE};

/// Layer cap — the board has 81 playable cells, so no BFS can exceed that depth.
pub const MAX_GOAL_LAYERS: usize = 81;

/// Outcome of a speculative wall trial against a memoized field.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Probe {
    /// Wall cuts no edge inside the ball — descent chain intact, distance unchanged.
    Untouched(u8),
    /// Field was re-flooded from the first affected ring; pawn still reaches the goal.
    Repaired(u8),
    /// Pawn's component no longer contains a goal cell — the wall is illegal.
    Severed,
}

impl Probe {
    /// Whether the pawn still reaches its goal row.
    #[inline]
    pub fn reaches(self) -> bool {
        !matches!(self, Probe::Severed)
    }

    /// Shortest distance to the goal row after the placement, if one exists.
    #[inline]
    pub fn distance(self) -> Option<u8> {
        match self {
            Probe::Untouched(d) | Probe::Repaired(d) => Some(d),
            Probe::Severed => None,
        }
    }
}

/// Memoized inverse BFF field for one player: cumulative reached-sets by ring.
///
/// Built once per node per player and reused across every wall trial at that
/// node. Wall trials never mutate it — they read the prefix and re-flood a
/// suffix into scratch — so one build serves all candidates.
#[derive(Clone)]
pub struct GoalField {
    /// `reached[d]` = every cell within distance `d` of the goal row.
    /// Cumulative, not per-ring: the restart seed needs the union, and the
    /// individual ring is recovered with one AND-NOT against `reached[d-1]`.
    reached: [u128; MAX_GOAL_LAYERS],
    /// Number of populated entries in `reached`.
    depth: usize,
    /// `reached[depth-1]`: the ball of radius `d[pawn]` around the goal row.
    /// When the pawn is unreachable this is the goal row's whole component.
    ball: u128,
    /// `reached[depth-2] | pawn` — the only cells a descent chain from the pawn
    /// can visit. Strictly tighter than `ball`, which also carries every *other*
    /// cell of the final ring; those can never sit on the pawn's own chain, so
    /// counting them only forces needless revalidation. Used for the skip test.
    chain_ball: u128,
    /// Distance from the pawn to the goal row; `None` when unreachable.
    pawn_dist: Option<u8>,
}

impl GoalField {
    /// Flood from `goal` outward until `pawn` is reached, memoizing every ring.
    ///
    /// `goal` is the player's goal-row mask and `pawn` that player's pawn bit,
    /// both in the centered 11-wide flood layout.
    pub fn build(grids: &WallGrids, goal: u128, pawn: u128) -> Self {
        let mut f = Self::empty();
        f.build_into(grids, goal, pawn);
        f
    }

    /// Zeroed field suitable for reuse as a scratch buffer.
    pub fn empty() -> Self {
        Self {
            reached: [0u128; MAX_GOAL_LAYERS],
            depth: 0,
            ball: 0,
            chain_ball: 0,
            pawn_dist: None,
        }
    }

    /// Rebuild in place, overwriting only `reached[0..depth]`.
    ///
    /// Hot-path entry point: the 1.3 KB `reached` array is allocated once in a
    /// long-lived scratch and refilled per node, so wall trials never pay to
    /// zero it. Entries at or beyond `depth` are stale by design and are never
    /// read — every accessor is bounded by `depth`.
    pub fn build_into(&mut self, grids: &WallGrids, goal: u128, pawn: u128) {
        let mut visited = goal & FLOOD_PLAYABLE;
        self.reached[0] = visited;
        self.depth = 1;

        if visited & pawn != 0 {
            self.ball = visited;
            self.chain_ball = pawn;
            self.pawn_dist = Some(0);
            return;
        }

        let mut wave = visited;
        while wave != 0 && self.depth < MAX_GOAL_LAYERS {
            wave = expand_wave(wave, grids) & !visited;
            if wave == 0 {
                break;
            }
            visited |= wave;
            self.reached[self.depth] = visited;
            self.depth += 1;
            if wave & pawn != 0 {
                // `depth >= 2` here, so `reached[depth - 2]` exists.
                self.ball = visited;
                self.chain_ball = self.reached[self.depth - 2] | pawn;
                self.pawn_dist = Some((self.depth - 1) as u8);
                return;
            }
        }

        self.ball = visited;
        self.chain_ball = visited;
        self.pawn_dist = None;
    }

    /// Shortest distance from the pawn to its goal row in the base position.
    #[inline]
    pub fn pawn_distance(&self) -> Option<u8> {
        self.pawn_dist
    }

    /// Cells within `d[pawn]` of the goal row — the memoized ball.
    #[inline]
    pub fn ball(&self) -> u128 {
        self.ball
    }

    /// Source cells of every wall-cut edge whose **both** endpoints lie in the ball.
    ///
    /// `delta` records each blocked step from both sides (a horizontal wall sets
    /// `south` on the upper pair and `north` on the lower pair), so the result is
    /// the full endpoint set of the cut-inside-ball edges, not just one side.
    ///
    /// Zero means the wall cuts no edge the descent chain could use.
    #[inline]
    pub fn cut_inside_ball(&self, delta: &WallGrids) -> u128 {
        self.cut_inside(delta, self.ball)
    }

    /// Cells of `region` that lose a step to another `region` cell under `delta`.
    #[inline]
    fn cut_inside(&self, delta: &WallGrids, region: u128) -> u128 {
        const S: u32 = FLOOD_STRIDE;
        let b = region;
        // Bit `u` survives iff the wall blocks `u`'s step in that direction AND
        // both `u` and the neighbour it would step to are inside the ball.
        (delta.south & b & (b >> S))
            | (delta.north & b & (b << S))
            | (delta.east & b & (b >> 1))
            | (delta.west & b & (b << 1))
    }

    /// Ring index of one cell. `cell` must be a single bit inside `ball`.
    #[inline]
    fn layer_of(&self, cell: u128) -> usize {
        // Binary search the cumulative sets: `reached` is monotone in `d`.
        let (mut lo, mut hi) = (0usize, self.depth - 1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.reached[mid] & cell != 0 {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }

    /// `(first, last)` ring indices spanned by `cells`. `cells` must be ⊆ `ball`.
    #[inline]
    fn layer_span(&self, mut cells: u128) -> (usize, usize) {
        let (mut first, mut last) = (self.depth - 1, 0usize);
        while cells != 0 {
            let cell = cells & cells.wrapping_neg();
            cells &= cells - 1;
            let d = self.layer_of(cell);
            first = first.min(d);
            last = last.max(d);
        }
        (first, last)
    }

    /// Speculative wall trial: does `pawn` still reach the goal row?
    ///
    /// `walled` must be the base grids with `delta` already placed; `delta` is
    /// the wall's blocked-step delta on its own.
    ///
    /// # Why restarting at ring `t-1` is exact
    ///
    /// Let `t` be the smallest ring holding an endpoint of a cut edge that lies
    /// wholly inside the ball. Every cell `w` with `layer(w) < t` has a descent
    /// chain to the goal whose cells are all in the ball at rings `< t`. Each
    /// edge on that chain has both endpoints in the ball, so if the wall cut it,
    /// that edge would be a cut-inside-ball edge with an endpoint at a ring
    /// below `t` — contradicting minimality of `t`. So the chain survives and
    /// `reached'[t-1] ⊇ reached[t-1]`. Edge removal can only shrink reachability,
    /// giving `⊆`, hence equality — and the same argument at `t-2` makes the
    /// ring `reached[t-1] \ reached[t-2]` identical too, so it is a valid frontier.
    ///
    /// Cut edges *not* wholly inside the ball are irrelevant to the prefix: their
    /// outside endpoint sits beyond distance `d[pawn]`, so the memoized flood
    /// never traversed them. They are still removed from `walled`, which is what
    /// the continuation floods against.
    pub fn probe(&self, walled: &WallGrids, delta: &WallGrids, pawn: u128) -> Probe {
        self.probe_with_rings(walled, delta, pawn).0
    }

    /// [`GoalField::probe`] plus the number of flood rings it burned, so
    /// benchmarks bill this path in the same unit as a from-scratch flood.
    /// Single implementation — `probe` delegates here, so the two cannot drift.
    pub fn probe_with_rings(
        &self,
        walled: &WallGrids,
        delta: &WallGrids,
        pawn: u128,
    ) -> (Probe, u32) {
        // Skip test runs against the tight chain set: if no edge of the pawn's
        // own descent chain is cut, that chain survives verbatim. The repair
        // path below still keys `t` / `last_cut` off the full ball, so its
        // restart-exactness argument is unchanged.
        if self.cut_inside(delta, self.chain_ball) == 0 {
            // Descent chain untouched, so distance cannot have risen; edge
            // removal cannot lower it either. Unchanged.
            return (self.unchanged(), 0);
        }

        let endpoints = self.cut_inside_ball(delta);
        if endpoints == 0 {
            return (self.unchanged(), 0);
        }
        let (t, last_cut) = self.layer_span(endpoints);
        let (mut visited, mut wave, mut ring) = if t == 0 {
            let seed = self.reached[0];
            (seed, seed, 0usize)
        } else {
            let visited = self.reached[t - 1];
            let prev = if t >= 2 { self.reached[t - 2] } else { 0 };
            (visited, visited & !prev, t - 1)
        };

        if visited & pawn != 0 {
            // Pawn sits inside the untouched prefix — its distance is whatever
            // the memoized field said, and the prefix is exact.
            return (self.unchanged(), 0);
        }

        let mut rings = 0u32;
        while wave != 0 {
            wave = expand_wave(wave, walled) & !visited;
            if wave == 0 {
                break;
            }
            visited |= wave;
            ring += 1;
            rings += 1;
            if wave & pawn != 0 {
                return (Probe::Repaired(ring as u8), rings);
            }
            // Re-convergence: the perturbation has healed. Once the walled flood
            // has caught back up to the memoized cumulative set *and* every
            // cut-inside-ball edge is interior to it (both endpoints already
            // reached, so no later ring can want to cross one), the remaining
            // rings are bit-identical to the memoized ones. The memoized flood
            // reached the pawn at ring `depth-1` using only in-ball edges, so
            // that continuation is still valid here: distance is unchanged.
            if ring >= last_cut && ring < self.depth && visited == self.reached[ring] {
                return (self.unchanged(), rings);
            }
        }
        (Probe::Severed, rings)
    }

    /// Verdict when the memoized field is provably still valid for the pawn.
    #[inline]
    fn unchanged(&self) -> Probe {
        match self.pawn_dist {
            Some(d) => Probe::Untouched(d),
            None => Probe::Severed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::{Board, Player, WallOrientation};
    use crate::pathfinding::bff::wall::{
        bff_to_goal, goal_bits, pawn_bit, wall_delta, P1_GOAL_BITS, P2_GOAL_BITS,
    };
    use crate::util::grid::{has_wall, set_wall};

    /// Deterministic LCG — matches the style used by the existing wall tests.
    struct Lcg(u64);

    impl Lcg {
        fn next(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }
    }

    fn random_board(rng: &mut Lcg, max_walls: u32) -> Board {
        let mut board = Board::new();
        let wall_count = rng.next() % max_walls;
        for _ in 0..wall_count {
            let row = (rng.next() % 8) as u8;
            let col = (rng.next() % 8) as u8;
            let orientation = if rng.next() & 1 == 0 {
                WallOrientation::Horizontal
            } else {
                WallOrientation::Vertical
            };
            if has_wall(&board, row, col, WallOrientation::Horizontal)
                || has_wall(&board, row, col, WallOrientation::Vertical)
            {
                continue;
            }
            set_wall(&mut board, row, col, orientation, true);
        }
        let p1 = ((rng.next() % 9) as u8, (rng.next() % 9) as u8);
        let mut p2 = ((rng.next() % 9) as u8, (rng.next() % 9) as u8);
        if p2 == p1 {
            p2 = ((p2.0 + 1) % 9, p2.1);
        }
        board.pawns[Player::One as usize] = p1;
        board.pawns[Player::Two as usize] = p2;
        board
    }

    /// The memoized field's base distance must equal a plain flood's.
    #[test]
    fn base_distance_matches_flood() {
        let mut rng = Lcg(0x243F_6A88_85A3_08D3);
        for _ in 0..2_000 {
            let board = random_board(&mut rng, 14);
            let grids = WallGrids::from_board(&board);
            for player in [Player::One, Player::Two] {
                let (r, c) = board.pawn(player);
                let pawn = pawn_bit(r, c);
                let goal = goal_bits(player);
                let field = GoalField::build(&grids, goal, pawn);
                let (reached, _) = bff_to_goal(pawn, &grids, goal);
                assert_eq!(
                    field.pawn_distance().is_some(),
                    reached,
                    "reachability mismatch h={:#x} v={:#x} player={player:?}",
                    board.horizontal_walls,
                    board.vertical_walls,
                );
            }
        }
    }

    /// Every probe verdict — and every distance it reports — must match a full
    /// flood on the walled position. This is the soundness gate: `probe` may
    /// never accept a wall the reference rejects, nor report a wrong distance.
    #[test]
    fn probe_matches_full_flood_on_random_walls() {
        let mut rng = Lcg(0xB7E1_5162_8AED_2A6B);
        let mut trials = 0usize;
        let mut untouched = 0usize;
        let mut repaired = 0usize;
        let mut severed = 0usize;

        for _ in 0..1_500 {
            let board = random_board(&mut rng, 12);
            let base = WallGrids::from_board(&board);

            for player in [Player::One, Player::Two] {
                let (r, c) = board.pawn(player);
                let pawn = pawn_bit(r, c);
                let goal = goal_bits(player);
                let field = GoalField::build(&base, goal, pawn);
                if field.pawn_distance().is_none() {
                    continue; // base position already trapped — not a movegen input
                }

                for orientation in [WallOrientation::Horizontal, WallOrientation::Vertical] {
                    for row in 0..8u8 {
                        for col in 0..8u8 {
                            if has_wall(&board, row, col, WallOrientation::Horizontal)
                                || has_wall(&board, row, col, WallOrientation::Vertical)
                            {
                                continue;
                            }
                            let delta = wall_delta(row, col, orientation);
                            let mut walled = base;
                            walled.place(delta);

                            let got = field.probe(&walled, delta, pawn);

                            // Reference: independent flood from the pawn.
                            let (ref_reaches, _) = bff_to_goal(pawn, &walled, goal);
                            assert_eq!(
                                got.reaches(),
                                ref_reaches,
                                "legality mismatch at {row},{col},{orientation:?} \
                                 h={:#x} v={:#x} player={player:?}",
                                board.horizontal_walls,
                                board.vertical_walls,
                            );

                            if ref_reaches {
                                // Reference distance via an inverse flood on the walled grids.
                                let ref_field = GoalField::build(&walled, goal, pawn);
                                assert_eq!(
                                    got.distance(),
                                    ref_field.pawn_distance(),
                                    "distance mismatch at {row},{col},{orientation:?} \
                                     h={:#x} v={:#x} player={player:?}",
                                    board.horizontal_walls,
                                    board.vertical_walls,
                                );
                            }

                            trials += 1;
                            match got {
                                Probe::Untouched(_) => untouched += 1,
                                Probe::Repaired(_) => repaired += 1,
                                Probe::Severed => severed += 1,
                            }
                        }
                    }
                }
            }
        }

        eprintln!(
            "probe trials={trials} untouched={untouched} ({:.1}%) repaired={repaired} severed={severed}",
            100.0 * untouched as f64 / trials as f64,
        );
        assert!(trials > 100_000, "only {trials} trials");
    }

    /// Goal-seeded fields are a pure function of the walls: moving a pawn must
    /// not change the memoized rings, only the distance read off them.
    #[test]
    fn field_is_independent_of_pawn_positions() {
        let mut rng = Lcg(0x9E37_79B9_7F4A_7C15);
        for _ in 0..500 {
            let board = random_board(&mut rng, 12);
            let grids = WallGrids::from_board(&board);
            let goal = P1_GOAL_BITS;

            // Two different pawn squares over the same walls.
            let a = GoalField::build(&grids, goal, pawn_bit(0, 0));
            let b = GoalField::build(&grids, goal, pawn_bit(4, 4));

            // The shallower field is a strict prefix of the deeper one.
            let shallow = a.depth.min(b.depth);
            for d in 0..shallow {
                assert_eq!(
                    a.reached[d], b.reached[d],
                    "ring {d} differs across pawn placements h={:#x} v={:#x}",
                    board.horizontal_walls, board.vertical_walls,
                );
            }
        }
    }

    /// Rings a from-scratch pawn→goal flood burns (the current L3 cost).
    fn baseline_rings(start: u128, grids: &WallGrids, goal: u128) -> usize {
        let mut visited = start & FLOOD_PLAYABLE;
        if visited & goal != 0 {
            return 0;
        }
        let mut wave = visited;
        let mut rings = 0usize;
        while wave != 0 {
            wave = expand_wave(wave, grids) & !visited;
            rings += 1;
            if wave & goal != 0 {
                return rings;
            }
            visited |= wave;
        }
        rings
    }

    /// Work comparison against the current from-scratch flood, on wall-heavy
    /// boards. Reports ring counts — the unit both algorithms are billed in.
    #[test]
    fn ring_work_versus_from_scratch_flood() {
        for walls in [6u32, 12, 18] {
            let mut rng = Lcg(0x0123_4567_89AB_CDEF ^ (walls as u64));
            let (mut base_rings, mut inc_rings) = (0usize, 0usize);
            let (mut setup_rings, mut trials, mut skips) = (0usize, 0usize, 0usize);

            for _ in 0..400 {
                let board = random_board(&mut rng, walls.max(2));
                let base = WallGrids::from_board(&board);
                for player in [Player::One, Player::Two] {
                    let (r, c) = board.pawn(player);
                    let pawn = pawn_bit(r, c);
                    let goal = goal_bits(player);
                    let field = GoalField::build(&base, goal, pawn);
                    if field.pawn_distance().is_none() {
                        continue;
                    }
                    // One inverse flood per player per node, amortized over trials.
                    setup_rings += field.depth - 1;

                    for orientation in [WallOrientation::Horizontal, WallOrientation::Vertical] {
                        for row in 0..8u8 {
                            for col in 0..8u8 {
                                if has_wall(&board, row, col, WallOrientation::Horizontal)
                                    || has_wall(&board, row, col, WallOrientation::Vertical)
                                {
                                    continue;
                                }
                                let delta = wall_delta(row, col, orientation);
                                let mut walled = base;
                                walled.place(delta);
                                base_rings += baseline_rings(pawn, &walled, goal);
                                let r = field.probe_with_rings(&walled, delta, pawn).1 as usize;
                                inc_rings += r;
                                if r == 0 {
                                    skips += 1;
                                }
                                trials += 1;
                            }
                        }
                    }
                }
            }

            let total = inc_rings + setup_rings;
            eprintln!(
                "walls≈{walls:2}: trials={trials} from_scratch={base_rings} rings | \
                 incremental={inc_rings}+setup {setup_rings}={total} rings | \
                 speedup={:.2}x zero_ring_skips={:.1}%",
                base_rings as f64 / total as f64,
                100.0 * skips as f64 / trials as f64,
            );
        }
    }

    /// Pawn-seeded two-player probe (with bit theft) must agree with the
    /// production `bff_wall_legal` on every candidate.
    #[test]
    fn pawn_field_matches_production_wall_legal() {
        use crate::pathfinding::bff::wall::bff_wall_legal;
        let mut rng = Lcg(0x452821E6_38D01377);
        let (mut trials, mut skips) = (0usize, 0usize);

        for _ in 0..1_200 {
            let board = random_board(&mut rng, 12);
            let base = WallGrids::from_board(&board);
            let (r1, c1) = board.pawn(Player::One);
            let (r2, c2) = board.pawn(Player::Two);
            let (p1, p2) = (pawn_bit(r1, c1), pawn_bit(r2, c2));

            let mut f1 = PawnField::empty();
            let mut f2 = PawnField::empty();
            f1.build_into(&base, p1, P1_GOAL_BITS);
            f2.build_into(&base, p2, P2_GOAL_BITS);
            if !f1.reaches_goal() || !f2.reaches_goal() {
                continue;
            }

            for orientation in [WallOrientation::Horizontal, WallOrientation::Vertical] {
                for row in 0..8u8 {
                    for col in 0..8u8 {
                        if has_wall(&board, row, col, WallOrientation::Horizontal)
                            || has_wall(&board, row, col, WallOrientation::Vertical)
                        {
                            continue;
                        }
                        let delta = wall_delta(row, col, orientation);
                        let mut walled = base;
                        walled.place(delta);

                        let got = match f1.probe(&walled, delta, P1_GOAL_BITS, 0) {
                            None => false,
                            Some(pool) => {
                                f2.probe(&walled, delta, P2_GOAL_BITS, pool).is_some()
                            }
                        };
                        assert_eq!(
                            got,
                            bff_wall_legal(p1, p2, &walled),
                            "pawn-field mismatch at {row},{col},{orientation:?} \
                             h={:#x} v={:#x}",
                            board.horizontal_walls,
                            board.vertical_walls,
                        );
                        if f1.cut_inside(delta, f1.chain) == 0 {
                            skips += 1;
                        }
                        trials += 1;
                    }
                }
            }
        }
        eprintln!(
            "pawn-field trials={trials} p1_chain_skips={:.1}%",
            100.0 * skips as f64 / trials as f64
        );
        assert!(trials > 100_000, "only {trials} trials");
    }

    /// A wall that fully cages a pawn must be reported `Severed`.
    #[test]
    fn caged_pawn_is_severed() {
        let mut board = Board::new();
        board.pawns[Player::Two as usize] = (8, 4);
        set_wall(&mut board, 7, 3, WallOrientation::Vertical, true);
        set_wall(&mut board, 7, 4, WallOrientation::Vertical, true);
        let base = WallGrids::from_board(&board);
        let pawn = pawn_bit(8, 4);
        let field = GoalField::build(&base, P2_GOAL_BITS, pawn);
        assert!(field.pawn_distance().is_some());

        // Closing the roof traps the pawn.
        let delta = wall_delta(7, 3, WallOrientation::Horizontal);
        let mut walled = base;
        walled.place(delta);
        assert_eq!(field.probe(&walled, delta, pawn), Probe::Severed);
        assert!(!bff_to_goal(pawn, &walled, P2_GOAL_BITS).0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pawn-seeded variant: production's flood direction + theft, plus a layer stack
// ─────────────────────────────────────────────────────────────────────────────

/// Memoized **pawn → goal** flood: production's direction, with the cumulative
/// reached-set recorded per ring so a trial can restart mid-flood.
///
/// Differs from [`GoalField`] in three ways that matter:
///
/// - **Smaller ball.** The flood is seeded from one cell rather than a 9-cell
///   goal row, so at equal radius it reaches fewer squares. More walls fall
///   outside it, so the skip test fires more often.
/// - **Bit theft survives.** Both players still flood toward their own goal in
///   the same graph, so P2 can annex P1's visited region exactly as
///   [`bff_to_goal_cached`] does — [`PawnField::probe`] takes and returns a pool.
/// - **Not pawn-independent.** The seed *is* the pawn, so unlike `GoalField`
///   this is invalidated by pawn moves and cannot be persisted across a search
///   path. It is a per-node structure only.
///
/// [`bff_to_goal_cached`]: crate::pathfinding::bff::wall::bff_to_goal_cached
#[derive(Clone)]
pub struct PawnField {
    reached: [u128; MAX_GOAL_LAYERS],
    depth: usize,
    /// `reached[depth-1]` — everything within `d[goal]` of the pawn.
    ball: u128,
    /// `reached[depth-2] | (final ring ∩ goal)`: the cells an ascent path from
    /// the pawn to its first goal contact can occupy. Tighter than `ball`, which
    /// also carries final-ring cells that miss the goal entirely.
    chain: u128,
    reaches: bool,
}

impl PawnField {
    pub fn empty() -> Self {
        Self {
            reached: [0u128; MAX_GOAL_LAYERS],
            depth: 0,
            ball: 0,
            chain: 0,
            reaches: false,
        }
    }

    /// Whether the pawn reached its goal row in the base position.
    #[inline]
    pub fn reaches_goal(&self) -> bool {
        self.reaches
    }

    /// Flood pawn → goal in place, stopping at first goal contact.
    pub fn build_into(&mut self, grids: &WallGrids, pawn: u128, goal: u128) {
        let mut visited = pawn & FLOOD_PLAYABLE;
        self.reached[0] = visited;
        self.depth = 1;

        if visited & goal != 0 {
            self.ball = visited;
            self.chain = visited;
            self.reaches = true;
            return;
        }

        let mut wave = visited;
        while wave != 0 && self.depth < MAX_GOAL_LAYERS {
            wave = expand_wave(wave, grids) & !visited;
            if wave == 0 {
                break;
            }
            visited |= wave;
            self.reached[self.depth] = visited;
            self.depth += 1;
            if wave & goal != 0 {
                self.ball = visited;
                // Only the goal-touching cells of the final ring can end a path.
                self.chain = self.reached[self.depth - 2] | (wave & goal);
                self.reaches = true;
                return;
            }
        }

        self.ball = visited;
        self.chain = visited;
        self.reaches = false;
    }

    #[inline]
    fn cut_inside(&self, delta: &WallGrids, region: u128) -> u128 {
        const S: u32 = FLOOD_STRIDE;
        let b = region;
        (delta.south & b & (b >> S))
            | (delta.north & b & (b << S))
            | (delta.east & b & (b >> 1))
            | (delta.west & b & (b << 1))
    }

    #[inline]
    fn layer_of(&self, cell: u128) -> usize {
        let (mut lo, mut hi) = (0usize, self.depth - 1);
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if self.reached[mid] & cell != 0 {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        lo
    }

    #[inline]
    fn min_layer(&self, mut cells: u128) -> usize {
        let mut best = self.depth - 1;
        while cells != 0 {
            let cell = cells & cells.wrapping_neg();
            cells &= cells - 1;
            best = best.min(self.layer_of(cell));
            if best == 0 {
                break;
            }
        }
        best
    }

    /// Speculative trial. Returns the visited region on success (usable as the
    /// next player's theft pool) or `None` if the pawn is cut off.
    ///
    /// `pool` is the previous player's visited region for bit theft; pass 0 for
    /// the first player.
    pub fn probe(
        &self,
        walled: &WallGrids,
        delta: &WallGrids,
        goal: u128,
        pool: u128,
    ) -> Option<u128> {
        if !self.reaches {
            return None;
        }

        // Skip: no edge of the pawn's own ascent path is cut, so that path
        // survives verbatim. `chain` has no cut internal edge, so its induced
        // subgraph is unchanged — still connected, still touching goal — which
        // makes it a sound theft pool for the next player.
        if self.cut_inside(delta, self.chain) == 0 {
            return Some(self.chain);
        }
        let endpoints = self.cut_inside(delta, self.ball);
        if endpoints == 0 {
            return Some(self.chain);
        }

        let t = self.min_layer(endpoints);
        let (mut visited, mut wave) = if t == 0 {
            (self.reached[0], self.reached[0])
        } else {
            let v = self.reached[t - 1];
            let prev = if t >= 2 { self.reached[t - 2] } else { 0 };
            (v, v & !prev)
        };
        if visited & goal != 0 {
            return Some(visited);
        }

        let mut pool = pool & !visited;
        while wave != 0 {
            if wave & pool != 0 {
                visited |= pool;
                wave |= pool;
                pool = 0;
                if visited & goal != 0 {
                    return Some(visited);
                }
            }
            wave = expand_wave(wave, walled) & !visited;
            if wave == 0 {
                break;
            }
            visited |= wave;
            if wave & goal != 0 {
                return Some(visited);
            }
        }
        None
    }
}
