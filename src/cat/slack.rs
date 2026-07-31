//! Slack-plane CAT: bit-parallel corridor attention with no Lee wave.
//!
//! CATv5 answers "how much does this square matter to a player's route" by
//! extracting four node-disjoint shortest paths and smearing a Lee wave outward
//! from each. Measured (`cat::build::tests::catv5_build_cost_decomposition`),
//! the wave is 53.6% of the build and the path extraction another 17.3% — and
//! neither is incrementally stable, since a single wall can reroute all four
//! paths wholesale.
//!
//! This computes the same underlying signal directly. For a player with
//! shortest distance `s`:
//!
//! ```text
//! slack[sq] = d_from_pawn[sq] + d_to_goal[sq] - s
//! ```
//!
//! `slack == 0` is exactly the union of all shortest paths (strictly better
//! information than four sampled disjoint ones); `slack == k` is the set of
//! squares reachable on a route `k` steps longer than optimal. The slack field
//! *is* the smoothing, so the wave disappears.
//!
//! It is also bit-parallel. A cell at from-layer `i` and to-layer `j` has slack
//! `i + j - s`, so each plane is a union of layer intersections:
//!
//! ```text
//! plane[k] = ⋃_i  from[i] & to[s + k - i]
//! ```
//!
//! ~4×depth AND/OR ops per player, no per-cell scatter, no bit popping.
//!
//! # Incrementality
//!
//! The two inputs have different invalidation rules, which is the whole point:
//!
//! - `d_to_goal` is seeded from a goal row, so it is a **pure function of the
//!   walls** — a pawn move leaves it valid. Wall moves repair it incrementally
//!   ([`crate::pathfinding::incremental::GoalField`]).
//! - `d_from_pawn` is seeded from the pawn, so it changes on that player's pawn
//!   move — but only that player's, and it is one flood.
//!
//! So a pawn move recomputes one flood for one player; the opponent's planes and
//! both to-goal fields are untouched.
//!
//! # Output
//!
//! Four binary `u128` planes per player. Binary planes are the natural NNUE
//! input: accumulator updates are add/sub of weight rows, replacing the f64
//! multiply-and-divide loop the CATv5 heat array forces.

use crate::pathfinding::bfs::layers::DistLayers;
use crate::util::grid::FLOOD_PLAYABLE;

/// Slack levels kept. Level `k` holds squares on a route `k` steps longer than
/// optimal; beyond 3 the signal is mostly noise for a 9×9 board.
pub const SLACK_PLANES: usize = 4;

/// Corridor-attention planes for one player.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SlackPlanes {
    /// `planes[k]` = squares with slack exactly `k`. Disjoint by construction.
    pub planes: [u128; SLACK_PLANES],
    /// Shortest distance from pawn to goal row; `u8::MAX` when unreachable.
    pub shortest: u8,
}

impl Default for SlackPlanes {
    fn default() -> Self {
        Self {
            planes: [0; SLACK_PLANES],
            shortest: u8::MAX,
        }
    }
}

impl SlackPlanes {
    /// Union of all levels — every square on any route within `SLACK_PLANES-1`
    /// steps of optimal.
    #[inline]
    pub fn corridor(&self) -> u128 {
        self.planes.iter().fold(0u128, |a, p| a | p)
    }

    /// Whether the player has any route at all.
    #[inline]
    pub fn reachable(&self) -> bool {
        self.shortest != u8::MAX
    }
}

/// Build slack planes from a forward (pawn-seeded) and inverse (goal-seeded)
/// layer decomposition.
///
/// `from_pawn.masks[i]` must be the cells at distance exactly `i` from the pawn,
/// and `to_goal.masks[j]` the cells at distance exactly `j` from the goal row.
/// `pawn` is the pawn's flood bit.
pub fn build_slack_planes(
    from_pawn: &DistLayers,
    to_goal: &DistLayers,
    pawn: u128,
) -> SlackPlanes {
    build_slack_planes_raw(
        &from_pawn.masks,
        from_pawn.depth,
        &to_goal.masks,
        to_goal.depth,
        pawn,
    )
}

/// Slice form, for callers holding bare layer arrays.
///
/// The search already maintains the to-goal layers per player in
/// `d0_layers`/`d1_layers` (filled by `refresh_dist` and LRU-cached), so it can
/// pass those straight in and pay only for the from-pawn flood.
pub fn build_slack_planes_raw(
    from_masks: &[u128],
    from_depth: usize,
    to_masks: &[u128],
    to_depth: usize,
    pawn: u128,
) -> SlackPlanes {
    // `s` = the pawn's own distance to the goal row.
    let Some(s) = (0..to_depth).find(|&d| to_masks[d] & pawn != 0) else {
        return SlackPlanes::default();
    };

    let mut planes = [0u128; SLACK_PLANES];
    for (k, plane) in planes.iter_mut().enumerate() {
        // slack k ⇔ i + j == s + k. Walk i and read the matching j.
        let target = s + k;
        let lo = target.saturating_sub(to_depth.saturating_sub(1));
        let hi = target.min(from_depth.saturating_sub(1));
        let mut acc = 0u128;
        for i in lo..=hi {
            let j = target - i;
            if j < to_depth {
                acc |= from_masks[i] & to_masks[j];
            }
        }
        *plane = acc & FLOOD_PLAYABLE;
    }

    SlackPlanes {
        planes,
        shortest: s as u8,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::{Board, Move, Player, WallOrientation};
    use crate::movegen::legal::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
    use crate::pathfinding::bfs::layers::{fill_dist_layers_from_sq, fill_dist_layers_to_goal_row};
    use crate::pathfinding::bfs::layers::{fill_dist_from_sq, fill_dist_to_goal_row};
    use crate::pathfinding::masks::DirMasks;
    use crate::pathfinding::BfsScratch;
    use crate::util::grid::{square_index, FLOOD_BIT_BY_SQ};

    fn corpus(games: usize, plies: usize) -> Vec<Board> {
        let mut seed = 0x2545_F491_4F6C_DD1Du64;
        let mut out = Vec::new();
        for _ in 0..games {
            let mut board = Board::new();
            for _ in 0..plies {
                if board.is_terminal().is_some() {
                    break;
                }
                let mut sc = BfsScratch::new();
                let mut mv = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
                let n = generate_legal_moves_slice(&mut board, &mut mv, &mut sc);
                if n == 0 {
                    break;
                }
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                let _ = board.make_move(mv[(seed as usize) % n]);
            }
            out.push(board);
        }
        out
    }

    /// The bit-parallel planes must equal slack computed from scalar distance
    /// fields, square for square. This is the correctness gate.
    #[test]
    fn planes_match_scalar_slack_field() {
        let mut checked = 0usize;
        for board in corpus(120, 30) {
            let masks = DirMasks::from_board(&board);
            for player in [Player::One, Player::Two] {
                let (r, c) = board.pawn(player);
                let start = square_index(r, c);
                let pawn = FLOOD_BIT_BY_SQ[start as usize];

                let mut from = DistLayers::default();
                let mut to = DistLayers::default();
                fill_dist_layers_from_sq(start, masks, &mut from);
                fill_dist_layers_to_goal_row(player, masks, &mut to);
                let got = build_slack_planes(&from, &to, pawn);

                // Scalar reference.
                let mut fwd = [0u8; 81];
                let mut inv = [0u8; 81];
                fill_dist_from_sq(start, masks, &mut fwd);
                fill_dist_to_goal_row(player, masks, &mut inv);
                let s = inv[start as usize];
                if s == u8::MAX {
                    assert!(!got.reachable(), "unreachable pawn must yield no planes");
                    continue;
                }
                assert_eq!(got.shortest, s, "shortest mismatch");

                for sq in 0..81usize {
                    let bit = FLOOD_BIT_BY_SQ[sq];
                    let expect = if fwd[sq] == u8::MAX || inv[sq] == u8::MAX {
                        None
                    } else {
                        let slack = fwd[sq] as usize + inv[sq] as usize - s as usize;
                        (slack < SLACK_PLANES).then_some(slack)
                    };
                    for k in 0..SLACK_PLANES {
                        let in_plane = got.planes[k] & bit != 0;
                        assert_eq!(
                            in_plane,
                            expect == Some(k),
                            "sq={sq} k={k} fwd={} inv={} s={s} h={:#x} v={:#x}",
                            fwd[sq],
                            inv[sq],
                            board.horizontal_walls,
                            board.vertical_walls,
                        );
                    }
                }
                checked += 1;
            }
        }
        assert!(checked > 100, "only {checked} fields checked");
    }

    /// Plane 0 must be exactly the union of all shortest paths, so it is a
    /// superset of any single shortest path the engine can find.
    #[test]
    fn plane_zero_contains_every_shortest_path() {
        for board in corpus(60, 30) {
            let masks = DirMasks::from_board(&board);
            for player in [Player::One, Player::Two] {
                let (r, c) = board.pawn(player);
                let start = square_index(r, c);
                let pawn = FLOOD_BIT_BY_SQ[start as usize];
                let mut from = DistLayers::default();
                let mut to = DistLayers::default();
                fill_dist_layers_from_sq(start, masks, &mut from);
                fill_dist_layers_to_goal_row(player, masks, &mut to);
                let planes = build_slack_planes(&from, &to, pawn);
                if !planes.reachable() {
                    continue;
                }

                let mut path = [0u8; 81];
                let mut sc = BfsScratch::new();
                if let Some(len) = sc.shortest_path(&board, player, &mut path) {
                    for &sq in &path[..len] {
                        assert_ne!(
                            planes.planes[0] & FLOOD_BIT_BY_SQ[sq as usize],
                            0,
                            "shortest-path square {sq} missing from plane 0"
                        );
                    }
                }
            }
        }
    }

    /// The to-goal side is pawn-independent: moving a pawn must not change it,
    /// which is what makes incremental reuse across a search path sound.
    #[test]
    fn to_goal_side_is_independent_of_pawn() {
        for board in corpus(40, 24) {
            let masks = DirMasks::from_board(&board);
            let mut a = DistLayers::default();
            fill_dist_layers_to_goal_row(Player::One, masks, &mut a);

            let mut moved = board.clone();
            moved.pawns[Player::One as usize] = (4, 4);
            let masks2 = DirMasks::from_board(&moved);
            let mut b = DistLayers::default();
            fill_dist_layers_to_goal_row(Player::One, masks2, &mut b);

            assert_eq!(a.depth, b.depth, "to-goal depth changed on pawn move");
            for d in 0..a.depth {
                assert_eq!(a.masks[d], b.masks[d], "to-goal ring {d} changed");
            }
        }
    }

    /// Head-to-head against the CATv5 build this replaces.
    #[test]
    fn slack_planes_versus_catv5_build_cost() {
        use crate::cat::build::build_catv5_heatmaps;
        use std::hint::black_box;
        use std::time::Instant;

        let boards = corpus(40, 30);
        const REPS: u32 = 200;

        let mut t_old = 0u128;
        for _ in 0..REPS {
            for b in &boards {
                let t = Instant::now();
                black_box(build_catv5_heatmaps(b));
                t_old += t.elapsed().as_nanos();
            }
        }

        // Slack planes for both players, including both floods each.
        let mut t_new = 0u128;
        for _ in 0..REPS {
            for b in &boards {
                let t = Instant::now();
                let masks = DirMasks::from_board(b);
                for player in [Player::One, Player::Two] {
                    let (r, c) = b.pawn(player);
                    let start = square_index(r, c);
                    let mut from = DistLayers::default();
                    let mut to = DistLayers::default();
                    fill_dist_layers_from_sq(start, masks, &mut from);
                    fill_dist_layers_to_goal_row(player, masks, &mut to);
                    black_box(build_slack_planes(
                        &from,
                        &to,
                        FLOOD_BIT_BY_SQ[start as usize],
                    ));
                }
                t_new += t.elapsed().as_nanos();
            }
        }

        // Same, but with DirMasks supplied by the caller (the search already has
        // it) — the duplication the profile flagged.
        let mut t_new_nomask = 0u128;
        let premasks: Vec<DirMasks> = boards.iter().map(DirMasks::from_board).collect();
        for _ in 0..REPS {
            for (b, masks) in boards.iter().zip(&premasks) {
                let t = Instant::now();
                for player in [Player::One, Player::Two] {
                    let (r, c) = b.pawn(player);
                    let start = square_index(r, c);
                    let mut from = DistLayers::default();
                    let mut to = DistLayers::default();
                    fill_dist_layers_from_sq(start, *masks, &mut from);
                    fill_dist_layers_to_goal_row(player, *masks, &mut to);
                    black_box(build_slack_planes(
                        &from,
                        &to,
                        FLOOD_BIT_BY_SQ[start as usize],
                    ));
                }
                t_new_nomask += t.elapsed().as_nanos();
            }
        }

        let n = (REPS as usize * boards.len()) as f64;
        let (o, w, m) = (
            t_old as f64 / n,
            t_new as f64 / n,
            t_new_nomask as f64 / n,
        );
        eprintln!("\n=== CAT build: CATv5 vs slack planes ===");
        eprintln!("  CATv5 build_catv5_heatmaps   {o:8.0} ns   1.00x");
        eprintln!("  slack planes (own DirMasks)  {w:8.0} ns   {:.2}x faster", o / w);
        eprintln!("  slack planes (masks reused)  {m:8.0} ns   {:.2}x faster", o / m);
        let _ = (WallOrientation::Horizontal,);
    }
}
