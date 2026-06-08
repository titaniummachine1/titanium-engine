//! Reachability — **bitwise flood fill** on the 9×9 pawn grid (uniform edge cost).
//!
//! Direction masks are built once per board snapshot; expansion is word-parallel
//! shifts on a **centered 11-wide u128 layout** (`grid::FLOOD_STRIDE`) so
//! east/west shifts land in side buffers instead of wrapping rows. CAT is
//! accumulated during the same level-BFS passes used for shortest-path distance.
//!
//! See `docs/video/PERFT-OPTIMIZATIONS.md` Layer 4 for timings and oracles.

use crate::board::{Board, Player};
use crate::grid::{
    can_step, flood_bit_sq, flood_sq_from_bit, goal_row, pack_flood_mask, square_index,
    unpack_square, FLOOD_PLAYABLE, FLOOD_STRIDE,
};

/// Per-square attention scores for move ordering / LMR (centi-units, not eval).
pub type ConsensusAttention = [u16; 81];

const CAT_BASE_CM: u16 = 100;
const CAT_DIST_PENALTY_CM: u16 = 3;
const CAT_MAX_PENALTY_CM: u16 = 30;

/// CAT weight for a square at `dist` steps from the pawn (centi-units).
///
/// `weight = 100 - clamp(dist * 3, 0, 30)`
/// - dist 0 → 100 cm, dist 10+ → 70 cm (floor).
#[inline]
pub fn attention_weight_cm(dist: u8) -> u16 {
    let penalty = u16::from(dist)
        .saturating_mul(CAT_DIST_PENALTY_CM)
        .min(CAT_MAX_PENALTY_CM);
    CAT_BASE_CM.saturating_sub(penalty)
}

/// Bit `sq` set iff a pawn on `sq` may step in that direction.
#[derive(Clone, Copy, Default)]
pub struct DirMasks {
    pub north: u128,
    pub south: u128,
    pub east: u128,
    pub west: u128,
}

impl DirMasks {
    pub fn from_board(board: &Board) -> Self {
        let mut m = Self::default();
        for r in 0..=8u8 {
            for c in 0..=8u8 {
                let sq = square_index(r, c);
                let bit = flood_bit_sq(sq);
                if can_step(board, r, c, -1, 0) {
                    m.north |= bit;
                }
                if can_step(board, r, c, 1, 0) {
                    m.south |= bit;
                }
                if can_step(board, r, c, 0, 1) {
                    m.east |= bit;
                }
                if can_step(board, r, c, 0, -1) {
                    m.west |= bit;
                }
            }
        }
        m
    }
}

#[inline]
fn goal_square_mask(player: Player) -> u128 {
    let grow = goal_row(player);
    let mut mask = 0u128;
    for c in 0..9u8 {
        mask |= flood_bit_sq(square_index(grow, c));
    }
    mask
}

/// Expand flood frontier in centered 11-wide layout (side buffers absorb E/W shifts).
#[inline]
fn expand_frontier(frontier: u128, masks: DirMasks) -> u128 {
    let north = (frontier & masks.north) >> FLOOD_STRIDE;
    let south = (frontier & masks.south) << FLOOD_STRIDE;
    let east = (frontier & masks.east) << 1;
    let west = (frontier & masks.west) >> 1;
    north | south | east | west
}

#[inline]
fn flood_fill_flood_bits(start_sq: u8, masks: DirMasks) -> u128 {
    let mut reached = flood_bit_sq(start_sq);
    let mut frontier = reached;
    while frontier != 0 {
        frontier = expand_frontier(frontier, masks) & !reached & FLOOD_PLAYABLE;
        reached |= frontier;
    }
    reached
}

#[inline]
#[cfg(test)]
fn flood_fill(start_sq: u8, masks: DirMasks) -> u128 {
    pack_flood_mask(flood_fill_flood_bits(start_sq, masks))
}

#[inline]
fn flood_to_goal(start_sq: u8, masks: DirMasks, goal_mask: u128) -> (bool, u128) {
    let mut reached = flood_bit_sq(start_sq);
    if reached & goal_mask != 0 {
        return (true, reached);
    }
    let mut frontier = reached;
    while frontier != 0 {
        frontier = expand_frontier(frontier, masks) & !reached & FLOOD_PLAYABLE;
        reached |= frontier;
        if frontier & goal_mask != 0 {
            return (true, reached);
        }
    }
    (false, reached)
}

#[inline]
fn predecessor_flood(bit: u32, reached: u128, masks: DirMasks) -> u32 {
    if bit >= FLOOD_STRIDE {
        let p = bit - FLOOD_STRIDE;
        if reached & (1u128 << p) != 0 && masks.south & (1u128 << p) != 0 {
            return p;
        }
    }
    if bit + FLOOD_STRIDE < 128 {
        let p = bit + FLOOD_STRIDE;
        if reached & (1u128 << p) != 0 && masks.north & (1u128 << p) != 0 {
            return p;
        }
    }
    if bit > 0 {
        let p = bit - 1;
        if reached & (1u128 << p) != 0 && masks.east & (1u128 << p) != 0 {
            return p;
        }
    }
    if bit + 1 < 128 {
        let p = bit + 1;
        if reached & (1u128 << p) != 0 && masks.west & (1u128 << p) != 0 {
            return p;
        }
    }
    bit
}

/// Accumulate one player's contribution into `out` (CAT forward pass + back-prop).
///
/// **Forward pass** — level-BFS from the player's pawn:
///   Each reached square accumulates `attention_weight_cm(dist)` = `100 - min(dist*3, 30)`.
///   Adjacent squares score 100 cm; squares ≥10 steps away floor at 70 cm.
///
/// **Back-propagation** — shortest-path reconstruction:
///   After the goal row is first touched, walk `parent[]` back to the pawn and add
///   the same `attention_weight_cm(dist)` a second time to every square on that path.
///   On-path squares near the pawn thus score up to 200 cm; far ones 140 cm.
///
/// Call twice (P1, P2) and sum into the same `out` array → unified heat map.
fn add_player_attention(
    board: &Board,
    player: Player,
    masks: DirMasks,
    out: &mut ConsensusAttention,
    dist: &mut [u8; 81],
    parent: &mut [u8; 81],
) {
    let (sr, sc) = board.pawn(player);
    let start = square_index(sr, sc);
    let goal_mask = goal_square_mask(player);

    dist.fill(u8::MAX);
    parent.fill(u8::MAX);
    dist[start as usize] = 0;

    let mut reached = flood_bit_sq(start);
    out[start as usize] = out[start as usize].saturating_add(attention_weight_cm(0));

    let mut frontier = reached;
    let mut layer = 0u8;
    let mut goal_sq = None;

    while frontier != 0 && goal_sq.is_none() {
        layer += 1;
        let new = expand_frontier(frontier, masks) & !reached & FLOOD_PLAYABLE;
        if new == 0 {
            break;
        }

        let w = attention_weight_cm(layer);
        let mut bits = new;
        while bits != 0 {
            let fb = bits.trailing_zeros();
            bits &= bits - 1;
            let sq = flood_sq_from_bit(fb).expect("playable flood bit");
            dist[sq as usize] = layer;
            parent[sq as usize] = flood_sq_from_bit(predecessor_flood(fb, reached, masks))
                .expect("predecessor on playable square");
            out[sq as usize] = out[sq as usize].saturating_add(w);
            if goal_sq.is_none() && goal_mask & (1u128 << fb) != 0 {
                goal_sq = Some(sq);
            }
        }

        reached |= new;
        frontier = new;
    }

    if let Some(mut sq) = goal_sq {
        loop {
            let w = attention_weight_cm(dist[sq as usize]);
            out[sq as usize] = out[sq as usize].saturating_add(w);
            if sq == start {
                break;
            }
            let p = parent[sq as usize];
            if p == u8::MAX {
                break;
            }
            sq = p;
        }
    }
}

/// Reused flood-fill scratch — pass through perft/move-gen hot loops.
#[derive(Clone)]
pub struct BfsScratch {
    visited: u128,
    queue: [u8; 81],
    depth: [u8; 81],
    parent: [u8; 81],
}

impl Default for BfsScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl BfsScratch {
    pub fn new() -> Self {
        Self {
            visited: 0,
            queue: [0; 81],
            depth: [0; 81],
            parent: [0; 81],
        }
    }

    /// Build the Consensus Attention Table (CAT) for this position.
    ///
    /// Runs `add_player_attention` for both players and sums the results into a
    /// single `[u16; 81]` heat map (centi-units).  Used in search for move ordering
    /// and LMR only — never called from perft or move generation.
    ///
    /// Score ranges per square:
    /// - On-path, close (dist ≤ 1): up to **200 cm** per player → 400 cm combined.
    /// - On-path, far (dist ≥ 10): 140 cm per player.
    /// - Off-path, any dist: 70–100 cm per player.
    ///
    /// See `docs/video/CAT-SPEC.md` for full specification.
    pub fn build_consensus_attention(&mut self, board: &Board) -> ConsensusAttention {
        let masks = DirMasks::from_board(board);
        let mut cat = [0u16; 81];
        add_player_attention(
            board,
            Player::One,
            masks,
            &mut cat,
            &mut self.depth,
            &mut self.parent,
        );
        add_player_attention(
            board,
            Player::Two,
            masks,
            &mut cat,
            &mut self.depth,
            &mut self.parent,
        );
        cat
    }

    /// Reachability only — bitwise flood to goal row.
    #[inline]
    pub fn can_reach_goal(&mut self, board: &Board, player: Player) -> bool {
        let masks = DirMasks::from_board(board);
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        flood_to_goal(start, masks, goal_square_mask(player)).0
    }

    /// Both players must reach their goal. Reuses P1's component mask when P2 is inside it.
    #[inline]
    pub fn both_players_reach_goals(&mut self, board: &Board) -> bool {
        let masks = DirMasks::from_board(board);
        let (r1, c1) = board.pawn(Player::One);
        let start1 = square_index(r1, c1);
        let goal1 = goal_square_mask(Player::One);
        let (ok1, comp1) = flood_to_goal(start1, masks, goal1);
        if !ok1 {
            return false;
        }

        let (r2, c2) = board.pawn(Player::Two);
        let start2 = square_index(r2, c2);
        let goal2 = goal_square_mask(Player::Two);
        let start2_bit = flood_bit_sq(start2);

        if comp1 & start2_bit != 0 {
            return comp1 & goal2 != 0;
        }

        flood_to_goal(start2, masks, goal2).0
    }

    /// Bitwise flood from `player`'s pawn — sets bits in `mask`.
    pub fn fill_reachable(&mut self, board: &Board, player: Player, mask: &mut u128) {
        let masks = DirMasks::from_board(board);
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        *mask |= pack_flood_mask(flood_fill_flood_bits(start, masks));
    }

    /// Union of squares reachable by either pawn.
    pub fn both_reachable_mask(&mut self, board: &Board) -> u128 {
        let mut mask = 0u128;
        self.fill_reachable(board, Player::One, &mut mask);
        self.fill_reachable(board, Player::Two, &mut mask);
        mask
    }

    /// Shortest pawn-step distance to any goal square on the goal row.
    pub fn shortest_distance(&mut self, board: &Board, player: Player) -> Option<u8> {
        let masks = DirMasks::from_board(board);
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        let goal_mask = goal_square_mask(player);

        let mut reached = flood_bit_sq(start);
        if reached & goal_mask != 0 {
            return Some(0);
        }

        let mut frontier = reached;
        let mut d = 0u8;
        while frontier != 0 {
            d += 1;
            frontier = expand_frontier(frontier, masks) & !reached & FLOOD_PLAYABLE;
            if frontier & goal_mask != 0 {
                return Some(d);
            }
            reached |= frontier;
        }
        None
    }

    /// Backward BFS from all goal-row cells — next hop toward goal per square.
    pub fn fill_next_toward_goal(
        &mut self,
        board: &Board,
        player: Player,
        next_out: &mut [u8; 81],
    ) {
        next_out.fill(u8::MAX);
        let grow = goal_row(player);

        self.visited = 0;
        let mut head = 0usize;
        let mut tail = 0usize;

        for col in 0..9u8 {
            let sq = square_index(grow, col);
            let mask = 1u128 << sq;
            if self.visited & mask == 0 {
                self.visited |= mask;
                self.queue[tail] = sq;
                tail += 1;
            }
        }

        const NEIGHBORS: [(i8, i8); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

        while head < tail {
            let sq = self.queue[head];
            head += 1;
            let (r, c) = unpack_square(sq);

            for (dr, dc) in NEIGHBORS {
                let nr = r as i16 + dr as i16;
                let nc = c as i16 + dc as i16;
                if !(0..=8).contains(&nr) || !(0..=8).contains(&nc) {
                    continue;
                }
                let nr = nr as u8;
                let nc = nc as u8;
                if !can_step(board, nr, nc, -dr, -dc) {
                    continue;
                }
                let nsq = square_index(nr, nc);
                let mask = 1u128 << nsq;
                if self.visited & mask != 0 {
                    continue;
                }
                self.visited |= mask;
                next_out[nsq as usize] = sq;
                self.queue[tail] = nsq;
                tail += 1;
            }
        }
    }
}

#[inline]
pub fn can_reach_goal(board: &Board, player: Player) -> bool {
    BfsScratch::new().can_reach_goal(board, player)
}

pub fn shortest_distance(board: &Board, player: Player) -> Option<u8> {
    BfsScratch::new().shortest_distance(board, player)
}

#[inline]
pub fn both_players_reach_goals(board: &Board) -> bool {
    BfsScratch::new().both_players_reach_goals(board)
}

#[cfg(test)]
mod naive_reference {
    //! Queue BFS oracle — validates bitwise flood fill independent of DirMasks.

    use super::*;
    use crate::grid::is_goal;

    const NEIGHBORS: [(i8, i8); 4] = [(1, 0), (0, 1), (-1, 0), (0, -1)];

    pub fn flood_fill_naive(board: &Board, start: u8) -> u128 {
        let mut visited = 1u128 << start;
        let mut queue = [0u8; 81];
        let mut head = 0usize;
        let mut tail = 1usize;
        queue[0] = start;

        while head < tail {
            let sq = queue[head];
            head += 1;
            let (r, c) = unpack_square(sq);
            for (dr, dc) in NEIGHBORS {
                if !can_step(board, r, c, dr, dc) {
                    continue;
                }
                let nr = (r as i8 + dr) as u8;
                let nc = (c as i8 + dc) as u8;
                let nsq = square_index(nr, nc);
                let bit = 1u128 << nsq;
                if visited & bit != 0 {
                    continue;
                }
                visited |= bit;
                queue[tail] = nsq;
                tail += 1;
            }
        }
        visited
    }

    pub fn can_reach_goal_naive(board: &Board, player: Player) -> bool {
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        let mut visited = 1u128 << start;
        let mut queue = [0u8; 81];
        let mut head = 0usize;
        let mut tail = 1usize;
        queue[0] = start;

        while head < tail {
            let sq = queue[head];
            head += 1;
            let (r, c) = unpack_square(sq);
            if is_goal(player, r) {
                return true;
            }
            for (dr, dc) in NEIGHBORS {
                if !can_step(board, r, c, dr, dc) {
                    continue;
                }
                let nr = (r as i8 + dr) as u8;
                let nc = (c as i8 + dc) as u8;
                let nsq = square_index(nr, nc);
                let bit = 1u128 << nsq;
                if visited & bit != 0 {
                    continue;
                }
                visited |= bit;
                queue[tail] = nsq;
                tail += 1;
            }
        }
        false
    }

    pub fn shortest_distance_naive(board: &Board, player: Player) -> Option<u8> {
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        let mut visited = 1u128 << start;
        let mut queue = [0u8; 81];
        let mut depth = [0u8; 81];
        let mut head = 0usize;
        let mut tail = 1usize;
        queue[0] = start;
        depth[0] = 0;

        while head < tail {
            let sq = queue[head];
            let d = depth[head];
            head += 1;
            let (r, c) = unpack_square(sq);
            if is_goal(player, r) {
                return Some(d);
            }
            for (dr, dc) in NEIGHBORS {
                if !can_step(board, r, c, dr, dc) {
                    continue;
                }
                let nr = (r as i8 + dr) as u8;
                let nc = (c as i8 + dc) as u8;
                let nsq = square_index(nr, nc);
                let bit = 1u128 << nsq;
                if visited & bit != 0 {
                    continue;
                }
                visited |= bit;
                queue[tail] = nsq;
                depth[tail] = d + 1;
                tail += 1;
            }
        }
        None
    }

    pub fn both_players_reach_goals_naive(board: &Board) -> bool {
        can_reach_goal_naive(board, Player::One) && can_reach_goal_naive(board, Player::Two)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::naive_reference::{
        both_players_reach_goals_naive, can_reach_goal_naive, flood_fill_naive,
        shortest_distance_naive,
    };
    use crate::board::WallOrientation;
    use crate::grid::set_wall;

    fn assert_bitwise_matches_naive(board: &Board) {
        let masks = DirMasks::from_board(board);
        let mut scratch = BfsScratch::new();

        for sq in 0u8..81 {
            let bitwise = flood_fill(sq, masks);
            let naive = flood_fill_naive(board, sq);
            assert_eq!(
                bitwise, naive,
                "reachable mismatch from sq {sq} on board {:?}",
                board
            );
        }

        for player in [Player::One, Player::Two] {
            assert_eq!(
                scratch.can_reach_goal(board, player),
                can_reach_goal_naive(board, player),
                "can_reach_goal mismatch for {player:?}"
            );
            assert_eq!(
                scratch.shortest_distance(board, player),
                shortest_distance_naive(board, player),
                "shortest_distance mismatch for {player:?}"
            );
        }

        assert_eq!(
            scratch.both_players_reach_goals(board),
            both_players_reach_goals_naive(board),
            "both_players_reach_goals mismatch"
        );

        let mut mask_bitwise = 0u128;
        scratch.fill_reachable(board, Player::One, &mut mask_bitwise);
        let mut mask_bitwise2 = 0u128;
        scratch.fill_reachable(board, Player::Two, &mut mask_bitwise2);
        let union_bitwise = mask_bitwise | mask_bitwise2;
        assert_eq!(union_bitwise, scratch.both_reachable_mask(board));
    }

    fn board_with_walls(walls: &[(u8, u8, WallOrientation)]) -> Board {
        let mut board = Board::new();
        for &(row, col, orientation) in walls {
            set_wall(&mut board, row, col, orientation, true);
        }
        board
    }

    #[test]
    fn dir_masks_agree_with_can_step_on_startpos() {
        let board = Board::new();
        let masks = DirMasks::from_board(&board);
        for r in 0..=8u8 {
            for c in 0..=8u8 {
                let sq = square_index(r, c);
                let bit = flood_bit_sq(sq);
                assert_eq!(
                    masks.north & bit != 0,
                    can_step(&board, r, c, -1, 0),
                    "north at ({r},{c})"
                );
                assert_eq!(
                    masks.south & bit != 0,
                    can_step(&board, r, c, 1, 0),
                    "south at ({r},{c})"
                );
                assert_eq!(
                    masks.east & bit != 0,
                    can_step(&board, r, c, 0, 1),
                    "east at ({r},{c})"
                );
                assert_eq!(
                    masks.west & bit != 0,
                    can_step(&board, r, c, 0, -1),
                    "west at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn bitwise_flood_matches_naive_queue_on_startpos() {
        assert_bitwise_matches_naive(&Board::new());
    }

    #[test]
    fn bitwise_flood_matches_naive_with_barrier() {
        let board = board_with_walls(&[
            (6, 0, WallOrientation::Horizontal),
            (6, 1, WallOrientation::Horizontal),
            (6, 2, WallOrientation::Horizontal),
            (6, 3, WallOrientation::Horizontal),
            (6, 4, WallOrientation::Horizontal),
            (6, 5, WallOrientation::Horizontal),
            (6, 6, WallOrientation::Horizontal),
            (6, 7, WallOrientation::Horizontal),
        ]);
        assert_bitwise_matches_naive(&board);
    }

    #[test]
    fn bitwise_flood_matches_naive_with_mixed_walls() {
        let board = board_with_walls(&[
            (3, 3, WallOrientation::Vertical),
            (4, 4, WallOrientation::Horizontal),
            (2, 6, WallOrientation::Vertical),
            (5, 1, WallOrientation::Horizontal),
            (7, 3, WallOrientation::Vertical),
        ]);
        assert_bitwise_matches_naive(&board);
    }

    #[test]
    fn bitwise_flood_matches_naive_perft_depth2_prefix() {
        // Replay first two plies (e2 e8) — exercises wall gen path checks on a real subtree.
        let mut board = Board::new();
        let _ = board.make_move(crate::board::Move::Pawn { row: 1, col: 4 });
        let _ = board.make_move(crate::board::Move::Pawn { row: 7, col: 4 });
        assert_bitwise_matches_naive(&board);
    }

    #[test]
    fn centered_layout_absorbs_east_shift_in_side_buffer() {
        use crate::grid::{flood_bit_index, FLOOD_COL_PAD, FLOOD_ROW_PAD, FLOOD_STRIDE};

        let board = Board::new();
        let masks = DirMasks::from_board(&board);

        // Playable grid fits inside u128 with side buffers (stride 11, max bit 108).
        assert!(flood_bit_index(8, 8) < 128);
        assert_eq!(FLOOD_PLAYABLE.count_ones(), 81);

        // Force east open at (0,8): shift lands in side buffer, never (1,0) on next row.
        let mut masks = masks;
        masks.east |= flood_bit_sq(square_index(0, 8));
        let frontier = flood_bit_sq(square_index(0, 8));
        let raw = expand_frontier(frontier, masks);
        assert_eq!(
            raw & flood_bit_sq(square_index(1, 0)),
            0,
            "must not reach (1,0) across rows"
        );
        let buffer_col = FLOOD_ROW_PAD * FLOOD_STRIDE + FLOOD_COL_PAD + 9;
        assert_ne!(
            raw & (1u128 << buffer_col),
            0,
            "east shift should land in side buffer"
        );

        assert_bitwise_matches_naive(&board);
    }

    #[test]
    fn start_position_reachable() {
        let board = Board::new();
        assert!(can_reach_goal(&board, Player::One));
        assert!(can_reach_goal(&board, Player::Two));
        assert_eq!(shortest_distance(&board, Player::One), Some(8));
        assert_eq!(shortest_distance(&board, Player::Two), Some(8));
    }

    #[test]
    fn full_barrier_blocks_p1() {
        let mut board = Board::new();
        for c in 0..8u8 {
            set_wall(&mut board, 6, c, WallOrientation::Horizontal, true);
        }
        assert!(!can_reach_goal(&board, Player::One));
    }

    #[test]
    fn scratch_matches_stack_bfs() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        assert_eq!(scratch.shortest_distance(&board, Player::One), Some(8));
        assert!(scratch.both_players_reach_goals(&board));
    }

    #[test]
    fn both_reachable_mask_includes_both_pawns() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let mask = scratch.both_reachable_mask(&board);
        assert_ne!(mask & (1u128 << square_index(0, 4)), 0);
        assert_ne!(mask & (1u128 << square_index(8, 4)), 0);
    }

    #[test]
    fn bitwise_flood_reaches_goal_row() {
        let board = Board::new();
        let masks = DirMasks::from_board(&board);
        let start = square_index(0, 4);
        let reached = flood_fill(start, masks);
        let mut goal_row_mask = 0u128;
        for c in 0..9u8 {
            goal_row_mask |= 1u128 << square_index(8, c);
        }
        assert_ne!(reached & goal_row_mask, 0);
    }

    #[test]
    fn cat_startpos_has_hot_center() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let cat = scratch.build_consensus_attention(&board);
        assert!(cat[square_index(0, 4) as usize] > 0);
        assert!(cat[square_index(8, 4) as usize] > 0);
        assert!(cat[square_index(4, 4) as usize] > 0);
    }

    #[test]
    fn attention_weight_clamps_at_distance_10() {
        assert_eq!(attention_weight_cm(0), 100);
        assert_eq!(attention_weight_cm(10), 70);
        assert_eq!(attention_weight_cm(20), 70);
    }

    #[test]
    fn ishtar_component_reuse_same_board() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        assert!(scratch.both_players_reach_goals(&board));
    }
}
