//! `BfsScratch` — reusable flood-fill scratch for movegen, search, and CAT.

use crate::cat::attention::CorridorAttention;
use crate::cat::build::{build_corridor_attention, corridor_bottleneck_count};
use crate::core::board::{Board, Player};
use crate::path::flood::{expand_frontier, flood_fill_flood_bits, flood_to_goal, goal_square_mask};
use crate::path::masks::DirMasks;
use crate::util::grid::{
    can_step, flood_bit_sq, goal_row, pack_flood_mask, square_index, unpack_square, FLOOD_PLAYABLE,
};

/// Reused flood-fill scratch — pass through perft/move-gen hot loops.
#[derive(Clone)]
pub struct BfsScratch {
    visited: u128,
    queue: [u8; 81],
    dist_from_pawn: [u8; 81],
    dist_to_goal: [u8; 81],
    /// Cached `DirMasks` for the current board hash — one build per movegen node.
    masks_hash: u64,
    masks: DirMasks,
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
            dist_from_pawn: [0; 81],
            dist_to_goal: [0; 81],
            masks_hash: 0,
            masks: DirMasks::default(),
        }
    }

    /// Direction masks for the current position — rebuilt only when `board.hash` changes.
    #[inline]
    pub fn dir_masks(&mut self, board: &Board) -> DirMasks {
        if self.masks_hash != board.hash {
            self.masks_hash = board.hash;
            self.masks = DirMasks::from_board(board);
        }
        self.masks
    }

    /// Call after in-place wall trials — `board.hash` may match a stale cache entry.
    #[inline]
    pub fn invalidate_dir_masks(&mut self) {
        self.masks_hash = !0;
    }

    pub(crate) fn dist_scratch_mut(&mut self) -> (&mut [u8; 81], &mut [u8; 81]) {
        (&mut self.dist_from_pawn, &mut self.dist_to_goal)
    }

    pub fn build_corridor_attention(&mut self, board: &Board) -> CorridorAttention {
        build_corridor_attention(self, board)
    }

    pub fn corridor_bottleneck_count(&mut self, board: &Board, player: Player) -> u8 {
        corridor_bottleneck_count(self, board, player)
    }

    #[inline]
    pub fn can_reach_goal(&mut self, board: &Board, player: Player) -> bool {
        let masks = self.dir_masks(board);
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        flood_to_goal(start, masks, goal_square_mask(player)).0
    }

    #[inline]
    pub fn both_players_reach_goals(&mut self, board: &Board) -> bool {
        both_players_reach_goals_with_masks(board, self.dir_masks(board))
    }

    pub fn fill_reachable(&mut self, board: &Board, player: Player, mask: &mut u128) {
        let masks = self.dir_masks(board);
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        *mask |= pack_flood_mask(flood_fill_flood_bits(start, masks));
    }

    pub fn both_reachable_mask(&mut self, board: &Board) -> u128 {
        let mut mask = 0u128;
        self.fill_reachable(board, Player::One, &mut mask);
        self.fill_reachable(board, Player::Two, &mut mask);
        mask
    }

    pub fn shortest_distance(&mut self, board: &Board, player: Player) -> Option<u8> {
        let masks = self.dir_masks(board);
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

/// Both players can reach their goal row — uses caller-supplied masks (wall trials).
#[inline]
pub fn both_players_reach_goals_with_masks(board: &Board, masks: DirMasks) -> bool {
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
