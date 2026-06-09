//! BFS distance fields used by CAT build and path queries.

use crate::core::board::Player;
use crate::path::flood::expand_frontier;
use crate::path::masks::DirMasks;
use crate::util::grid::{flood_bit_sq, flood_sq_from_bit, goal_row, square_index, FLOOD_PLAYABLE};

/// Fill `dist_from[sq]` with BFS distance from `start`. Unreachable → `u8::MAX`.
pub fn fill_dist_from_sq(start: u8, masks: DirMasks, dist_from: &mut [u8; 81]) {
    dist_from.fill(u8::MAX);
    dist_from[start as usize] = 0;
    let mut reached = flood_bit_sq(start);
    let mut frontier = reached;
    let mut layer = 0u8;
    while frontier != 0 {
        layer += 1;
        let new = expand_frontier(frontier, masks) & !reached & FLOOD_PLAYABLE;
        if new == 0 {
            break;
        }
        let mut bits = new;
        while bits != 0 {
            let fb = bits.trailing_zeros();
            bits &= bits - 1;
            let sq = flood_sq_from_bit(fb).expect("playable flood bit");
            dist_from[sq as usize] = layer;
        }
        reached |= new;
        frontier = new;
    }
}

/// Fill `dist_to[sq]` with BFS distance to any goal-row cell for `player`.
pub fn fill_dist_to_goal_row(player: Player, masks: DirMasks, dist_to: &mut [u8; 81]) {
    let grow = goal_row(player);
    dist_to.fill(u8::MAX);

    let mut reached = 0u128;
    for c in 0..9u8 {
        let sq = square_index(grow, c);
        dist_to[sq as usize] = 0;
        reached |= flood_bit_sq(sq);
    }

    let mut frontier = reached;
    let mut layer = 0u8;
    while frontier != 0 {
        layer += 1;
        let new = expand_frontier(frontier, masks) & !reached & FLOOD_PLAYABLE;
        if new == 0 {
            break;
        }
        let mut bits = new;
        while bits != 0 {
            let fb = bits.trailing_zeros();
            bits &= bits - 1;
            let sq = flood_sq_from_bit(fb).expect("playable flood bit");
            dist_to[sq as usize] = layer;
        }
        reached |= new;
        frontier = new;
    }
}
