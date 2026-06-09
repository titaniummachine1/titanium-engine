//! Bitwise flood-fill primitives (centered 11-wide u128 layout).

use crate::core::board::Player;
use crate::path::masks::DirMasks;
use crate::util::grid::{flood_bit_sq, goal_row, pack_flood_mask, square_index, FLOOD_PLAYABLE, FLOOD_STRIDE};

#[inline]
pub fn goal_square_mask(player: Player) -> u128 {
    let grow = goal_row(player);
    let mut mask = 0u128;
    for c in 0..9u8 {
        mask |= flood_bit_sq(square_index(grow, c));
    }
    mask
}

/// Expand flood frontier in centered 11-wide layout (side buffers absorb E/W shifts).
#[inline]
pub fn expand_frontier(frontier: u128, masks: DirMasks) -> u128 {
    let north = (frontier & masks.north) >> FLOOD_STRIDE;
    let south = (frontier & masks.south) << FLOOD_STRIDE;
    let east = (frontier & masks.east) << 1;
    let west = (frontier & masks.west) >> 1;
    north | south | east | west
}

#[inline]
pub fn flood_fill_flood_bits(start_sq: u8, masks: DirMasks) -> u128 {
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
pub fn flood_fill(start_sq: u8, masks: DirMasks) -> u128 {
    pack_flood_mask(flood_fill_flood_bits(start_sq, masks))
}

#[inline]
pub fn flood_to_goal(start_sq: u8, masks: DirMasks, goal_mask: u128) -> (bool, u128) {
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
