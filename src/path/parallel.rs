//! Movegen V11 wall-legality core — fixed port of the `quoridor_parallel_engine` POC.
//!
//! One u128 register holds the whole 9×9 board (centered 11-wide flood layout,
//! shared with `path::flood`). Wall topology lives in four directional
//! "step out of this square is blocked" bitboards, so a speculative wall trial
//! is two OR/AND-NOT mask flips instead of a `DirMasks::from_board` rebuild.
//! Legality of a wall is then a linear-time SIMD-style flood: every frontier
//! cell expands in all four directions per iteration via four shifts.
//!
//! Fixes applied to the original POC:
//! 1. Layout: 9 rows × 16-bit stride needs 144 bits — does not fit u128
//!    (the "row 8 = bits 128..137" comment was out of range). The centered
//!    11-stride layout tops out at bit 108 and its buffer ring absorbs every
//!    off-board shift.
//! 2. Expansion: the POC's "directional ray sweeps" (`!f & f.wrapping_neg()`,
//!    `first_blocker - 1`, …) treat the whole register as a single ray — with
//!    more than one frontier bit the carry chains leak across rows and skip
//!    blockers. Replaced with the correct one-step parallel dilation: all
//!    frontier cells advance one square in all four directions per iteration.
//! 3. Wall gating: blocked-step masks must gate the *source* square before the
//!    shift (`(wave & !blocked) << k`), not be subtracted from destinations.
//! 4. Bit theft: when Player 2's wave first touches Player 1's cached flood it
//!    annexes the whole region (pawn connectivity is undirected), but the POC
//!    never re-tested the annexed cells against Player 2's goal — a flood that
//!    inherited goal-row cells could still report "trapped". The annexed pool
//!    is now goal-tested at theft time.

use crate::core::board::{Board, Player, WallOrientation};
use crate::util::grid::{flood_bit_index, FLOOD_PLAYABLE, FLOOD_STRIDE};

/// Per-direction blocked-step masks in flood-bit layout.
/// Bit set ⇒ a pawn on that square may NOT step in that direction.
/// `south` = toward row 8 (Player 1's goal), `north` = toward row 0.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct WallGrids {
    pub east: u128,
    pub west: u128,
    pub north: u128,
    pub south: u128,
}

#[inline]
const fn cell(row: u8, col: u8) -> u128 {
    1u128 << flood_bit_index(row, col)
}

const fn goal_row_bits(row: u8) -> u128 {
    let mut mask = 0u128;
    let mut col = 0u8;
    while col < 9 {
        mask |= cell(row, col);
        col += 1;
    }
    mask
}

/// Player 1 wins on row 8.
pub const P1_GOAL_BITS: u128 = goal_row_bits(8);
/// Player 2 wins on row 0.
pub const P2_GOAL_BITS: u128 = goal_row_bits(0);

#[inline]
pub const fn goal_bits(player: Player) -> u128 {
    match player {
        Player::One => P1_GOAL_BITS,
        Player::Two => P2_GOAL_BITS,
    }
}

/// Flood bit of a pawn square.
#[inline]
pub const fn pawn_bit(row: u8, col: u8) -> u128 {
    cell(row, col)
}

/// Horizontal wall at slot (r, c) closes the edges (r,c)↕(r+1,c) and (r,c+1)↕(r+1,c+1).
const fn h_wall_delta(slot: usize) -> WallGrids {
    let r = (slot / 8) as u8;
    let c = (slot % 8) as u8;
    WallGrids {
        east: 0,
        west: 0,
        north: cell(r + 1, c) | cell(r + 1, c + 1),
        south: cell(r, c) | cell(r, c + 1),
    }
}

/// Vertical wall at slot (r, c) closes the edges (r,c)↔(r,c+1) and (r+1,c)↔(r+1,c+1).
const fn v_wall_delta(slot: usize) -> WallGrids {
    let r = (slot / 8) as u8;
    let c = (slot % 8) as u8;
    WallGrids {
        east: cell(r, c) | cell(r + 1, c),
        west: cell(r, c + 1) | cell(r + 1, c + 1),
        north: 0,
        south: 0,
    }
}

const H_WALL_DELTAS: [WallGrids; 64] = {
    let mut t = [WallGrids::ZERO; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = h_wall_delta(i);
        i += 1;
    }
    t
};

const V_WALL_DELTAS: [WallGrids; 64] = {
    let mut t = [WallGrids::ZERO; 64];
    let mut i = 0;
    while i < 64 {
        t[i] = v_wall_delta(i);
        i += 1;
    }
    t
};

impl WallGrids {
    pub const ZERO: Self = Self {
        east: 0,
        west: 0,
        north: 0,
        south: 0,
    };

    /// Build from the board's packed u64 wall sets — O(#walls placed).
    pub fn from_board(board: &Board) -> Self {
        let mut grids = Self::ZERO;
        let mut h = board.horizontal_walls;
        while h != 0 {
            grids.place(&H_WALL_DELTAS[h.trailing_zeros() as usize]);
            h &= h - 1;
        }
        let mut v = board.vertical_walls;
        while v != 0 {
            grids.place(&V_WALL_DELTAS[v.trailing_zeros() as usize]);
            v &= v - 1;
        }
        grids
    }

    /// Speculatively apply a wall (Step 1 of the validation pipeline).
    #[inline]
    pub fn place(&mut self, delta: &WallGrids) {
        self.east |= delta.east;
        self.west |= delta.west;
        self.north |= delta.north;
        self.south |= delta.south;
    }

    /// Roll back a speculative wall. Non-colliding walls never share blocked
    /// edges, so clearing the delta's bits restores the previous state exactly.
    #[inline]
    pub fn remove(&mut self, delta: &WallGrids) {
        self.east &= !delta.east;
        self.west &= !delta.west;
        self.north &= !delta.north;
        self.south &= !delta.south;
    }
}

/// Blocked-step delta for one wall (internal slot coords, row/col in 0..8).
#[inline]
pub fn wall_delta(row: u8, col: u8, orientation: WallOrientation) -> &'static WallGrids {
    let slot = (row as usize) * 8 + col as usize;
    match orientation {
        WallOrientation::Horizontal => &H_WALL_DELTAS[slot],
        WallOrientation::Vertical => &V_WALL_DELTAS[slot],
    }
}

/// One parallel dilation step: every wave cell advances one square in all four
/// directions; blocked-step masks gate sources, the buffer ring + playable
/// mask kill off-board shifts. 12 bit-ops on two registers, branch-free.
#[inline]
pub fn expand_wave(wave: u128, grids: &WallGrids) -> u128 {
    let east = (wave & !grids.east) << 1;
    let west = (wave & !grids.west) >> 1;
    let south = (wave & !grids.south) << FLOOD_STRIDE;
    let north = (wave & !grids.north) >> FLOOD_STRIDE;
    (east | west | south | north) & FLOOD_PLAYABLE
}

/// Selfish flood with early goal exit. Returns (goal reached, visited bits) —
/// the visited set doubles as the history cache for the second player's run.
#[inline]
pub fn flood_to_goal_grids(start: u128, grids: &WallGrids, goal: u128) -> (bool, u128) {
    let mut visited = start & FLOOD_PLAYABLE;
    if visited & goal != 0 {
        return (true, visited);
    }
    let mut wave = visited;
    while wave != 0 {
        wave = expand_wave(wave, grids) & !visited;
        if wave & goal != 0 {
            return (true, visited | wave);
        }
        visited |= wave;
    }
    (false, visited)
}

/// Second-player flood with bit theft: on first contact with the cached first-
/// player region the whole region is annexed (and goal-tested — POC fix #4),
/// so shared corridors are never re-flooded.
#[inline]
pub fn flood_to_goal_with_cache(start: u128, cache: u128, grids: &WallGrids, goal: u128) -> bool {
    let mut visited = start & FLOOD_PLAYABLE;
    if visited & goal != 0 {
        return true;
    }
    let mut wave = visited;
    let mut pool = cache & !visited;
    while wave != 0 {
        if wave & pool != 0 {
            if pool & goal != 0 {
                return true;
            }
            visited |= pool;
            wave |= pool;
            pool = 0;
        }
        wave = expand_wave(wave, grids) & !visited;
        if wave & goal != 0 {
            return true;
        }
        visited |= wave;
    }
    false
}

/// Step 3 of the POC pipeline: Player 1 floods selfishly (filling the cache),
/// Player 2 floods with bit theft. Either flood stagnating ⇒ illegal wall.
#[inline]
pub fn both_players_reach_goals_grids(p1_start: u128, p2_start: u128, grids: &WallGrids) -> bool {
    let (ok1, p1_visited) = flood_to_goal_grids(p1_start, grids, P1_GOAL_BITS);
    if !ok1 {
        return false;
    }
    flood_to_goal_with_cache(p2_start, p1_visited, grids, P2_GOAL_BITS)
}

/// Convenience wrapper for one-off queries (oracle / replay validation).
pub fn both_players_reach_goals_parallel(board: &Board) -> bool {
    let grids = WallGrids::from_board(board);
    let (r1, c1) = board.pawn(Player::One);
    let (r2, c2) = board.pawn(Player::Two);
    both_players_reach_goals_grids(pawn_bit(r1, c1), pawn_bit(r2, c2), &grids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::{Board, Player, WallOrientation};
    use crate::util::grid::{can_step, goal_row, set_wall, square_index, unpack_square};

    /// Queue BFS over `can_step` — the obviously-correct reference.
    fn reach_goal_naive(board: &Board, start: (u8, u8), player: Player) -> bool {
        let mut seen = [false; 81];
        let mut queue = [0u8; 81];
        let (mut head, mut tail) = (0usize, 1usize);
        queue[0] = square_index(start.0, start.1);
        seen[queue[0] as usize] = true;
        while head < tail {
            let sq = queue[head];
            head += 1;
            let (r, c) = unpack_square(sq);
            if r == goal_row(player) {
                return true;
            }
            for (dr, dc) in [(1i8, 0i8), (-1, 0), (0, 1), (0, -1)] {
                if !can_step(board, r, c, dr, dc) {
                    continue;
                }
                let nsq = square_index((r as i8 + dr) as u8, (c as i8 + dc) as u8);
                if !seen[nsq as usize] {
                    seen[nsq as usize] = true;
                    queue[tail] = nsq;
                    tail += 1;
                }
            }
        }
        false
    }

    fn grids_match_board(board: &Board) {
        let grids = WallGrids::from_board(board);
        for r in 0..9u8 {
            for c in 0..9u8 {
                let bit = cell(r, c);
                assert_eq!(
                    can_step(board, r, c, 1, 0),
                    r < 8 && grids.south & bit == 0,
                    "south step mismatch at ({r},{c})"
                );
                assert_eq!(
                    can_step(board, r, c, -1, 0),
                    r > 0 && grids.north & bit == 0,
                    "north step mismatch at ({r},{c})"
                );
                assert_eq!(
                    can_step(board, r, c, 0, 1),
                    c < 8 && grids.east & bit == 0,
                    "east step mismatch at ({r},{c})"
                );
                assert_eq!(
                    can_step(board, r, c, 0, -1),
                    c > 0 && grids.west & bit == 0,
                    "west step mismatch at ({r},{c})"
                );
            }
        }
    }

    #[test]
    fn wall_grids_match_can_step_for_every_single_wall() {
        for orientation in [WallOrientation::Horizontal, WallOrientation::Vertical] {
            for row in 0..8u8 {
                for col in 0..8u8 {
                    let mut board = Board::new();
                    set_wall(&mut board, row, col, orientation, true);
                    grids_match_board(&board);
                }
            }
        }
    }

    #[test]
    fn empty_board_both_reach() {
        assert!(both_players_reach_goals_parallel(&Board::new()));
    }

    #[test]
    fn adjacent_pawns_near_goal_regression() {
        // V10's partial-component shortcut returned false here (false negative).
        let mut board = Board::new();
        board.pawns[Player::One as usize] = (7, 4);
        board.pawns[Player::Two as usize] = (6, 4);
        assert!(both_players_reach_goals_parallel(&board));
    }

    #[test]
    fn fully_caged_pawn_is_detected() {
        // Box P2's pawn start (8,4): walls below and on both sides.
        let mut board = Board::new();
        set_wall(&mut board, 7, 3, WallOrientation::Horizontal, true);
        set_wall(&mut board, 7, 3, WallOrientation::Vertical, true);
        set_wall(&mut board, 7, 4, WallOrientation::Vertical, true);
        assert!(!reach_goal_naive(&board, (8, 4), Player::Two));
        assert!(!both_players_reach_goals_parallel(&board));
    }

    #[test]
    fn theft_pool_goal_is_detected() {
        // P1 ahead of P2 so that P1's early-exit flood owns the row-0 cells
        // P2 needs; the annexed pool itself must satisfy P2's goal (fix #4).
        let mut board = Board::new();
        board.pawns[Player::One as usize] = (1, 4);
        board.pawns[Player::Two as usize] = (2, 4);
        assert!(both_players_reach_goals_parallel(&board));
    }

    #[test]
    fn random_walls_match_naive_reference() {
        // Deterministic LCG — no rand dependency.
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (state >> 33) as u32
        };

        for _ in 0..500 {
            let mut board = Board::new();
            let wall_count = next() % 12;
            for _ in 0..wall_count {
                let row = (next() % 8) as u8;
                let col = (next() % 8) as u8;
                let orientation = if next() & 1 == 0 {
                    WallOrientation::Horizontal
                } else {
                    WallOrientation::Vertical
                };
                // Raw overlap guard only — trapping configurations are wanted here.
                if crate::util::grid::has_wall(&board, row, col, WallOrientation::Horizontal)
                    || crate::util::grid::has_wall(&board, row, col, WallOrientation::Vertical)
                {
                    continue;
                }
                set_wall(&mut board, row, col, orientation, true);
            }
            let p1 = ((next() % 9) as u8, (next() % 9) as u8);
            let mut p2 = ((next() % 9) as u8, (next() % 9) as u8);
            if p2 == p1 {
                p2 = ((p2.0 + 1) % 9, p2.1);
            }
            board.pawns[Player::One as usize] = p1;
            board.pawns[Player::Two as usize] = p2;

            grids_match_board(&board);

            let grids = WallGrids::from_board(&board);
            let expected = reach_goal_naive(&board, p1, Player::One)
                && reach_goal_naive(&board, p2, Player::Two);
            let got =
                both_players_reach_goals_grids(pawn_bit(p1.0, p1.1), pawn_bit(p2.0, p2.1), &grids);
            assert_eq!(got, expected, "walls h={:#x} v={:#x} p1={:?} p2={:?}",
                board.horizontal_walls, board.vertical_walls, p1, p2);

            // Single-player floods must match the reference too.
            let (got1, _) = flood_to_goal_grids(pawn_bit(p1.0, p1.1), &grids, P1_GOAL_BITS);
            assert_eq!(got1, reach_goal_naive(&board, p1, Player::One));
            let (got2, _) = flood_to_goal_grids(pawn_bit(p2.0, p2.1), &grids, P2_GOAL_BITS);
            assert_eq!(got2, reach_goal_naive(&board, p2, Player::Two));
        }
    }

    #[test]
    fn place_remove_round_trips() {
        let board = Board::new();
        let base = WallGrids::from_board(&board);
        for orientation in [WallOrientation::Horizontal, WallOrientation::Vertical] {
            for row in 0..8u8 {
                for col in 0..8u8 {
                    let mut grids = base;
                    let delta = wall_delta(row, col, orientation);
                    grids.place(delta);
                    grids.remove(delta);
                    assert_eq!(grids, base);
                }
            }
        }
    }
}
