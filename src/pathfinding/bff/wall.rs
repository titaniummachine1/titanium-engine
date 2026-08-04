//! Movegen V11 wall-legality core — fixed port of the `quoridor_parallel_engine` POC.
//!
//! `bff_*` means binary flood fill: bitboard BFS used for fast path-to-goal
//! checks after tentative wall placements. It is
//! standard flood fill on `u128` masks, not a separate search or NN subsystem.
//!
//! One u128 register holds the whole 9×9 board (centered 11-wide flood layout,
//! shared with `pathfinding::bff`). Wall topology lives in four directional
//! "step out of this square is blocked" bitboards, so a speculative wall trial
//! is two OR/AND-NOT mask flips instead of a `DirMasks::from_board` rebuild.
//! Legality of a wall is then a linear-time bitboard flood: every frontier
//! cell expands in all four directions per iteration via four shifts (SIMD-style
//! bit ops are an implementation accelerator, not a different legality rule).
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
use crate::pathfinding::bff::expand_frontier;
use crate::pathfinding::masks::DirMasks;
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

    /// Whether this wall delta blocks an edge incident to `squares`.
    #[inline]
    pub fn touches(&self, squares: u128) -> bool {
        (self.east | self.west | self.north | self.south) & squares != 0
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

/// Binary / bitboard flood fill to goal. Returns (goal reached, visited bits) —
/// the visited set doubles as the history cache for the second player's run.
#[inline]
pub fn bff_to_goal(start: u128, grids: &WallGrids, goal: u128) -> (bool, u128) {
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

/// Second-player bitboard flood with **cached reachable-mask splice** (visited-bit reuse;
/// informal “bit theft”): on first contact with the first player's visited region the whole region is annexed (and
/// goal-tested — POC fix #4), so shared corridors are never re-flooded.
#[inline]
pub fn bff_to_goal_cached(start: u128, cache: u128, grids: &WallGrids, goal: u128) -> bool {
    bff_to_goal_cached_with_visited(start, cache, grids, goal).0
}

/// Cached second-player flood, including every square reached before success.
/// When the wave meets `cache`, bit stealing annexes that entire reached set.
#[inline]
pub fn bff_to_goal_cached_with_visited(
    start: u128,
    cache: u128,
    grids: &WallGrids,
    goal: u128,
) -> (bool, u128) {
    let mut visited = start & FLOOD_PLAYABLE;
    if visited & goal != 0 {
        return (true, visited);
    }
    let mut wave = visited;
    let mut pool = cache & !visited;
    while wave != 0 {
        if wave & pool != 0 {
            visited |= pool;
            wave |= pool;
            pool = 0;
            if visited & goal != 0 {
                return (true, visited);
            }
        }
        wave = expand_wave(wave, grids) & !visited;
        visited |= wave;
        if wave & goal != 0 {
            return (true, visited);
        }
    }
    (false, visited)
}

/// Wall legality via binary flood fill: both players must reach their goal row
/// after a tentative placement. Player 1 floods selfishly (filling the cache);
/// player 2 floods with visited-bit reuse. Either flood stagnating ⇒ illegal wall.
#[inline]
pub fn bff_wall_legal(p1_start: u128, p2_start: u128, grids: &WallGrids) -> bool {
    bff_wall_legal_with_proof(p1_start, p2_start, grids).0
}

/// Wall legality plus the union of both successful flood proofs.
#[inline]
pub fn bff_wall_legal_with_proof(
    p1_start: u128,
    p2_start: u128,
    grids: &WallGrids,
) -> (bool, u128) {
    let (ok1, p1_visited) = bff_to_goal(p1_start, grids, P1_GOAL_BITS);
    if !ok1 {
        return (false, 0);
    }
    let (ok2, p2_visited) =
        bff_to_goal_cached_with_visited(p2_start, p1_visited, grids, P2_GOAL_BITS);
    if !ok2 {
        return (false, 0);
    }
    (true, p1_visited | p2_visited)
}

/// One concrete shortest path to a goal row, stored in the SAME per-direction
/// shape as [`WallGrids`]: `east` holds every cell the path leaves heading east,
/// and so on.  A wall's blocked-step delta is in that shape too, so "does this
/// wall cut this path" is a 4-way AND — exact, no adjacency slop.
///
/// This is the witness the wall loop reasons about.  It is deliberately NOT the
/// reachable set: a reachable set covers most of the board, so almost every wall
/// touches it and it proves nothing.  A path is ~8-16 steps, so almost no wall
/// touches it and it proves a great deal.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct PathWitness {
    pub east: u128,
    pub west: u128,
    pub north: u128,
    pub south: u128,
    /// Square this path starts from. A path only proves anything for a player
    /// standing on it, so an inherited path is checked against the pawn before
    /// it is trusted.
    pub start: u128,
    /// Number of steps in this path (BFS depth when minted).  This is the
    /// shortest distance from `start` to the goal — eval can use it directly
    /// instead of doing its own flood.
    pub length: u8,
}

impl PathWitness {
    /// True when this path still proves `pawn` reaches its goal on a board whose
    /// full blocked-step set is `grids`. Cheap enough to re-check on every reuse,
    /// which is what makes inheriting a parent's paths sound no matter which
    /// move produced this node.
    #[inline]
    pub fn valid_for(&self, grids: &WallGrids, pawn: u128) -> bool {
        self.start == pawn && !self.cut_by(grids)
    }

    /// True when `delta` blocks at least one step of this path — i.e. the path
    /// alone no longer proves the player still gets home, so doubt remains.
    #[inline]
    pub fn cut_by(&self, delta: &WallGrids) -> bool {
        ((self.east & delta.east)
            | (self.west & delta.west)
            | (self.north & delta.north)
            | (self.south & delta.south))
            != 0
    }

    /// If the pawn moved along this path, truncate the path to start from the
    /// pawn's current position and return true.  Returns false if the pawn is
    /// not on the path (path should be dropped).
    ///
    /// The path is a sequence of steps from `start` to the goal.  Each cell on
    /// the path has exactly one direction bit set (the direction it leaves in).
    /// To truncate: walk from `start` along the path until we reach `pawn`,
    /// removing each intermediate cell's direction bit.  If we reach `pawn`,
    /// update `start` to `pawn` and we're done.  If we reach the goal without
    /// finding `pawn`, the pawn is not on the path.
    pub fn truncate_to(&mut self, pawn: u128) -> bool {
        if self.start == pawn {
            return true;
        }
        let mut cur = self.start;
        let mut steps_removed = 0u8;
        // Walk along the path branchlessly.  Each cell has exactly one
        // direction bit set (the direction it leaves in), so all 4
        // shifts can be computed simultaneously — only one produces a
        // non-zero result.  8 bit ops per step, 1 predictable branch.
        for _ in 0..80 {
            let next = ((cur & self.east) << 1)
                | ((cur & self.west) >> 1)
                | ((cur & self.south) << FLOOD_STRIDE)
                | ((cur & self.north) >> FLOOD_STRIDE);
            if next == 0 {
                return false; // reached goal without finding pawn
            }
            self.east &= !cur;
            self.west &= !cur;
            self.south &= !cur;
            self.north &= !cur;
            cur = next;
            steps_removed += 1;
            if cur == pawn {
                self.start = pawn;
                self.length -= steps_removed;
                return true;
            }
        }
        false
    }
}

/// Extract a `PathWitness` from eval's goal-distance layers.  The layers are
/// produced by `flood_into_layers(goal_bits, masks, ...)` — `layers[d]` = cells
/// at distance d from the goal.  To find the pawn's shortest path, locate the
/// pawn's layer (its distance), then backtrack from pawn at layer d through
/// decreasing layers to layer 0 (the goal), recording each step's direction.
///
/// This lets eval's flood double as the witness bootstrap: no separate witness
/// flood is needed at nodes where eval has already flooded.
pub fn path_witness_from_eval_layers(
    pawn_bit: u128,
    layers: &[u128; 81],
    depth: usize,
    masks: DirMasks,
) -> Option<PathWitness> {
    // Find the pawn's distance (which layer it's in).
    let mut pawn_dist = 0usize;
    let mut found = false;
    for d in 0..depth {
        if layers[d] & pawn_bit != 0 {
            pawn_dist = d;
            found = true;
            break;
        }
    }
    if !found {
        return None;
    }
    if pawn_dist == 0 {
        // Pawn is already on the goal.
        return Some(PathWitness {
            start: pawn_bit,
            length: 0,
            ..PathWitness::default()
        });
    }
    // Backtrack from pawn at layer pawn_dist down to layer 0.
    let mut cur = pawn_bit;
    let mut path = PathWitness {
        start: pawn_bit,
        length: pawn_dist as u8,
        ..PathWitness::default()
    };
    for d in (1..=pawn_dist).rev() {
        let prev_layer = layers[d - 1];
        let pred_mask = expand_frontier(cur, masks) & prev_layer;
        if pred_mask == 0 {
            return None;
        }
        let pred = pred_mask & pred_mask.wrapping_neg();
        let diff = cur.trailing_zeros() as i32 - pred.trailing_zeros() as i32;
        match diff {
            1 => path.east |= pred,
            -1 => path.west |= pred,
            d2 if d2 == FLOOD_STRIDE as i32 => path.south |= pred,
            d2 if d2 == -(FLOOD_STRIDE as i32) => path.north |= pred,
            _ => return None,
        }
        cur = pred;
    }
    Some(path)
}

/// Deepest BFS layer we will record.  A 9x9 board cannot need more.
pub const MAX_PATH_LAYERS: usize = 96;

/// Lee-wave flood that keeps its layer stack and reconstructs ONE shortest path.
///
/// Returns `None` exactly when the goal is unreachable, so this doubles as the
/// legality test for the player it is run for — a caller that needs a fresh
/// witness never has to flood twice: one pass both proves a path exists and
/// hands back which path it is.
///
/// The visited set comes back too, so the second player's flood can still steal
/// these bits exactly as it does after a plain flood.
pub fn bff_path_to_goal_with_visited(
    start: u128,
    grids: &WallGrids,
    goal: u128,
) -> (Option<PathWitness>, u128) {
    let mut layers = [0u128; MAX_PATH_LAYERS];
    let mut visited = start & FLOOD_PLAYABLE;
    layers[0] = visited;
    let origin = visited;
    if visited & goal != 0 {
        return (
            Some(PathWitness {
                start: origin,
                length: 0,
                ..PathWitness::default()
            }),
            visited,
        );
    }
    let mut wave = visited;
    let mut depth = 0usize;
    loop {
        wave = expand_wave(wave, grids) & !visited;
        if wave == 0 {
            return (None, visited);
        }
        depth += 1;
        visited |= wave;
        layers[depth] = wave;
        if wave & goal != 0 {
            break;
        }
        if depth + 1 >= MAX_PATH_LAYERS {
            return (None, visited);
        }
    }

    // Walk back goal -> start, one layer at a time.  `expand_wave` is the
    // flood-bit-space equivalent of the pawn LUT with enemy_key=0: it gives
    // the wall-gated cardinal neighbours of a cell (no diagonal jumps, no
    // opponent).  Reusing it here means the wall gating is done once in the
    // shift expression — no separate per-direction wall ANDs or sequential
    // branches.  The predecessor is whichever wall-reachable neighbour of
    // `cur` sits in the previous layer.
    let mut cur = (wave & goal) & (wave & goal).wrapping_neg();
    let mut path = PathWitness {
        start: origin,
        ..PathWitness::default()
    };
    for i in (1..=depth).rev() {
        let prev = layers[i - 1];
        let pred_mask = expand_wave(cur, grids) & prev;
        // Every layer-i cell was reached from layer i-1 by construction.
        debug_assert!(pred_mask != 0, "no predecessor for path backtrack");
        if pred_mask == 0 {
            return (None, visited);
        }
        // Pick the lowest-set-bit predecessor deterministically.
        let pred = pred_mask & pred_mask.wrapping_neg();
        // Direction from the bit-position difference: cur_bit - pred_bit.
        //   +1            → pred is west  of cur → path steps east  from pred
        //   -1            → pred is east  of cur → path steps west  from pred
        //   +FLOOD_STRIDE → pred is north of cur → path steps south from pred
        //   -FLOOD_STRIDE → pred is south of cur → path steps north from pred
        let diff = cur.trailing_zeros() as i32 - pred.trailing_zeros() as i32;
        match diff {
            1 => path.east |= pred,
            -1 => path.west |= pred,
            d if d == FLOOD_STRIDE as i32 => path.south |= pred,
            d if d == -(FLOOD_STRIDE as i32) => path.north |= pred,
            _ => unreachable!("invalid predecessor direction"),
        }
        cur = pred;
    }
    path.length = depth as u8;
    (Some(path), visited)
}

/// [`bff_path_to_goal_with_visited`] when the visited set is not needed.
#[inline]
pub fn bff_path_to_goal(start: u128, grids: &WallGrids, goal: u128) -> Option<PathWitness> {
    bff_path_to_goal_with_visited(start, grids, goal).0
}

/// Layered cached flood: same as `bff_path_to_goal_with_visited` but with
/// bit-theft from `cache` (P1's visited set).  When the wave first touches
/// the cache, the entire cached region is annexed into the current layer —
/// so the backtrack can walk through annexed cells just like normal layers.
/// Returns both the path witness and the full visited set.
///
/// This lets the bootstrap case mint a P2 witness without losing bit-theft
/// speed: P2 still steals P1's reachable region, but now also gets a path.
/// Result of a layered cached flood: path found, bit-theft proven (no path,
/// but pre-theft layers stored for later resume), or unreachable.
pub enum CachedPathResult {
    /// P2 reached the goal on its own — path extracted, layers sound.
    Path(PathWitness, u128),
    /// P2 touched P1's cached region — proven reachable, but no path.
    /// Carries P2's pre-theft layers, visited, and frontier so a later
    /// single-side-doubt flood can resume from this point instead of
    /// restarting from scratch.
    BitTheftProven {
        visited: u128,
        layers: Box<[u128; MAX_PATH_LAYERS]>,
        depth: usize,
    },
    /// P2 cannot reach the goal — wall is illegal.
    Unreachable(u128),
}

/// Resume a P2 flood from stored pre-theft layers.  The layers represent
/// P2's BFS distances from its start on a PREVIOUS wall configuration.
/// The new `grids` may differ (different trial wall).  If the new delta
/// blocks an edge within the stored visited region, the layers are stale
/// and the caller must restart from scratch — check `delta_touches_visited`
/// first.
pub fn bff_resume_path_to_goal(
    layers: &[u128; MAX_PATH_LAYERS],
    depth: usize,
    visited: u128,
    grids: &WallGrids,
    goal: u128,
) -> (Option<PathWitness>, u128) {
    // Work buffer: copy of the immutable pre-theft layers.  Resume flood
    // appends new layers into this buffer so the backtrack can walk the
    // full stack.
    let mut work_layers = *layers;
    let origin = work_layers[0];
    let mut wave = work_layers[depth];
    let mut visited = visited;
    let mut depth = depth;
    // Check if we already reached the goal in the stored layers.
    if visited & goal != 0 {
        for i in (1..=depth).rev() {
            if work_layers[i] & goal != 0 {
                wave = work_layers[i] & goal;
                depth = i;
                break;
            }
        }
    } else {
        // Continue flooding from the stored frontier with the new grids,
        // appending each new layer into the work buffer.
        loop {
            wave = expand_wave(wave, grids) & !visited;
            if wave == 0 {
                return (None, visited);
            }
            depth += 1;
            visited |= wave;
            work_layers[depth] = wave;
            if depth >= MAX_PATH_LAYERS {
                return (None, visited);
            }
            if wave & goal != 0 {
                break;
            }
        }
    }

    // Walk back goal -> start using the full work buffer.
    let mut cur = (wave & goal) & (wave & goal).wrapping_neg();
    let mut path = PathWitness {
        start: origin,
        ..PathWitness::default()
    };
    for i in (1..=depth).rev() {
        let prev = work_layers[i - 1];
        let pred_mask = expand_wave(cur, grids) & prev;
        if pred_mask == 0 {
            return (None, visited);
        }
        let pred = pred_mask & pred_mask.wrapping_neg();
        let diff = cur.trailing_zeros() as i32 - pred.trailing_zeros() as i32;
        match diff {
            1 => path.east |= pred,
            -1 => path.west |= pred,
            d if d == FLOOD_STRIDE as i32 => path.south |= pred,
            d if d == -(FLOOD_STRIDE as i32) => path.north |= pred,
            _ => return (None, visited),
        }
        cur = pred;
    }
    path.length = depth as u8;
    (Some(path), visited)
}

/// True when `delta` blocks an edge within `visited` — meaning the stored
/// pre-theft layers are stale and the flood cannot safely resume.
#[inline]
pub fn delta_touches_visited(delta: &WallGrids, visited: u128) -> bool {
    // delta.east marks cells whose east step is blocked (step from C to C<<1).
    // If C is in visited AND C's east neighbor (C<<1) is also in visited, the
    // blocked edge is within the visited region — layers are stale.
    let east_blocked = delta.east & visited;
    let west_blocked = delta.west & visited;
    let south_blocked = delta.south & visited;
    let north_blocked = delta.north & visited;
    ((east_blocked << 1) & visited != 0)
        || ((west_blocked >> 1) & visited != 0)
        || ((south_blocked << FLOOD_STRIDE) & visited != 0)
        || ((north_blocked >> FLOOD_STRIDE) & visited != 0)
}

pub fn bff_path_to_goal_cached_with_visited(
    start: u128,
    cache: u128,
    grids: &WallGrids,
    goal: u128,
) -> CachedPathResult {
    let mut layers = [0u128; MAX_PATH_LAYERS];
    let mut visited = start & FLOOD_PLAYABLE;
    layers[0] = visited;
    let origin = visited;
    if visited & goal != 0 {
        return CachedPathResult::Path(
            PathWitness {
                start: origin,
                length: 0,
                ..PathWitness::default()
            },
            visited,
        );
    }
    let mut wave = visited;
    let mut pool = cache & !visited;
    let mut depth = 0usize;
    let mut layers_frozen = false;
    // Pre-theft state: saved when bit-theft triggers, so a later flood can
    // resume from this point instead of restarting from scratch.
    let mut pre_theft_layers = Box::new([0u128; MAX_PATH_LAYERS]);
    let mut pre_theft_depth = 0usize;
    loop {
        // Bit theft: on first contact with the cached region, annex P1's
        // reachable set and keep flooding for the legality verdict — but
        // freeze layer storage.  Annexed cells have arbitrary P2-distances,
        // so any layer at or past this point is unsound for path extraction.
        // P2 gets a witness only when it reaches the goal on its own before
        // ever touching P1's territory.
        if !layers_frozen && pool != 0 && wave & pool != 0 {
            // Save pre-theft state: layers[0..depth] and the frontier wave
            // are P2's own BFS progress, all at correct distances.
            pre_theft_layers = Box::new(layers);
            pre_theft_depth = depth;
            visited |= pool;
            wave |= pool;
            pool = 0;
            layers_frozen = true;
            if visited & goal != 0 {
                return CachedPathResult::BitTheftProven {
                    visited,
                    layers: pre_theft_layers,
                    depth: pre_theft_depth,
                };
            }
            // Continue flooding for legality, but don't store layers.
            continue;
        }
        wave = expand_wave(wave, grids) & !visited;
        if wave == 0 {
            return CachedPathResult::Unreachable(visited);
        }
        visited |= wave;
        if !layers_frozen {
            depth += 1;
            layers[depth] = wave;
        }
        if wave & goal != 0 {
            if layers_frozen {
                return CachedPathResult::BitTheftProven {
                    visited,
                    layers: pre_theft_layers,
                    depth: pre_theft_depth,
                };
            }
            break;
        }
        if !layers_frozen && depth + 1 >= MAX_PATH_LAYERS {
            return CachedPathResult::Unreachable(visited);
        }
    }

    // P2 reached the goal on its own — layers are sound, extract the path.
    let mut cur = (wave & goal) & (wave & goal).wrapping_neg();
    let mut path = PathWitness {
        start: origin,
        ..PathWitness::default()
    };
    for i in (1..=depth).rev() {
        let prev = layers[i - 1];
        let pred_mask = expand_wave(cur, grids) & prev;
        debug_assert!(pred_mask != 0, "no predecessor for cached path backtrack");
        if pred_mask == 0 {
            return CachedPathResult::Unreachable(visited);
        }
        let pred = pred_mask & pred_mask.wrapping_neg();
        let diff = cur.trailing_zeros() as i32 - pred.trailing_zeros() as i32;
        match diff {
            1 => path.east |= pred,
            -1 => path.west |= pred,
            d if d == FLOOD_STRIDE as i32 => path.south |= pred,
            d if d == -(FLOOD_STRIDE as i32) => path.north |= pred,
            _ => unreachable!("invalid predecessor direction"),
        }
        cur = pred;
    }
    path.length = depth as u8;
    CachedPathResult::Path(path, visited)
}


/// Convenience wrapper for one-off queries (oracle / replay validation).
pub fn bff_wall_legal_board(board: &Board) -> bool {
    let grids = WallGrids::from_board(board);
    let (r1, c1) = board.pawn(Player::One);
    let (r2, c2) = board.pawn(Player::Two);
    bff_wall_legal(pawn_bit(r1, c1), pawn_bit(r2, c2), &grids)
}

// ─────────────────────────────────────────────────────────────────────────────
// Kogge-Stone occluded fill (research only — SLOWER on 9×9, do NOT wire in)
// ─────────────────────────────────────────────────────────────────────────────
//
// `expand_wave` advances the frontier ONE square per iteration, so the BFS loop
// runs once per unit of path *length* (~9–40 on a snaking board). The occluded
// fill instead smears the frontier along an entire open run in O(log w) shifts,
// so the loop runs once per *turn* in the path (~2–8). Same answer, fewer iters.
//
// VERDICT (benches/flood_modes.rs, regime study over startpos + 15 canta
// middlegames, native build): KS is SLOWER in EVERY regime — 0.63–0.97× on the
// hot flood candidates perft actually runs, and even 0.68× on the wide-open
// startpos board it was supposed to dominate. Wiring KS into the movegen hot
// path is what regressed perft(5) from ~11 s to ~15 s.
//
// Why the "open board → KS is hundredfold faster" intuition fails here: that
// holds on a LARGE graph where path length ≫ setup cost. Quoridor is 9×9 (≤8
// rings open, ~16 walled), so the one-step flood finishes in a handful of
// iterations, while KS pays `KsProp::new` (24 shift/AND ops) TWICE per call
// (P1 + P2, rebuilt per wall trial because each trial's delta changes the
// propagators). The setup never amortises on a board this small. The remembered
// "hundredfold" was KS vs the naive QUEUE BFS — but one-step bitboard flood fill
// (`expand_wave` / `bff_wall_legal`) already captures that win on 9×9. There is
// NO board-difficulty crossover to adapt to on 9×9: `bff_wall_legal` wins everywhere.
// KS is kept, correct, and oracle-tested purely as a larger-board reference.
//
// Why no anti-wrap file masks are needed (the POC's ray-sweep bug does NOT apply):
// the propagator `p` is `(!blocked >> shift) & FLOOD_PLAYABLE`, so it is ZERO on
// every buffer-ring bit. The doubling step `p &= p << s` ANDs the propagator with
// its own shift, so any run that would have to cross a buffer column picks up that
// zero and stops. An east jump of 2/4/8 from cols 7/8 lands on a buffer bit (or a
// next-row bit reachable only *through* a buffer bit), where the doubled
// propagator is 0 — so the leak `expand_wave`'s critics feared cannot occur.
// `random_walls_match_naive_reference` exercises this against the scalar BFS.

/// Precomputed occluded-fill propagator stages for one `WallGrids`.
///
/// Each direction's propagator is `(!blocked >> shift) & FLOOD_PLAYABLE` and is
/// *constant* across every super-step of a flood — so we build the doubled stages
/// (`p`, `p & p<<s`, …) once per flood instead of per iteration. We carry shifts
/// 1,2,4 only: a fully-open 9-cell run then fills in two super-steps instead of
/// one, which is cheaper on average than paying a 4th (shift-8) round every step.
struct KsProp {
    // east (shift +1), stages for shifts 1 and 2; shift-4 reaches the rest.
    e1: u128,
    e2: u128,
    e4: u128,
    w1: u128,
    w2: u128,
    w4: u128,
    s1: u128,
    s2: u128,
    s4: u128,
    n1: u128,
    n2: u128,
    n4: u128,
}

impl KsProp {
    #[inline]
    fn new(grids: &WallGrids) -> Self {
        const S: u32 = FLOOD_STRIDE;
        let e1 = (!grids.east << 1) & FLOOD_PLAYABLE;
        let e2 = e1 & (e1 << 1);
        let e4 = e2 & (e2 << 2);
        let w1 = (!grids.west >> 1) & FLOOD_PLAYABLE;
        let w2 = w1 & (w1 >> 1);
        let w4 = w2 & (w2 >> 2);
        let s1 = (!grids.south << S) & FLOOD_PLAYABLE;
        let s2 = s1 & (s1 << S);
        let s4 = s2 & (s2 << (2 * S));
        let n1 = (!grids.north >> S) & FLOOD_PLAYABLE;
        let n2 = n1 & (n1 >> S);
        let n4 = n2 & (n2 >> (2 * S));
        Self {
            e1,
            e2,
            e4,
            w1,
            w2,
            w4,
            s1,
            s2,
            s4,
            n1,
            n2,
            n4,
        }
    }

    /// One occluded super-step: horizontal fill then vertical fill on the result,
    /// so each step grows an L-shaped (single-turn) region instead of one ring.
    #[inline]
    fn expand(&self, wave: u128) -> u128 {
        const S: u32 = FLOOD_STRIDE;
        let mut g = wave;
        // east
        g |= self.e1 & (g << 1);
        g |= self.e2 & (g << 2);
        g |= self.e4 & (g << 4);
        // west
        g |= self.w1 & (g >> 1);
        g |= self.w2 & (g >> 2);
        g |= self.w4 & (g >> 4);
        // south
        g |= self.s1 & (g << S);
        g |= self.s2 & (g << (2 * S));
        g |= self.s4 & (g << (4 * S));
        // north
        g |= self.n1 & (g >> S);
        g |= self.n2 & (g >> (2 * S));
        g |= self.n4 & (g >> (4 * S));
        g
    }
}

/// Alternative bitboard flood implementation (Kogge–Stone occluded fill). Same
/// path-to-goal semantics as [`bff_to_goal`]; kept for bench comparison only.
#[inline]
pub fn bff_ks_to_goal(start: u128, grids: &WallGrids, goal: u128) -> (bool, u128) {
    let mut visited = start & FLOOD_PLAYABLE;
    if visited & goal != 0 {
        return (true, visited);
    }
    let prop = KsProp::new(grids);
    loop {
        let next = prop.expand(visited);
        if next & goal != 0 {
            return (true, next);
        }
        if next == visited {
            return (false, visited);
        }
        visited = next;
    }
}

/// Alternative implementation of [`bff_to_goal_cached`] (Player 2, visited-bit reuse).
#[inline]
pub fn bff_ks_to_goal_cached(start: u128, cache: u128, grids: &WallGrids, goal: u128) -> bool {
    let mut visited = start & FLOOD_PLAYABLE;
    if visited & goal != 0 {
        return true;
    }
    let prop = KsProp::new(grids);
    let mut pool = cache & !visited;
    loop {
        let mut next = prop.expand(visited);
        if pool != 0 && next & pool != 0 {
            if pool & goal != 0 {
                return true;
            }
            next |= pool;
            pool = 0;
        }
        if next & goal != 0 {
            return true;
        }
        if next == visited {
            return false;
        }
        visited = next;
    }
}

/// Alternative implementation of [`bff_wall_legal`] (Kogge–Stone bench path only).
#[inline]
pub fn bff_ks_wall_legal(p1_start: u128, p2_start: u128, grids: &WallGrids) -> bool {
    let (ok1, p1_visited) = bff_ks_to_goal(p1_start, grids, P1_GOAL_BITS);
    if !ok1 {
        return false;
    }
    bff_ks_to_goal_cached(p2_start, p1_visited, grids, P2_GOAL_BITS)
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
        assert!(bff_wall_legal_board(&Board::new()));
    }

    #[test]
    fn adjacent_pawns_near_goal_regression() {
        // V10's partial-component shortcut returned false here (false negative).
        let mut board = Board::new();
        board.pawns[Player::One as usize] = (7, 4);
        board.pawns[Player::Two as usize] = (6, 4);
        assert!(bff_wall_legal_board(&board));
    }

    #[test]
    fn fully_caged_pawn_is_detected() {
        // Box P2's pawn start (8,4): walls below and on both sides.
        let mut board = Board::new();
        set_wall(&mut board, 7, 3, WallOrientation::Horizontal, true);
        set_wall(&mut board, 7, 3, WallOrientation::Vertical, true);
        set_wall(&mut board, 7, 4, WallOrientation::Vertical, true);
        assert!(!reach_goal_naive(&board, (8, 4), Player::Two));
        assert!(!bff_wall_legal_board(&board));
    }

    #[test]
    fn theft_pool_goal_is_detected() {
        // P1 ahead of P2 so that P1's early-exit flood owns the row-0 cells
        // P2 needs; the annexed pool itself must satisfy P2's goal (fix #4).
        let mut board = Board::new();
        board.pawns[Player::One as usize] = (1, 4);
        board.pawns[Player::Two as usize] = (2, 4);
        assert!(bff_wall_legal_board(&board));
    }

    #[test]
    fn random_walls_match_naive_reference() {
        // Deterministic LCG — no rand dependency.
        let mut state = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
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
            let got = bff_wall_legal(pawn_bit(p1.0, p1.1), pawn_bit(p2.0, p2.1), &grids);
            assert_eq!(
                got, expected,
                "walls h={:#x} v={:#x} p1={:?} p2={:?}",
                board.horizontal_walls, board.vertical_walls, p1, p2
            );

            // Kogge-Stone occluded fill must agree with the step-by-step flood.
            let got_ks = bff_ks_wall_legal(pawn_bit(p1.0, p1.1), pawn_bit(p2.0, p2.1), &grids);
            assert_eq!(
                got_ks, expected,
                "KS walls h={:#x} v={:#x} p1={:?} p2={:?}",
                board.horizontal_walls, board.vertical_walls, p1, p2
            );

            // Single-player floods must match the reference too.
            let (got1, vis1) = bff_to_goal(pawn_bit(p1.0, p1.1), &grids, P1_GOAL_BITS);
            assert_eq!(got1, reach_goal_naive(&board, p1, Player::One));
            let (got2, _) = bff_to_goal(pawn_bit(p2.0, p2.1), &grids, P2_GOAL_BITS);
            assert_eq!(got2, reach_goal_naive(&board, p2, Player::Two));

            // KS single-player reachability + visited-set parity.
            let (got1_ks, vis1_ks) = bff_ks_to_goal(pawn_bit(p1.0, p1.1), &grids, P1_GOAL_BITS);
            assert_eq!(got1_ks, got1, "KS p1 reach mismatch");
            // When neither reaches the goal the full reachable set must be identical;
            // on a goal hit either may early-exit with a partial set, so only compare
            // the reached flag there.
            if !got1 {
                assert_eq!(vis1_ks, vis1, "KS p1 visited-set mismatch (no goal)");
            }
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
