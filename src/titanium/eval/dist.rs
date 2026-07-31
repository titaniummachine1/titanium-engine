//! ACE-native distance fields via Binary Flood Fill (`pathfinding::bff`).
//!
//! Floods run directly in ACE cell index (row 0 = top) using `DirMasks::from_ace_game`
//! — no Titanium `Board` rebuild and no row-flip remap. Each layer is one
//! `expand_frontier` u128 step (same binary / bitboard flood family as CAT / movegen `bff_*`).

use crate::pathfinding::bff::{expand_frontier, flood_distance_field};
use crate::pathfinding::masks::DirMasks;
use crate::titanium::game::GameState;
use crate::util::grid::{flood_bit_sq, FLOOD_BIT_BY_SQ, FLOOD_PLAYABLE, FLOOD_SQ_BY_BIT};

/// Corridor cells considered for choke detection (matches CAT bottleneck band).
pub const CHOKE_DELTA_MAX: u8 = 2;

const fn ace_goal_bits(row: usize) -> u128 {
    let mut bits = 0u128;
    let mut c = 0usize;
    while c < 9 {
        bits |= FLOOD_BIT_BY_SQ[row * 9 + c];
        c += 1;
    }
    bits
}

const ACE_P0_GOAL_BITS: u128 = ace_goal_bits(0);
const ACE_P1_GOAL_BITS: u128 = ace_goal_bits(8);

#[inline]
pub fn ace_goal_bits_for_player(player: usize) -> u128 {
    if player == 0 {
        ACE_P0_GOAL_BITS
    } else {
        ACE_P1_GOAL_BITS
    }
}

/// Scatter BFS layers from `seed` into `out` (255 = unreachable).
fn flood_scatter(seed: u128, masks: DirMasks, out: &mut [u8; 81]) {
    crate::bench_instr::record(
        |b| &mut b.flood_scatter,
        || flood_scatter_inner(seed, masks, out),
    )
}

fn flood_scatter_inner(seed: u128, masks: DirMasks, out: &mut [u8; 81]) {
    flood_distance_field(seed, masks, out);
}

/// Layer-native BFS flood: records each `expand_frontier` wavefront as a `u128`
/// layer mask into `layers` (no per-cell "bit stealing" scatter). Returns the
/// number of layers written. `layers[d]` holds exactly the cells at BFS distance
/// `d` from `seed`. This is the cheap form the search uses for distance features
/// (width via `popcount`, a square's distance via `dist_in_layers`) — the dense
/// `[u8;81]` field is only materialized when a consumer truly needs random reads.
pub fn flood_into_layers(seed: u128, masks: DirMasks, layers: &mut [u128; 81]) -> usize {
    let mut reached = seed & FLOOD_PLAYABLE;
    let mut frontier = reached;
    layers[0] = frontier;
    let mut depth = 1usize;
    while frontier != 0 && depth < 81 {
        let new = expand_frontier(frontier, masks) & !reached & FLOOD_PLAYABLE;
        if new == 0 {
            break;
        }
        layers[depth] = new;
        depth += 1;
        reached |= new;
        frontier = new;
    }
    // Zero any stale tail so popcount/lookups over `..depth` stay clean on reuse.
    for slot in layers.iter_mut().take(81).skip(depth) {
        *slot = 0;
    }
    depth
}

/// Inverse layer flood: distance-to-goal-row layer masks for `player` (ACE index).
/// `layers[d]` = cells `d` steps from the goal row. No dense scatter.
pub fn fill_ace_dist_layers_to_goal(
    player: usize,
    masks: DirMasks,
    layers: &mut [u128; 81],
) -> usize {
    flood_into_layers(ace_goal_bits_for_player(player), masks, layers)
}

#[inline]
pub fn fill_ace_dist_layers_to_goal_p0(masks: DirMasks, layers: &mut [u128; 81]) -> usize {
    flood_into_layers(ACE_P0_GOAL_BITS, masks, layers)
}

#[inline]
pub fn fill_ace_dist_layers_to_goal_p1(masks: DirMasks, layers: &mut [u128; 81]) -> usize {
    flood_into_layers(ACE_P1_GOAL_BITS, masks, layers)
}

pub fn materialize_distance_layers(layers: &[u128; 81], depth: usize, out: &mut [u8; 81]) {
    crate::bench_instr::record(
        |b| &mut b.mat_layers,
        || materialize_distance_layers_inner(layers, depth, out),
    )
}

fn materialize_distance_layers_inner(layers: &[u128; 81], depth: usize, out: &mut [u8; 81]) {
    out.fill(255);
    for (d, mut bits) in layers.iter().copied().take(depth).enumerate() {
        while bits != 0 {
            let fb = bits.trailing_zeros();
            bits &= bits - 1;
            let sq = FLOOD_SQ_BY_BIT[fb as usize];
            if sq != u8::MAX {
                out[sq as usize] = d as u8;
            }
        }
    }
}

/// BFS distance of square `sq` from the layer masks (255 = unreachable). O(depth)
/// bit tests — replaces a dense `dist[sq]` random read.
#[inline]
pub fn dist_in_layers(layers: &[u128; 81], depth: usize, sq: u8) -> u8 {
    let bit = flood_bit_sq(sq);
    for (d, layer) in layers.iter().take(depth).enumerate() {
        if layer & bit != 0 {
            return d as u8;
        }
    }
    255
}

/// Count of squares at BFS distance `d` (the "width" eval feature) — `popcount`
/// of one layer mask instead of scanning all 81 cells.
#[inline]
pub fn width_in_layers(layers: &[u128; 81], depth: usize, d: u8) -> u32 {
    let di = d as usize;
    if di >= depth {
        return 0;
    }
    (layers[di] & FLOOD_PLAYABLE).count_ones()
}

/// Wall placement geometry for incremental edge-cut tests (dense wall ids).
#[inline]
pub fn wall_incr_probe_squares(m: i16) -> Option<(usize, usize, usize, usize)> {
    if !crate::titanium::is_wall_move(m) {
        return None;
    }
    let slot = crate::titanium::wall_slot(m);
    let a = (slot >> 3) * 9 + (slot & 7);
    let (b2, c2, e2) = if crate::titanium::is_hwall_move(m) {
        (a + 9, a + 1, a + 10) // hw: two vertical edges
    } else {
        (a + 1, a + 9, a + 10) // vw: two horizontal edges
    };
    Some((a, b2, c2, e2))
}

/// True when a newly placed wall may change this player's goal-distance field.
///
/// Uses the ACE incremental test: a wall only affects shortest-path distances
/// when it blocks an edge between cells whose goal-distance differs (BFS
/// gradient edge on some shortest route).
#[inline]
pub fn wall_incr_cuts_player_dist(d: &[u8; 81], m: i16) -> bool {
    let Some((a, b2, c2, e2)) = wall_incr_probe_squares(m) else {
        return false;
    };
    d[a] != d[b2] || d[c2] != d[e2]
}

/// Cells on this player's shortest-path DAG: every square reachable from the
/// pawn by strictly descending the goal-distance gradient. Bit `i` = ACE square
/// `i` (81 bits, low-order).
///
/// This is the union of all of the player's shortest routes — the same object as
/// `slack == 0` in the slack-plane formulation, and strictly more information
/// than a sample of node-disjoint witness paths. Computed by one descent BFS
/// over `d`, which `refresh_dist` already maintains, so it costs no flood.
pub fn dag_cells_from_pawn(g: &GameState, pawn: usize, d: &[u8; 81]) -> u128 {
    if pawn >= 81 || d[pawn] == u8::MAX {
        return 0;
    }
    let mut seen = 1u128 << pawn;
    let mut stack = [0u8; 81];
    let mut top = 1usize;
    stack[0] = pawn as u8;
    while top > 0 {
        top -= 1;
        let u = stack[top] as usize;
        let du = d[u];
        if du == 0 {
            continue;
        }
        for dir in 0..4usize {
            if !g.can_step(u, dir) {
                continue;
            }
            let v = (u as i16 + crate::titanium::game::DELTA[dir]) as usize;
            if d[v] != du - 1 || (seen >> v) & 1 == 1 {
                continue;
            }
            seen |= 1u128 << v;
            stack[top] = v as u8;
            top += 1;
        }
    }
    seen
}

/// True when the wall blocks an edge of this player's shortest-path DAG.
///
/// This is the reference engine's LMR/LMP "tight edge" classifier. An edge
/// `(u, v)` lies on a shortest route iff its endpoints sit on adjacent distance
/// rings and the deeper endpoint is itself reachable by descent from the pawn —
/// checking that endpoint suffices, because descending through it necessarily
/// puts the shallower endpoint on the DAG too.
///
/// Walls whose two blocked edges both miss the DAG cannot change this player's
/// distance at all, which is what makes them safe to reduce hard.
#[inline]
pub fn wall_cuts_player_dag(dag: u128, d: &[u8; 81], m: i16) -> bool {
    let Some((a, b2, c2, e2)) = wall_incr_probe_squares(m) else {
        return false;
    };
    dag_edge(dag, d, a, b2) || dag_edge(dag, d, c2, e2)
}

#[inline]
fn dag_edge(dag: u128, d: &[u8; 81], u: usize, v: usize) -> bool {
    let (du, dv) = (d[u], d[v]);
    if du == u8::MAX || dv == u8::MAX {
        return false;
    }
    // Equal rings lie on no shortest route; otherwise test the deeper endpoint.
    if du == dv + 1 {
        (dag >> u) & 1 == 1
    } else if dv == du + 1 {
        (dag >> v) & 1 == 1
    } else {
        false
    }
}

/// Per-player incremental refresh flags for a single wall added on top of `d0`/`d1`.
#[inline]
pub fn wall_incr_refresh_flags(d0: &[u8; 81], d1: &[u8; 81], m: i16) -> (bool, bool) {
    (
        wall_incr_cuts_player_dist(d0, m),
        wall_incr_cuts_player_dist(d1, m),
    )
}

/// Inverse flood: distance from each cell to `player`'s goal row (ACE index).
pub fn fill_ace_dist_to_goal(g: &GameState, player: usize, ace_dist: &mut [u8; 81]) {
    let masks = DirMasks::from_ace_game(g);
    fill_ace_dist_to_goal_with_masks(player, masks, ace_dist);
}

/// Inverse flood with caller-provided topology masks. Search refreshes both
/// players on the same wall geometry, so constructing the masks once avoids a
/// duplicate 81-cell topology scan.
pub fn fill_ace_dist_to_goal_with_masks(player: usize, masks: DirMasks, ace_dist: &mut [u8; 81]) {
    flood_scatter(ace_goal_bits_for_player(player), masks, ace_dist);
}

#[inline]
pub fn fill_ace_dist_to_goal_with_masks_p0(masks: DirMasks, ace_dist: &mut [u8; 81]) {
    flood_scatter(ACE_P0_GOAL_BITS, masks, ace_dist);
}

#[inline]
pub fn fill_ace_dist_to_goal_with_masks_p1(masks: DirMasks, ace_dist: &mut [u8; 81]) {
    flood_scatter(ACE_P1_GOAL_BITS, masks, ace_dist);
}

/// Forward flood: steps from `ace_start` pawn square (ACE index).
pub fn fill_ace_dist_from_pawn(g: &GameState, ace_start: usize, ace_dist: &mut [u8; 81]) {
    let masks = DirMasks::from_ace_game(g);
    fill_ace_dist_from_pawn_with_masks(ace_start, masks, ace_dist);
}

#[inline]
pub fn fill_ace_dist_from_pawn_with_masks(
    ace_start: usize,
    masks: DirMasks,
    ace_dist: &mut [u8; 81],
) {
    if ace_start >= 81 {
        ace_dist.fill(255);
        return;
    }
    flood_scatter(flood_bit_sq(ace_start as u8), masks, ace_dist);
}

/// CAT-style corridor delta: dist_from_pawn + dist_to_goal − shortest (multi-path band).
pub fn fill_corridor_delta(from: &[u8; 81], to: &[u8; 81], shortest: u8, out: &mut [u8; 81]) {
    for i in 0..81 {
        let f = from[i];
        let t = to[i];
        if f == 255 || t == 255 || shortest == 255 {
            out[i] = 255;
        } else {
            out[i] = (u16::from(f) + u16::from(t)).saturating_sub(u16::from(shortest)) as u8;
        }
    }
}

/// Build sparse shortest-route and one-tempo-alternative masks.
pub fn fill_route_masks(
    from: &[u8; 81],
    to: &[u8; 81],
    shortest: u8,
    route: &mut [u8; 81],
    near: &mut [u8; 81],
) {
    route.fill(0);
    near.fill(0);
    if shortest == 255 {
        return;
    }
    for i in 0..81 {
        if from[i] == 255 || to[i] == 255 {
            continue;
        }
        let total = u16::from(from[i]) + u16::from(to[i]);
        if total == u16::from(shortest) {
            route[i] = 1;
        } else if total == u16::from(shortest) + 1 {
            near[i] = 1;
        }
    }
}

pub fn fill_distance_layers(to_goal: &[u8; 81], layers: &mut [u128; 81]) {
    layers.fill(0);
    for sq in 0..81u8 {
        let d = to_goal[sq as usize];
        if d != 255 {
            layers[d as usize] |= flood_bit_sq(sq);
        }
    }
}

pub fn shortest_route_bits(
    pawn: usize,
    shortest: u8,
    layers: &[u128; 81],
    masks: DirMasks,
) -> u128 {
    if shortest == 255 {
        return 0;
    }
    let mut frontier = flood_bit_sq(pawn as u8);
    let mut route = frontier;
    for d in (1..=shortest as usize).rev() {
        frontier = expand_frontier(frontier, masks) & layers[d - 1];
        route |= frontier;
    }
    route
}

/// Shortest-route support via bit-parallel frontier expansion over decreasing
/// goal-distance layers. `flank` is the one-step neighborhood outside that
/// support (the grid's +2-tempo alternatives). No forward distance scatter is
/// needed, which keeps this suitable for every leaf evaluation.
pub fn fill_sparse_route_masks(
    g: &GameState,
    pawn: usize,
    to_goal: &[u8; 81],
    route: &mut [u8; 81],
    flank: &mut [u8; 81],
) {
    route.fill(0);
    flank.fill(0);
    let shortest = to_goal[pawn];
    if shortest == 255 {
        return;
    }
    let masks = DirMasks::from_ace_game(g);
    let mut layers = [0u128; 81];
    fill_distance_layers(to_goal, &mut layers);
    let route_bits = shortest_route_bits(pawn, shortest, &layers, masks);
    let flank_bits = expand_frontier(route_bits, masks) & !route_bits & FLOOD_PLAYABLE;
    for sq in 0..81u8 {
        let bit = flood_bit_sq(sq);
        route[sq as usize] = u8::from(route_bits & bit != 0);
        flank[sq as usize] = u8::from(flank_bits & bit != 0);
    }
}

fn neighbor_squares(sq: u8, masks: DirMasks, out: &mut [u8; 4]) -> usize {
    let bit = flood_bit_sq(sq);
    let mut n = 0usize;
    if masks.north & bit != 0 {
        out[n] = sq - 9;
        n += 1;
    }
    if masks.south & bit != 0 {
        out[n] = sq + 9;
        n += 1;
    }
    if masks.east & bit != 0 {
        out[n] = sq + 1;
        n += 1;
    }
    if masks.west & bit != 0 {
        out[n] = sq - 1;
        n += 1;
    }
    n
}

/// Encode route forcedness for NNUE: `1/(1+forward_continuations)` scaled to u8 (÷16 in eval).
#[inline]
pub fn encode_choke(forward_continuations: u8) -> u8 {
    ((16.0 / (1.0 + f64::from(forward_continuations))).round() as u8).min(16)
}

/// Shared cell importance: both players' route relevance `1/(1+delta_p0+delta_p1)`.
#[inline]
pub fn encode_contested(delta_p0: u8, delta_p1: u8) -> u8 {
    if delta_p0 == 255 || delta_p1 == 255 {
        return 0;
    }
    let sum = u16::from(delta_p0) + u16::from(delta_p1);
    ((16.0 / (1.0 + sum as f64)).round() as u8).min(16)
}

/// Count neighbors that continue forward on/near a shortest corridor (CAT flex test).
fn forward_continuations(
    sq: u8,
    masks: DirMasks,
    dist_from: &[u8; 81],
    dist_to: &[u8; 81],
    delta_arr: &[u8; 81],
) -> u8 {
    let from = dist_from[sq as usize];
    let to = dist_to[sq as usize];
    if from == 255 || to == 0 || to == 255 {
        return 0;
    }
    let mut neighbors = [0u8; 4];
    let n = neighbor_squares(sq, masks, &mut neighbors);
    let mut count = 0u8;
    for &next in &neighbors[..n] {
        let next_from = dist_from[next as usize];
        let next_to = dist_to[next as usize];
        if next_from == from.saturating_add(1)
            && next_to < to
            && delta_arr[next as usize] <= CHOKE_DELTA_MAX
        {
            count = count.saturating_add(1);
        }
    }
    count
}

/// Near-shortest band included in path-crossing counts (delta 0 = shortest, 1 = +1 tempo).
pub const CROSS_DELTA_MAX: u8 = 1;

/// Number of shortest / near-shortest routes through each cell (path overlap hot-spot).
/// Reuses existing distance fields + one O(81) DP — no extra flood.
pub fn fill_path_crossing(
    g: &GameState,
    dist_from: &[u8; 81],
    dist_to: &[u8; 81],
    shortest: u8,
    out: &mut [u8; 81],
) {
    out.fill(0);
    if shortest == 255 {
        return;
    }
    let masks = DirMasks::from_ace_game(g);
    let mut delta = [255u8; 81];
    fill_corridor_delta(dist_from, dist_to, shortest, &mut delta);

    let on_route = |sq: usize| -> bool {
        if delta[sq] > CROSS_DELTA_MAX {
            return false;
        }
        let t = u16::from(dist_from[sq]) + u16::from(dist_to[sq]);
        t == u16::from(shortest) || t == u16::from(shortest) + 1
    };

    let mut fwd = [0u16; 81];
    for sq in 0..81 {
        if dist_from[sq] == 0 && on_route(sq) {
            fwd[sq] = 1;
        }
    }
    for d in 1..=shortest.saturating_add(1) {
        for sq in 0..81usize {
            if dist_from[sq] != d || !on_route(sq) {
                continue;
            }
            let mut neighbors = [0u8; 4];
            let n = neighbor_squares(sq as u8, masks, &mut neighbors);
            let mut sum = 0u16;
            for &prev in &neighbors[..n] {
                let p = prev as usize;
                if dist_from[p] + 1 == d && dist_to[p] > dist_to[sq] && on_route(p) {
                    sum = sum.saturating_add(fwd[p]);
                }
            }
            fwd[sq] = sum;
        }
    }

    let mut bwd = [0u16; 81];
    for sq in 0..81 {
        if dist_to[sq] == 0 && on_route(sq) {
            bwd[sq] = 1;
        }
    }
    for d in 1..=shortest.saturating_add(1) {
        for sq in 0..81usize {
            if dist_to[sq] != d || !on_route(sq) {
                continue;
            }
            let mut neighbors = [0u8; 4];
            let n = neighbor_squares(sq as u8, masks, &mut neighbors);
            let mut sum = 0u16;
            for &next in &neighbors[..n] {
                let q = next as usize;
                if dist_to[q] + 1 == d && dist_from[sq] + 1 == dist_from[q] && on_route(q) {
                    sum = sum.saturating_add(bwd[q]);
                }
            }
            bwd[sq] = sum;
        }
    }

    for sq in 0..81 {
        if !on_route(sq) {
            continue;
        }
        out[sq] = fwd[sq].saturating_mul(bwd[sq]).min(255) as u8;
    }
}

/// Shared contested corridor: `1/(1+delta_p0+delta_p1)` per cell (both routes matter).
pub fn fill_contested(
    corridor_delta_p0: &[u8; 81],
    corridor_delta_p1: &[u8; 81],
    out: &mut [u8; 81],
) {
    for i in 0..81 {
        out[i] = encode_contested(corridor_delta_p0[i], corridor_delta_p1[i]);
    }
}

/// Route forcedness per cell: local branching factor `1/(1+forward_continuations)`.
/// Stored u8 is `round(16 × value)`; eval divides by 16 (0 cont → 16/16 = 1.0, 1 → 8/16 = 0.5, …).
pub fn fill_choke_points(
    g: &GameState,
    dist_from: &[u8; 81],
    dist_to: &[u8; 81],
    shortest: u8,
    out: &mut [u8; 81],
) {
    out.fill(0);
    if shortest == 255 {
        return;
    }
    let masks = DirMasks::from_ace_game(g);
    let mut delta = [255u8; 81];
    fill_corridor_delta(dist_from, dist_to, shortest, &mut delta);
    for sq in 0..81u8 {
        if dist_to[sq as usize] == 0 || delta[sq as usize] > CHOKE_DELTA_MAX {
            continue;
        }
        let cont = forward_continuations(sq, masks, dist_from, dist_to, &delta);
        out[sq as usize] = encode_choke(cont);
    }
}

/// Convenience: all NNUE geometry fields for one player.
pub fn fill_player_geometry(
    g: &GameState,
    player: usize,
    goal_field: &mut [u8; 81],
    pawn_field: &mut [u8; 81],
    delta_field: &mut [u8; 81],
    choke_field: &mut [u8; 81],
) {
    fill_ace_dist_to_goal(g, player, goal_field);
    fill_ace_dist_from_pawn(g, g.pawn[player], pawn_field);
    let shortest = goal_field[g.pawn[player]];
    fill_corridor_delta(pawn_field, goal_field, shortest, delta_field);
    fill_choke_points(g, pawn_field, goal_field, shortest, choke_field);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::titanium::algebraic_to_move_id;

    fn pos(moves: &[&str]) -> GameState {
        let mut g = GameState::new();
        for m in moves {
            g.make_move(algebraic_to_move_id(m));
        }
        g
    }

    #[test]
    fn ace_flood_matches_public_distance_api() {
        for g in [
            GameState::new(),
            pos(&["e2", "e8", "e3", "e7", "d3h", "f5v"]),
            pos(&["e2", "e8", "e3", "e7", "d4h"]),
        ] {
            for player in [0usize, 1] {
                let mut flood = [0u8; 81];
                let mut public_api = [0u8; 81];
                fill_ace_dist_to_goal(&g, player, &mut flood);
                g.compute_dist(player, &mut public_api);
                assert_eq!(flood, public_api, "goal field player {player}");
                fill_ace_dist_from_pawn(&g, g.pawn[player], &mut flood);
                g.compute_steps_from(g.pawn[player], &mut public_api);
                assert_eq!(flood, public_api, "pawn field player {player}");
            }
        }
    }

    #[test]
    fn shared_masks_match_independent_goal_floods() {
        for g in [
            GameState::new(),
            pos(&["e2", "e8", "e3", "e7", "d3h", "f5v"]),
        ] {
            let masks = DirMasks::from_ace_game(&g);
            for player in [0usize, 1] {
                let mut independent = [0u8; 81];
                let mut shared = [0u8; 81];
                fill_ace_dist_to_goal(&g, player, &mut independent);
                fill_ace_dist_to_goal_with_masks(player, masks, &mut shared);
                assert_eq!(shared, independent, "goal field player {player}");
            }
        }
    }

    #[test]
    fn path_crossing_positive_on_open_corridor() {
        let g = GameState::new();
        let mut goal = [0u8; 81];
        let mut from = [0u8; 81];
        fill_ace_dist_to_goal(&g, 0, &mut goal);
        fill_ace_dist_from_pawn(&g, g.pawn[0], &mut from);
        let s = goal[g.pawn[0]];
        let mut cross = [0u8; 81];
        fill_path_crossing(&g, &from, &goal, s, &mut cross);
        assert!(cross.iter().filter(|&&v| v > 0).count() >= 9);
        assert!(cross[g.pawn[0]] > 0);
        let g2 = pos(&["e2", "e8", "e3", "e7", "d4h"]);
        let mut goal2 = [0u8; 81];
        let mut from2 = [0u8; 81];
        fill_ace_dist_to_goal(&g2, 0, &mut goal2);
        fill_ace_dist_from_pawn(&g2, g2.pawn[0], &mut from2);
        let mut cross2 = [0u8; 81];
        fill_path_crossing(&g2, &from2, &goal2, goal2[g2.pawn[0]], &mut cross2);
        assert!(
            cross2.iter().any(|&v| v > 1),
            "wall should create multi-path hot spots"
        );
    }

    #[test]
    fn sparse_route_masks_partition_short_and_plus_one_cells() {
        let g = GameState::new();
        let mut goal = [0u8; 81];
        let mut from = [0u8; 81];
        fill_ace_dist_to_goal(&g, 0, &mut goal);
        fill_ace_dist_from_pawn(&g, g.pawn[0], &mut from);
        let mut route = [0u8; 81];
        let mut near = [0u8; 81];
        fill_route_masks(&from, &goal, goal[g.pawn[0]], &mut route, &mut near);
        for i in 0..81 {
            assert_eq!(route[i] & near[i], 0);
        }
        assert_eq!(route[g.pawn[0]], 1);
        assert!(route.iter().sum::<u8>() >= 9);
    }

    #[test]
    fn bit_sparse_route_matches_exact_shortest_support() {
        let g = pos(&["e2", "e8", "e3", "e7", "d4h"]);
        for player in [0usize, 1] {
            let mut goal = [0u8; 81];
            let mut from = [0u8; 81];
            fill_ace_dist_to_goal(&g, player, &mut goal);
            fill_ace_dist_from_pawn(&g, g.pawn[player], &mut from);
            let mut expected = [0u8; 81];
            let mut plus_one = [0u8; 81];
            fill_route_masks(
                &from,
                &goal,
                goal[g.pawn[player]],
                &mut expected,
                &mut plus_one,
            );
            let mut route = [0u8; 81];
            let mut flank = [0u8; 81];
            fill_sparse_route_masks(&g, g.pawn[player], &goal, &mut route, &mut flank);
            assert_eq!(route, expected);
            for i in 0..81 {
                assert_eq!(route[i] & flank[i], 0);
            }
        }
    }

    #[test]
    fn dag_untouched_wall_cannot_change_that_players_distance_oracle() {
        use crate::titanium::game::GameState;

        fn lcg(state: &mut u64) -> u64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *state
        }

        let mut rng = 0x51ED_2701_A33Fu64;
        let mut trials = 0usize;
        let mut untouched = 0usize;
        let mut tight = 0usize;

        for _ in 0..600 {
            let mut g = GameState::new();
            let plies = 4 + (lcg(&mut rng) % 30) as usize;
            for _ in 0..plies {
                let mut moves = [0i16; 160];
                let mut n = g.gen_pawn_moves(&mut moves, 0);
                for wt in 0..2usize {
                    for slot in 0..64usize {
                        if g.wall_legal(wt, slot) {
                            moves[n] = (if wt == 0 {
                                crate::titanium::MOVE_HW_BASE
                            } else {
                                crate::titanium::MOVE_VW_BASE
                            }) + slot as i16;
                            n += 1;
                        }
                    }
                }
                if n == 0 {
                    break;
                }
                g.make_move(moves[(lcg(&mut rng) as usize) % n]);
            }

            let mut d = [[0u8; 81]; 2];
            fill_ace_dist_to_goal(&g, 0, &mut d[0]);
            fill_ace_dist_to_goal(&g, 1, &mut d[1]);
            let dag = [
                dag_cells_from_pawn(&g, g.pawn[0], &d[0]),
                dag_cells_from_pawn(&g, g.pawn[1], &d[1]),
            ];
            let pre = [d[0][g.pawn[0]], d[1][g.pawn[1]]];

            for wt in 0..2usize {
                for slot in 0..64usize {
                    if !g.wall_legal(wt, slot) {
                        continue;
                    }
                    let m = (if wt == 0 {
                        crate::titanium::MOVE_HW_BASE
                    } else {
                        crate::titanium::MOVE_VW_BASE
                    }) + slot as i16;
                    trials += 1;
                    let mut g2 = g.clone();
                    g2.make_move(m);
                    for side in 0..2usize {
                        if wall_cuts_player_dag(dag[side], &d[side], m) {
                            tight += 1;
                            continue;
                        }
                        untouched += 1;
                        let mut post = [0u8; 81];
                        fill_ace_dist_to_goal(&g2, side, &mut post);
                        assert_eq!(
                            post[g2.pawn[side]], pre[side],
                            "wall {m} missed side {side}'s DAG but changed its distance                              ({} -> {})",
                            pre[side], post[g2.pawn[side]]
                        );
                    }
                }
            }
        }

        assert!(trials > 5_000, "too few wall trials: {trials}");
        assert!(untouched > 1_000, "classifier never fired: {untouched}");
        eprintln!(
            "dag classifier: {trials} walls, {untouched} DAG-untouched (sound), {tight} tight"
        );
    }

    #[test]
    fn wall_incr_no_edge_cut_implies_unchanged_pawn_distance_oracle() {
        use crate::titanium::game::GameState;

        fn lcg(state: &mut u64) -> u64 {
            *state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            *state
        }

        let mut rng = 0xC47E_0DEDu64;
        let mut wall_trials = 0usize;
        let mut no_edge_trials = 0usize;
        let mut false_negatives = 0usize;

        for _ in 0..800 {
            let mut g = GameState::new();
            let plies = 4 + (lcg(&mut rng) % 28) as usize;
            for _ in 0..plies {
                let mut moves = [0i16; 160];
                let mut n = g.gen_pawn_moves(&mut moves, 0);
                for wt in 0..2usize {
                    for slot in 0..64usize {
                        if g.wall_legal(wt, slot) {
                            moves[n] = (if wt == 0 {
                                crate::titanium::MOVE_HW_BASE
                            } else {
                                crate::titanium::MOVE_VW_BASE
                            }) + slot as i16;
                            n += 1;
                        }
                    }
                }
                if n == 0 {
                    break;
                }
                g.make_move(moves[(lcg(&mut rng) as usize) % n]);
            }

            let mut d0 = [0u8; 81];
            let mut d1 = [0u8; 81];
            fill_ace_dist_to_goal(&g, 0, &mut d0);
            fill_ace_dist_to_goal(&g, 1, &mut d1);
            let pre0 = d0[g.pawn[0]];
            let pre1 = d1[g.pawn[1]];

            for wt in 0..2usize {
                for slot in 0..64usize {
                    if !g.wall_legal(wt, slot) {
                        continue;
                    }
                    let m = (if wt == 0 {
                        crate::titanium::MOVE_HW_BASE
                    } else {
                        crate::titanium::MOVE_VW_BASE
                    }) + slot as i16;
                    wall_trials += 1;
                    let (refresh0, refresh1) = wall_incr_refresh_flags(&d0, &d1, m);
                    if refresh0 || refresh1 {
                        continue;
                    }
                    no_edge_trials += 1;
                    g.make_move(m);
                    let mut child_d0 = [0u8; 81];
                    let mut child_d1 = [0u8; 81];
                    fill_ace_dist_to_goal(&g, 0, &mut child_d0);
                    fill_ace_dist_to_goal(&g, 1, &mut child_d1);
                    let post0 = child_d0[g.pawn[0]];
                    let post1 = child_d1[g.pawn[1]];
                    if post0 != pre0 || post1 != pre1 {
                        false_negatives += 1;
                    }
                    g.unmake_move();
                }
            }
        }

        assert!(
            wall_trials >= 5000,
            "oracle coverage too small: {wall_trials} wall trials"
        );
        assert!(
            no_edge_trials >= 500,
            "oracle no-edge coverage too small: {no_edge_trials}"
        );
        assert_eq!(
            false_negatives, 0,
            "edge-cut skip would misclassify {false_negatives} / {no_edge_trials} no-edge walls"
        );
    }

    #[test]
    fn encode_choke_smooth() {
        assert_eq!(encode_choke(0), 16);
        assert_eq!(encode_choke(1), 8);
        assert_eq!(encode_choke(2), 5); // round(16/3)
        assert_eq!(encode_choke(3), 4);
    }

    #[test]
    fn encode_contested_smooth() {
        assert_eq!(encode_contested(0, 0), 16);
        assert_eq!(encode_contested(0, 1), 8);
        assert_eq!(encode_contested(1, 1), 5); // round(16/3)
        assert_eq!(encode_contested(255, 0), 0);
    }

    #[test]
    fn contested_on_shared_shortest_cells() {
        let g = GameState::new();
        let mut goal0 = [0u8; 81];
        let mut goal1 = [0u8; 81];
        let mut from0 = [0u8; 81];
        let mut from1 = [0u8; 81];
        fill_ace_dist_to_goal(&g, 0, &mut goal0);
        fill_ace_dist_to_goal(&g, 1, &mut goal1);
        fill_ace_dist_from_pawn(&g, g.pawn[0], &mut from0);
        fill_ace_dist_from_pawn(&g, g.pawn[1], &mut from1);
        let mut delta0 = [255u8; 81];
        let mut delta1 = [255u8; 81];
        fill_corridor_delta(&from0, &goal0, goal0[g.pawn[0]], &mut delta0);
        fill_corridor_delta(&from1, &goal1, goal1[g.pawn[1]], &mut delta1);
        let mut contested = [0u8; 81];
        fill_contested(&delta0, &delta1, &mut contested);
        assert!(
            contested.iter().any(|&v| v == 16),
            "startpos has fully contested corridor cells"
        );
    }
}
