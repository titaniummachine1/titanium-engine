//! Zero-delay wall-immune corridor detection for the experimental wall-ignorance
//! forced-loss certificate (Titanium v15 experimental).

use crate::pathfinding::bff::flood_to_goal;
use crate::pathfinding::masks::DirMasks;
use crate::titanium::dist::ace_goal_bits_for_player;
use crate::titanium::game::{GameState, BORDER, DELTA, DIRBIT};

/// Undirected board edge in ACE cell indices (canonical: lower index first).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BoardEdge {
    pub a: usize,
    pub b: usize,
}

impl BoardEdge {
    pub const EMPTY: Self = Self { a: 0, b: 0 };

    #[inline]
    pub fn new(a: usize, b: usize) -> Self {
        if a <= b {
            Self { a, b }
        } else {
            Self { a: b, b: a }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StrictRunnerGuarantee {
    pub side: usize,
    pub base_own_moves_to_goal: u8,
    pub max_own_moves_to_goal: u8,
    pub exact_own_moves_to_goal: Option<u8>,
    pub path: [u8; 81],
    pub path_len: u8,
    pub protected_edges: [BoardEdge; 80],
    pub protected_edge_count: u8,
    pub kind: RunnerGuaranteeKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunnerGuaranteeKind {
    ZeroDelayCorridor,
    StrictImmutablePath,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunnerGuarantee {
    pub side: usize,
    /// Maximum future own pawn moves required to reach a goal under the proven strategy.
    pub max_own_moves_to_goal: u8,
    pub path: Vec<usize>,
    pub protected_edges: Vec<BoardEdge>,
    pub kind: RunnerGuaranteeKind,
}

pub struct CorridorScratch;

impl Default for CorridorScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl CorridorScratch {
    pub fn new() -> Self {
        Self
    }
}

#[inline]
pub fn is_goal_cell(side: usize, cell: usize) -> bool {
    (side == 0 && cell < 9) || (side == 1 && cell >= 72)
}

#[inline]
pub fn shortest_distance(g: &GameState, side: usize) -> u8 {
    let mut dist = [255u8; 81];
    g.compute_dist(side, &mut dist);
    dist[g.pawn[side]]
}

fn topology_neighbors(g: &GameState, cell: usize, out: &mut [usize; 4]) -> usize {
    let bm = g.blocked[cell] | BORDER[cell];
    let mut n = 0usize;
    for d in 0..4 {
        if bm & DIRBIT[d] != 0 {
            continue;
        }
        out[n] = (cell as i16 + DELTA[d]) as usize;
        n += 1;
    }
    n
}

/// Reconstruct one deterministic Lee-wave shortest path by descending the
/// already-computed goal-distance field. No queue, parent map, or heap is used.
pub fn reconstruct_one_shortest_path_from_goal_field(
    g: &GameState,
    side: usize,
    goal_dist: &[u8; 81],
    out: &mut [u8; 81],
) -> Option<usize> {
    let mut cell = g.pawn[side];
    let distance = goal_dist[cell];
    if distance == u8::MAX {
        return None;
    }
    out[0] = cell as u8;
    let mut len = 1usize;
    while !is_goal_cell(side, cell) {
        let next_distance = goal_dist[cell].checked_sub(1)?;
        let mut neighbors = [0usize; 4];
        let count = topology_neighbors(g, cell, &mut neighbors);
        let next = neighbors[..count]
            .iter()
            .copied()
            .find(|&candidate| goal_dist[candidate] == next_distance)?;
        out[len] = next as u8;
        len += 1;
        cell = next;
    }
    debug_assert_eq!(len - 1, distance as usize);
    Some(len)
}

/// Fixed-array form of [`walls_that_block_edge`].
pub fn wall_moves_blocking_edge(edge: BoardEdge, out: &mut [i16; 2]) -> usize {
    let a = edge.a;
    let b = edge.b;
    let ar = a / 9;
    let ac = a % 9;
    let br = b / 9;
    let bc = b % 9;
    let mut count = 0usize;

    if ac == bc {
        let north = a.min(b);
        let row = north / 9;
        let col = north % 9;
        if row < 8 && col < 8 {
            out[count] = crate::titanium::MOVE_HW_BASE + (row * 8 + col) as i16;
            count += 1;
        }
        if row < 8 && col > 0 {
            out[count] = crate::titanium::MOVE_HW_BASE + (row * 8 + col - 1) as i16;
            count += 1;
        }
    } else if ar == br {
        let west = a.min(b);
        let row = west / 9;
        let col = west % 9;
        if row < 8 && col < 8 {
            out[count] = crate::titanium::MOVE_VW_BASE + (row * 8 + col) as i16;
            count += 1;
        }
        if row > 0 && col < 8 {
            out[count] = crate::titanium::MOVE_VW_BASE + ((row - 1) * 8 + col) as i16;
            count += 1;
        }
    }
    count
}

#[inline]
pub fn opponent_can_place_before_edge(
    root_side_to_move: usize,
    runner: usize,
    edge_index: usize,
) -> bool {
    root_side_to_move != runner || edge_index > 0
}

#[inline]
fn wall_move_index(mv: i16) -> usize {
    if crate::titanium::is_hwall_move(mv) {
        crate::titanium::wall_slot(mv)
    } else {
        64 + crate::titanium::wall_slot(mv)
    }
}

/// Prove that one concrete shortest path cannot be blocked by any future legal
/// opponent wall before the runner passes the affected edge. Every distinct
/// blocking wall is tested at each real opponent opportunity where it could
/// still affect an untraversed edge.
pub fn prove_strict_immutable_path(g: &GameState, runner: usize) -> Option<StrictRunnerGuarantee> {
    if runner > 1 || g.winner() >= 0 {
        return None;
    }

    let mut goal_dist = [u8::MAX; 81];
    g.compute_dist(runner, &mut goal_dist);
    let mut path = [0u8; 81];
    let path_len = reconstruct_one_shortest_path_from_goal_field(g, runner, &goal_dist, &mut path)?;
    let edge_count = path_len.checked_sub(1)?;
    let mut edges = [BoardEdge::EMPTY; 80];
    let mut first_affected_edge = [u8::MAX; 128];
    let mut last_affected_edge = [u8::MAX; 128];

    for edge_index in 0..edge_count {
        let edge = BoardEdge::new(path[edge_index] as usize, path[edge_index + 1] as usize);
        edges[edge_index] = edge;
        let mut blocking = [0i16; 2];
        let blocking_count = wall_moves_blocking_edge(edge, &mut blocking);
        for &mv in &blocking[..blocking_count] {
            let index = wall_move_index(mv);
            first_affected_edge[index] = first_affected_edge[index].min(edge_index as u8);
            last_affected_edge[index] = if last_affected_edge[index] == u8::MAX {
                edge_index as u8
            } else {
                last_affected_edge[index].max(edge_index as u8)
            };
        }
    }

    let opponent = 1 - runner;
    if g.wl[opponent] > 0 {
        let first_opportunity = usize::from(g.turn == runner);
        for (index, &last_edge) in last_affected_edge.iter().enumerate() {
            if last_edge == u8::MAX || first_opportunity > last_edge as usize {
                continue;
            }
            debug_assert!(first_affected_edge[index] <= last_edge);
            let (wall_type, slot) = if index < 64 {
                (0, index)
            } else {
                (1, index - 64)
            };

            // A wall may be legal at an earlier runner square but illegal at
            // the last one (for example, once that square becomes enclosed).
            // Therefore latest-only testing is not a sound monotonic shortcut.
            // Check every real opponent opportunity until the last path edge
            // this wall can still block. The candidate set itself remains
            // deduplicated, and the search state is never mutated.
            for path_index in first_opportunity..=last_edge as usize {
                debug_assert!(opponent_can_place_before_edge(g.turn, runner, path_index));
                let mut sim = g.clone();
                sim.pawn[runner] = path[path_index] as usize;
                sim.turn = opponent;
                if sim.wall_legal(wall_type, slot) {
                    return None;
                }
            }
        }
    }

    let distance = edge_count as u8;
    Some(StrictRunnerGuarantee {
        side: runner,
        base_own_moves_to_goal: distance,
        max_own_moves_to_goal: distance,
        exact_own_moves_to_goal: None,
        path,
        path_len: path_len as u8,
        protected_edges: edges,
        protected_edge_count: edge_count as u8,
        kind: RunnerGuaranteeKind::StrictImmutablePath,
    })
}

pub fn reconstruct_shortest_goal_path(
    g: &GameState,
    side: usize,
    _scratch: &mut CorridorScratch,
) -> Option<Vec<usize>> {
    let mut goal_dist = [u8::MAX; 81];
    g.compute_dist(side, &mut goal_dist);
    let mut fixed_path = [0u8; 81];
    let len = reconstruct_one_shortest_path_from_goal_field(g, side, &goal_dist, &mut fixed_path)?;
    Some(
        fixed_path[..len]
            .iter()
            .map(|&cell| cell as usize)
            .collect(),
    )
}

pub fn any_goal_reachable_without_edge(
    g: &GameState,
    side: usize,
    start: usize,
    removed: BoardEdge,
    _scratch: &mut CorridorScratch,
) -> bool {
    if side > 1 || start >= 81 {
        return false;
    }
    let masks = DirMasks::from_ace_game(g).without_ace_edge(removed.a, removed.b);
    flood_to_goal(start as u8, masks, ace_goal_bits_for_player(side)).0
}

pub fn detect_zero_delay_corridor(
    g: &GameState,
    side: usize,
    scratch: &mut CorridorScratch,
) -> Option<RunnerGuarantee> {
    let path = reconstruct_shortest_goal_path(g, side, scratch)?;
    let distance = path.len().checked_sub(1)? as u8;

    if distance == 0 {
        return Some(RunnerGuarantee {
            side,
            max_own_moves_to_goal: 0,
            path,
            protected_edges: Vec::new(),
            kind: RunnerGuaranteeKind::ZeroDelayCorridor,
        });
    }

    let start = path[0];
    let mut protected_edges = Vec::with_capacity(path.len() - 1);

    for pair in path.windows(2) {
        let edge = BoardEdge::new(pair[0], pair[1]);
        if any_goal_reachable_without_edge(g, side, start, edge, scratch) {
            return None;
        }
        protected_edges.push(edge);
    }

    debug_assert_eq!(distance, shortest_distance(g, side));
    Some(RunnerGuarantee {
        side,
        max_own_moves_to_goal: distance,
        path,
        protected_edges,
        kind: RunnerGuaranteeKind::ZeroDelayCorridor,
    })
}

/// Debug/test helper: dense wall move ids geometrically capable of blocking `edge`.
pub fn walls_that_block_edge(edge: BoardEdge) -> Vec<i16> {
    let mut fixed = [0i16; 2];
    let count = wall_moves_blocking_edge(edge, &mut fixed);
    fixed[..count].to_vec()
}

/// Test/manually-built position: white column-4 zero-delay corridor, black separated.
pub fn build_column_four_corridor_fixture() -> GameState {
    let mut g = GameState::new();
    g.pawn[0] = 4 * 9 + 4;
    g.pawn[1] = 7 * 9 + 7;
    g.turn = 0;
    g.wl = [10, 10];
    let target_dist = shortest_distance(&g, 0);

    let mut scratch = CorridorScratch::new();
    if fixture_has_cert_margin(&g, &mut scratch) {
        return g;
    }

    let mut candidates: Vec<(usize, usize)> = Vec::new();
    for wtype in 0..2usize {
        for slot in 0..64usize {
            candidates.push((wtype, slot));
        }
    }

    for _pass in 0..32 {
        let mut improved = false;
        for &(wtype, slot) in &candidates {
            if (wtype == 0 && g.hw[slot] != 0) || (wtype == 1 && g.vw[slot] != 0) {
                continue;
            }
            if !g.wall_fits(wtype, slot) {
                continue;
            }
            g.set_wall_bits(wtype, slot, true);
            if wtype == 0 {
                g.hw[slot] = 1;
            } else {
                g.vw[slot] = 1;
            }
            let ok_path = g.has_path(0) && g.has_path(1);
            let same_dist = shortest_distance(&g, 0) == target_dist;
            if !ok_path || !same_dist {
                g.set_wall_bits(wtype, slot, false);
                if wtype == 0 {
                    g.hw[slot] = 0;
                } else {
                    g.vw[slot] = 0;
                }
                continue;
            }
            if fixture_has_cert_margin(&g, &mut scratch) {
                return g;
            }
            improved = true;
        }
        if !improved {
            break;
        }
    }

    for p1 in (9..72).rev() {
        if p1 == g.pawn[0] {
            continue;
        }
        g.pawn[1] = p1;
        if fixture_has_cert_margin(&g, &mut scratch) {
            return g;
        }
    }

    g
}

fn fixture_has_cert_margin(g: &GameState, scratch: &mut CorridorScratch) -> bool {
    use crate::titanium::cert_bridge::paths_overlap;
    use crate::titanium::wall_ignore_cert::earliest_terminal_ply;
    let Some(guarantee) = detect_zero_delay_corridor(g, 0, scratch) else {
        return false;
    };
    let mut d0 = [0u8; 81];
    let mut d1 = [0u8; 81];
    g.compute_dist(0, &mut d0);
    g.compute_dist(1, &mut d1);
    if paths_overlap(g, &d0, &d1) {
        return false;
    }
    let w = guarantee.max_own_moves_to_goal;
    let b = shortest_distance(g, 1);
    if b == 255 {
        return false;
    }
    earliest_terminal_ply(0, 0, w) < earliest_terminal_ply(1, 0, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::titanium::algebraic_to_move_id;

    fn apply_moves(moves: &[&str]) -> GameState {
        let mut g = GameState::new();
        for m in moves {
            g.make_move(algebraic_to_move_id(m));
        }
        g
    }

    #[test]
    fn basic_zero_delay_corridor_detected() {
        let g = build_column_four_corridor_fixture();
        let mut scratch = CorridorScratch::new();
        let guarantee = detect_zero_delay_corridor(&g, 0, &mut scratch).expect("corridor");
        assert!(guarantee.max_own_moves_to_goal > 0);
        assert_eq!(
            guarantee.protected_edges.len(),
            guarantee.max_own_moves_to_goal as usize
        );
        assert_eq!(guarantee.path.first().copied(), Some(g.pawn[0]));
        assert!(is_goal_cell(0, *guarantee.path.last().unwrap()));
    }

    #[test]
    fn wall_anchor_audit_illegal_for_protected_edges() {
        let mut g = build_column_four_corridor_fixture();
        let mut scratch = CorridorScratch::new();
        let guarantee = detect_zero_delay_corridor(&g, 0, &mut scratch).expect("corridor");
        for edge in &guarantee.protected_edges {
            for mv in walls_that_block_edge(*edge) {
                let slot = if crate::titanium::is_hwall_move(mv) {
                    crate::titanium::wall_slot(mv)
                } else {
                    crate::titanium::wall_slot(mv)
                };
                let wtype = if crate::titanium::is_hwall_move(mv) {
                    0
                } else {
                    1
                };
                assert!(
                    !g.wall_legal(wtype, slot),
                    "blocking wall {mv} on edge {:?} must be illegal",
                    edge
                );
            }
        }
    }

    #[test]
    fn unique_shortest_with_longer_bypass_rejected() {
        // Short prefix + chokepoint + goal — bypass via long prefix exists.
        let g = apply_moves(&["e2", "e8", "e3", "e7", "d3h", "f6h", "e4h"]);
        let mut scratch = CorridorScratch::new();
        assert!(
            detect_zero_delay_corridor(&g, 0, &mut scratch).is_none(),
            "longer bypass must prevent zero-delay certificate"
        );
    }

    #[test]
    fn lee_wave_reconstruction_matches_goal_distance() {
        let g = apply_moves(&["e2", "e8", "e3", "e7", "d3h", "f6h", "e4h"]);
        for side in 0..2 {
            let mut dist = [u8::MAX; 81];
            g.compute_dist(side, &mut dist);
            let mut path = [0u8; 81];
            let len = reconstruct_one_shortest_path_from_goal_field(&g, side, &dist, &mut path)
                .expect("reachable goal");
            assert_eq!(len - 1, dist[g.pawn[side]] as usize);
            assert_eq!(path[0] as usize, g.pawn[side]);
            assert!(is_goal_cell(side, path[len - 1] as usize));
            for pair in path[..len].windows(2) {
                let a = pair[0] as usize;
                let b = pair[1] as usize;
                assert_eq!(dist[b] + 1, dist[a]);
                let mut neighbors = [0usize; 4];
                let count = topology_neighbors(&g, a, &mut neighbors);
                assert!(neighbors[..count].contains(&b));
            }
        }
    }

    fn near_goal_runner(turn: usize, runner_distance: usize) -> GameState {
        let mut g = GameState::new();
        g.pawn = [4 + runner_distance * 9, 68];
        g.turn = turn;
        g.wl = [10, 10];
        g
    }

    #[test]
    fn runner_crosses_edge_zero_before_opponent_can_wall() {
        let g = near_goal_runner(0, 1);
        let guarantee = prove_strict_immutable_path(&g, 0).expect("edge zero is too late");
        assert_eq!(guarantee.kind, RunnerGuaranteeKind::StrictImmutablePath);
        assert_eq!(guarantee.max_own_moves_to_goal, 1);
    }

    #[test]
    fn opponent_to_move_can_block_edge_zero() {
        let g = near_goal_runner(1, 1);
        assert!(prove_strict_immutable_path(&g, 0).is_none());
    }

    #[test]
    fn later_legal_wall_rejects_strict_path() {
        let g = near_goal_runner(0, 2);
        assert!(prove_strict_immutable_path(&g, 0).is_none());
    }

    #[test]
    fn detector_never_mutates_real_position_or_inventory() {
        let g = near_goal_runner(0, 1);
        let original = g.clone();
        let _ = prove_strict_immutable_path(&g, 0);
        assert_eq!(g.wl, original.wl);
        assert_eq!(g.hash_lo, original.hash_lo);
        assert_eq!(g.hash_hi, original.hash_hi);
        assert_eq!(g.pawn, original.pawn);
        assert_eq!(g.blocked, original.blocked);
        assert_eq!(g.hw, original.hw);
        assert_eq!(g.vw, original.vw);
        assert_eq!(g.turn, original.turn);
    }

    #[test]
    fn no_opponent_walls_proves_selected_shortest_path() {
        let mut g = near_goal_runner(1, 4);
        g.wl[1] = 0;
        let guarantee = prove_strict_immutable_path(&g, 0).expect("no blocking inventory");
        assert_eq!(guarantee.base_own_moves_to_goal, 4);
        assert_eq!(guarantee.max_own_moves_to_goal, 4);
        assert_eq!(guarantee.protected_edge_count, 4);
    }
}
