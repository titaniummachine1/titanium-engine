//! Zero-delay wall-immune corridor detection for the experimental wall-ignorance
//! forced-loss certificate (Titanium v15 experimental).

use crate::titanium::game::{GameState, BORDER, DELTA, DIRBIT};
use std::collections::VecDeque;

/// Undirected board edge in ACE cell indices (canonical: lower index first).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct BoardEdge {
    pub a: usize,
    pub b: usize,
}

impl BoardEdge {
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
pub enum RunnerGuaranteeKind {
    ZeroDelayCorridor,
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

pub struct CorridorScratch {
    queue: VecDeque<usize>,
    visited: [bool; 81],
    parent: [Option<usize>; 81],
}

impl Default for CorridorScratch {
    fn default() -> Self {
        Self::new()
    }
}

impl CorridorScratch {
    pub fn new() -> Self {
        Self {
            queue: VecDeque::with_capacity(81),
            visited: [false; 81],
            parent: [None; 81],
        }
    }

    fn reset_path_search(&mut self) {
        self.queue.clear();
        self.visited = [false; 81];
        self.parent = [None; 81];
    }

    fn reset_reachability(&mut self) {
        self.queue.clear();
        self.visited = [false; 81];
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

pub fn reconstruct_shortest_goal_path(
    g: &GameState,
    side: usize,
    scratch: &mut CorridorScratch,
) -> Option<Vec<usize>> {
    scratch.reset_path_search();
    let start = g.pawn[side];
    scratch.visited[start] = true;
    scratch.parent[start] = None;
    scratch.queue.push_back(start);

    let mut found_goal = None;
    while let Some(current) = scratch.queue.pop_front() {
        if is_goal_cell(side, current) {
            found_goal = Some(current);
            break;
        }
        let mut nb = [0usize; 4];
        let nn = topology_neighbors(g, current, &mut nb);
        for i in 0..nn {
            let next = nb[i];
            if scratch.visited[next] {
                continue;
            }
            scratch.visited[next] = true;
            scratch.parent[next] = Some(current);
            scratch.queue.push_back(next);
        }
    }

    let goal = found_goal?;
    let mut reversed = vec![goal];
    let mut cursor = goal;
    while cursor != start {
        cursor = scratch.parent[cursor]?;
        reversed.push(cursor);
    }
    reversed.reverse();
    debug_assert_eq!(
        reversed.len().checked_sub(1),
        Some(shortest_distance(g, side) as usize)
    );
    Some(reversed)
}

pub fn any_goal_reachable_without_edge(
    g: &GameState,
    side: usize,
    start: usize,
    removed: BoardEdge,
    scratch: &mut CorridorScratch,
) -> bool {
    scratch.reset_reachability();
    scratch.visited[start] = true;
    scratch.queue.push_back(start);

    while let Some(current) = scratch.queue.pop_front() {
        if is_goal_cell(side, current) {
            return true;
        }
        let mut nb = [0usize; 4];
        let nn = topology_neighbors(g, current, &mut nb);
        for i in 0..nn {
            let next = nb[i];
            if BoardEdge::new(current, next) == removed {
                continue;
            }
            if scratch.visited[next] {
                continue;
            }
            scratch.visited[next] = true;
            scratch.queue.push_back(next);
        }
    }
    false
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

/// Debug/test helper: wall move ids (100+ / 200+) geometrically capable of blocking `edge`.
pub fn walls_that_block_edge(edge: BoardEdge) -> Vec<i16> {
    let a = edge.a;
    let b = edge.b;
    let mut out = Vec::with_capacity(2);
    let ar = a / 9;
    let ac = a % 9;
    let br = b / 9;
    let bc = b % 9;

    if ac == bc {
        // N-S adjacency
        let north = a.min(b);
        let row = north / 9;
        let col = north % 9;
        if row < 8 && col < 8 {
            out.push(100 + (row * 8 + col) as i16);
        }
        if row < 8 && col > 0 {
            out.push(100 + (row * 8 + (col - 1)) as i16);
        }
    } else if ar == br {
        // E-W adjacency
        let west = a.min(b);
        let row = west / 9;
        let col = west % 9;
        if row < 8 && col < 8 {
            out.push(200 + (row * 8 + col) as i16);
        }
        if row > 0 && col < 8 {
            out.push(200 + ((row - 1) * 8 + col) as i16);
        }
    }
    out
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
                let slot = if mv < 200 {
                    (mv - 100) as usize
                } else {
                    (mv - 200) as usize
                };
                let wtype = if mv < 200 { 0 } else { 1 };
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
}
