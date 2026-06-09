//! CAT-backed move pruning and ordering — does **not** generate moves.
//!
//! Legal moves always come from `moves::generate_legal_moves_slice`. This module
//! filters them using BFS shortest-path data and multi-route corridor heat for
//! both players, then feeds αβ / MCTS with a smaller, tactically relevant set.

use crate::cat::attention::CorridorAttention;
use crate::cat::constants::{CAT_COLD_CM, CAT_HOT_CM, DIST_PENALTY};
use crate::core::board::{Board, Move, Player, WallOrientation};
use crate::movegen::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::opening::book::BookHint;
use crate::path::BfsScratch;
use crate::util::grid::{has_wall, is_goal, square_index, unpack_square, wall_touch_squares};
/// Wasted turn: opponent gets to improve on reply.
pub const TEMPO_PENALTY: i32 = -10;
const WALL_CROSS_GAP_CM: i32 = 40;
const WALL_CROSS_BLOCK_CM: i32 = 35;

pub fn wall_blocks_path_step(mv: Move, sq1: u8, sq2: u8) -> bool {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    let (r1, c1) = unpack_square(sq1);
    let (r2, c2) = unpack_square(sq2);
    match orientation {
        WallOrientation::Horizontal => {
            if c1 == c2 && r1.abs_diff(r2) == 1 {
                let min_r = r1.min(r2);
                min_r == row && (c1 == col || c1 == col + 1)
            } else {
                false
            }
        }
        WallOrientation::Vertical => {
            if r1 == r2 && c1.abs_diff(c2) == 1 {
                let min_c = c1.min(c2);
                min_c == col && (r1 == row || r1 == row + 1)
            } else {
                false
            }
        }
    }
}

pub fn wall_intersects_path(mv: Move, path: &[u8], len: usize) -> bool {
    if len <= 1 {
        return false;
    }
    for i in 0..(len - 1) {
        if wall_blocks_path_step(mv, path[i], path[i + 1]) {
            return true;
        }
    }
    false
}

pub fn get_shortest_path(
    board: &Board,
    player: Player,
    bfs: &mut BfsScratch,
    path_out: &mut [u8; 81],
) -> usize {
    let mut next_out = [u8::MAX; 81];
    bfs.fill_next_toward_goal(board, player, &mut next_out);

    let (pr, pc) = board.pawn(player);
    let mut current = square_index(pr, pc);
    let mut len = 0;
    while current != u8::MAX {
        path_out[len] = current;
        len += 1;
        if len >= 81 {
            break;
        }
        current = next_out[current as usize];
    }
    len
}

pub fn path_distance(player: Player, path: &[u8], len: usize) -> u8 {
    if len == 0 {
        return DIST_PENALTY;
    }
    let last_sq = path[len - 1];
    let (r, _) = unpack_square(last_sq);
    if is_goal(player, r) {
        (len - 1) as u8
    } else {
        DIST_PENALTY
    }
}

/// Net race swing from playing a wall: opponent path lengthening minus our path lengthening.
pub fn wall_net_race(
    board: &mut Board,
    mv: Move,
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
) -> i32 {
    let Move::Wall { .. } = mv else {
        return 0;
    };
    let us = board.side();
    let opp_gain = opp_path_gain(board, mv, opp_dist, bfs);
    let undo = board.make_move(mv);
    let our_after = bfs.shortest_distance(board, us).unwrap_or(DIST_PENALTY);
    board.unmake_move(undo);
    let our_loss = i32::from(our_after.saturating_sub(our_dist));
    opp_gain - our_loss
}

pub fn min_wall_net_race(our_dist: u8, opp_dist: u8) -> i32 {
    if our_dist > opp_dist {
        // Losing the race — any wall that lengthens the opponent counts.
        1
    } else if our_dist == opp_dist {
        // Tied — need a stronger swing to spend a wall.
        2
    } else {
        1
    }
}

pub fn opp_path_gain(board: &mut Board, mv: Move, opp_dist: u8, bfs: &mut BfsScratch) -> i32 {
    let Move::Wall { .. } = mv else {
        return 0;
    };
    let opp = board.side().opposite();
    let undo = board.make_move(mv);
    let new_opp = bfs.shortest_distance(board, opp).unwrap_or(DIST_PENALTY);
    board.unmake_move(undo);
    i32::from(new_opp.saturating_sub(opp_dist))
}

pub fn our_path_gain(board: &mut Board, mv: Move, our_dist: u8, bfs: &mut BfsScratch) -> i32 {
    let Move::Pawn { .. } = mv else {
        return 0;
    };
    let us = board.side();
    let undo = board.make_move(mv);
    let new_our = bfs.shortest_distance(board, us).unwrap_or(DIST_PENALTY);
    board.unmake_move(undo);
    i32::from(our_dist.saturating_sub(new_our))
}

pub fn move_immediate_gain(
    board: &mut Board,
    mv: Move,
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
) -> i32 {
    match mv {
        Move::Pawn { .. } => {
            let g = our_path_gain(board, mv, our_dist, bfs);
            if g > 0 {
                g
            } else {
                TEMPO_PENALTY
            }
        }
        Move::Wall { .. } => {
            let g = opp_path_gain(board, mv, opp_dist, bfs);
            if g > 0 {
                g
            } else {
                TEMPO_PENALTY
            }
        }
    }
}

pub fn is_tactical_move(
    board: &mut Board,
    mv: Move,
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
) -> bool {
    match mv {
        Move::Pawn { .. } => our_path_gain(board, mv, our_dist, bfs) > 0,
        Move::Wall { .. } => opp_path_gain(board, mv, opp_dist, bfs) > 0,
    }
}

#[inline]
fn wall_coord_in_bounds(row: u8, col: u8) -> bool {
    row <= 7 && col <= 7
}

fn is_cross_gap_wall(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    if !wall_coord_in_bounds(row, col) || has_wall(board, row, col, orientation) {
        return false;
    }
    match orientation {
        WallOrientation::Horizontal => {
            row >= 1
                && row <= 6
                && has_wall(board, row - 1, col, WallOrientation::Vertical)
                && has_wall(board, row + 1, col, WallOrientation::Vertical)
        }
        WallOrientation::Vertical => {
            col >= 1
                && col <= 6
                && has_wall(board, row, col - 1, WallOrientation::Horizontal)
                && has_wall(board, row, col + 1, WallOrientation::Horizontal)
        }
    }
}

fn blocks_cross_gap_wall(board: &Board, row: u8, col: u8, orientation: WallOrientation) -> bool {
    if is_cross_gap_wall(board, row, col, orientation) || !wall_coord_in_bounds(row, col) {
        return false;
    }
    match orientation {
        WallOrientation::Horizontal => {
            for dc in [-1i8, 1i8] {
                let gap_col = col as i8 + dc;
                if !(1..=6).contains(&gap_col) {
                    continue;
                }
                let gc = gap_col as u8;
                if row >= 1
                    && row <= 6
                    && has_wall(board, row - 1, gc, WallOrientation::Vertical)
                    && has_wall(board, row + 1, gc, WallOrientation::Vertical)
                {
                    return true;
                }
            }
        }
        WallOrientation::Vertical => {
            for dr in [-1i8, 1i8] {
                let gap_row = row as i8 + dr;
                if !(1..=6).contains(&gap_row) {
                    continue;
                }
                let gr = gap_row as u8;
                if col >= 1
                    && col <= 6
                    && has_wall(board, gr, col - 1, WallOrientation::Horizontal)
                    && has_wall(board, gr, col + 1, WallOrientation::Horizontal)
                {
                    return true;
                }
            }
        }
    }
    false
}

fn wall_shape_local_heat(
    cat: &CorridorAttention,
    row: u8,
    col: u8,
    orientation: WallOrientation,
) -> u16 {
    let edge = cat.wall_edge_heat(row, col, orientation);
    let touch = wall_touch_squares(row, col, orientation)
        .iter()
        .map(|&(r, c)| cat.square_heat(r, c))
        .max()
        .unwrap_or(0);
    edge.max(touch)
}

pub fn wall_shape_attention_bonus(board: &Board, mv: Move, cat: &CorridorAttention) -> i32 {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return 0;
    };
    if wall_shape_local_heat(cat, row, col, orientation) < CAT_HOT_CM {
        return 0;
    }
    if is_cross_gap_wall(board, row, col, orientation) {
        WALL_CROSS_GAP_CM
    } else if blocks_cross_gap_wall(board, row, col, orientation) {
        WALL_CROSS_BLOCK_CM
    } else {
        0
    }
}

/// Live squares orthogonally adjacent to sealed-off (unreachable) territory.
pub fn corridor_mouth_mask(reachable: u128) -> u128 {
    let mut mouths = 0u128;
    for sq in 0u8..81 {
        if reachable & (1u128 << sq) == 0 {
            continue;
        }
        let (r, c) = unpack_square(sq);
        for (dr, dc) in [(-1i8, 0), (1, 0), (0, -1), (0, 1)] {
            let nr = r as i16 + dr as i16;
            let nc = c as i16 + dc as i16;
            if !(0..=9).contains(&nr) || !(0..=9).contains(&nc) {
                continue;
            }
            let neighbor = square_index(nr as u8, nc as u8);
            if reachable & (1u128 << neighbor) == 0 {
                mouths |= 1u128 << sq;
                break;
            }
        }
    }
    mouths
}

/// Mouth squares, their reachable ring, and adjacent sealed cells (the gap slot itself).
pub fn gap_play_zone_mask(reachable: u128) -> u128 {
    let mouths = corridor_mouth_mask(reachable);
    let mut zone = mouths;
    for sq in 0u8..81 {
        if mouths & (1u128 << sq) == 0 {
            continue;
        }
        let (r, c) = unpack_square(sq);
        for (dr, dc) in [(-1i8, 0), (1, 0), (0, -1), (0, 1)] {
            let nr = r as i16 + dr as i16;
            let nc = c as i16 + dc as i16;
            if !(0..=9).contains(&nr) || !(0..=9).contains(&nc) {
                continue;
            }
            // Include both live ring and the sealed gap cell — half-walls and cross-gap H/V land here.
            zone |= 1u128 << square_index(nr as u8, nc as u8);
        }
    }
    zone
}

/// Touches sealed (unreachable) territory that is not part of the gap mouth play zone.
fn wall_probes_sealed_interior(mv: Move, reachable: u128, gap_zone: u128) -> bool {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    for (r, c) in wall_touch_squares(row, col, orientation) {
        let sq = square_index(r, c);
        if reachable & (1u128 << sq) == 0 && gap_zone & (1u128 << sq) == 0 {
            return true;
        }
    }
    false
}

fn wall_touches_gap_zone(mv: Move, gap_zone: u128) -> bool {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    for (r, c) in wall_touch_squares(row, col, orientation) {
        if gap_zone & (1u128 << square_index(r, c)) != 0 {
            return true;
        }
    }
    false
}

/// Wall touches a reachable square on a live corridor (not sealed void).
fn wall_touches_live_corridor(mv: Move, cat: &CorridorAttention, reachable: u128) -> bool {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    if cat.wall_edge_heat(row, col, orientation) >= CAT_COLD_CM {
        return true;
    }
    for (r, c) in wall_touch_squares(row, col, orientation) {
        if reachable & (1u128 << square_index(r, c)) == 0 {
            continue;
        }
        if cat.square_heat(r, c) >= CAT_COLD_CM {
            return true;
        }
    }
    false
}

fn wall_in_dead_zone(mv: Move, reachable: u128) -> bool {
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    for (r, c) in wall_touch_squares(row, col, orientation) {
        if reachable & (1u128 << square_index(r, c)) != 0 {
            return false;
        }
    }
    true
}

/// Whether a wall can affect either player's reasonable routes to goal.
pub fn wall_should_search(
    mv: Move,
    cat: &CorridorAttention,
    reachable: u128,
    gap_zone: u128,
    board: &mut Board,
    our_dist: u8,
    opp_dist: u8,
    opp_path: &[u8],
    opp_path_len: usize,
    bfs: &mut BfsScratch,
) -> bool {
    if wall_in_dead_zone(mv, reachable) {
        return false;
    }
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    // Gap geometry: H through V|V (or flank block beside it) can seal/open the pocket.
    // If the wall is not in a dead zone it touches live territory — always search it.
    if is_cross_gap_wall(board, row, col, orientation)
        || blocks_cross_gap_wall(board, row, col, orientation)
    {
        return true;
    }
    // Any wall touching the playable/sealed mouth, gap slot, or immediate ring.
    if gap_zone != 0 && wall_touches_gap_zone(mv, gap_zone) {
        return true;
    }
    // Wall reaches into sealed void away from the gap — no tactical value.
    if gap_zone != 0 && wall_probes_sealed_interior(mv, reachable, gap_zone) {
        return false;
    }
    let on_opp_path = wall_intersects_path(mv, opp_path, opp_path_len);
    if on_opp_path {
        return true;
    }
    if opp_path_gain(board, mv, opp_dist, bfs) > 0 {
        return true;
    }
    // Cold / passive walls stay in the move list — search applies LMR from CAT heat & net race.
    let net = wall_net_race(board, mv, our_dist, opp_dist, bfs);
    let min_net = min_wall_net_race(our_dist, opp_dist);
    let behind = our_dist >= opp_dist;
    if behind {
        if net < min_net {
            return false;
        }
    } else if net < min_net {
        return false;
    }

    true
}

/// Hard skip — dead void or sealed interior away from gap; never searched (not LMR).
pub fn wall_completely_skipped(mv: Move, board: &Board, reachable: u128, gap_zone: u128) -> bool {
    if wall_in_dead_zone(mv, reachable) {
        return true;
    }
    let Move::Wall {
        row,
        col,
        orientation,
    } = mv
    else {
        return false;
    };
    if is_cross_gap_wall(board, row, col, orientation)
        || blocks_cross_gap_wall(board, row, col, orientation)
    {
        return false;
    }
    if gap_zone != 0 && wall_touches_gap_zone(mv, gap_zone) {
        return false;
    }
    gap_zone != 0 && wall_probes_sealed_interior(mv, reachable, gap_zone)
}

/// Filter legal moves for search — never generates moves, only prunes `moves` output.
pub fn collect_search_moves(
    board: &mut Board,
    buf: &mut [Move],
    bfs: &mut BfsScratch,
    tactical_only: bool,
    allow_walls: bool,
) -> usize {
    let mut scratch = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let full = generate_legal_moves_slice(board, &mut scratch, bfs);
    if full == 0 {
        return 0;
    }

    let mut opp_dist = DIST_PENALTY;
    let mut opp_path = [0u8; 81];
    let mut opp_path_len = 0usize;
    let cat = if allow_walls {
        let opp = board.side().opposite();
        opp_dist = bfs.shortest_distance(board, opp).unwrap_or(DIST_PENALTY);
        opp_path_len = get_shortest_path(board, opp, bfs, &mut opp_path);
        bfs.build_corridor_attention(board)
    } else {
        CorridorAttention::default()
    };
    let our_dist = bfs
        .shortest_distance(board, board.side())
        .unwrap_or(DIST_PENALTY);
    let reachable = bfs.both_reachable_mask(board);
    let gap_zone = if allow_walls {
        gap_play_zone_mask(reachable)
    } else {
        0
    };

    let mut n = 0usize;

    for i in 0..full {
        let mv = scratch[i];
        match mv {
            Move::Pawn { .. } => {
                if tactical_only && our_path_gain(board, mv, our_dist, bfs) <= 0 {
                    continue;
                }
                buf[n] = mv;
                n += 1;
            }
            Move::Wall { .. } => {
                if !allow_walls {
                    continue;
                }
                if !wall_should_search(
                    mv,
                    &cat,
                    reachable,
                    gap_zone,
                    board,
                    our_dist,
                    opp_dist,
                    &opp_path,
                    opp_path_len,
                    bfs,
                ) {
                    continue;
                }
                buf[n] = mv;
                n += 1;
            }
        }
    }

    if n == 0 && !tactical_only {
        buf[..full].copy_from_slice(&scratch[..full]);
        return full;
    }
    if n == 0 && tactical_only {
        for i in 0..full {
            if matches!(scratch[i], Move::Pawn { .. }) {
                buf[n] = scratch[i];
                n += 1;
            }
        }
    }
    n
}

fn cat_score_for_move(mv: Move, cat: &CorridorAttention) -> i32 {
    match mv {
        Move::Pawn { row, col } => i32::from(cat.square_heat(row, col)),
        Move::Wall {
            row,
            col,
            orientation,
        } => i32::from(cat.wall_edge_heat(row, col, orientation)),
    }
}

/// Combined corridor heat for LMR / futility (higher = more likely to matter).
pub fn move_corridor_attention(board: &Board, mv: Move, cat: &CorridorAttention) -> i32 {
    cat_score_for_move(mv, cat) + wall_shape_attention_bonus(board, mv, cat)
}

pub fn move_order_score(
    board: &mut Board,
    mv: Move,
    tt_best: Option<Move>,
    book_hint: Option<BookHint>,
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
    cat: &CorridorAttention,
) -> i32 {
    if tt_best == Some(mv) {
        return 10_000;
    }
    if let Some(hint) = book_hint {
        if hint.mv == mv {
            // PV bias only — tactical race gains and TT still outrank theory.
            let bias = i32::from(hint.stm_bias) / 4;
            return 9_000 + i32::from(hint.priority) + bias;
        }
    }
    let gain = move_immediate_gain(board, mv, our_dist, opp_dist, bfs);
    let behind = our_dist > opp_dist;
    let race_pressure = behind || opp_dist <= 4;

    if matches!(mv, Move::Wall { .. }) {
        let net = wall_net_race(board, mv, our_dist, opp_dist, bfs);
        let min_net = min_wall_net_race(our_dist, opp_dist);
        if net < min_net {
            return -20_000 + move_corridor_attention(board, mv, cat);
        }
        if race_pressure {
            return 15_000 + net * 120 + move_corridor_attention(board, mv, cat) / 8;
        }
        if net >= min_net {
            return 12_000 + net * 80;
        }
    }

    if our_dist >= opp_dist {
        if matches!(mv, Move::Pawn { .. }) && gain > 0 {
            // Lateral / slow sprint while clearly losing the race is not a tactic.
            let closes_gap = gain >= 2 || our_dist.saturating_sub(1) <= opp_dist;
            if behind && !closes_gap {
                return 800 + gain * 40;
            }
            return 14_000 + gain * 100;
        }
    }
    if gain > 0 {
        1000 + gain * 100
    } else {
        move_corridor_attention(board, mv, cat) + TEMPO_PENALTY
    }
}

pub fn order_moves(
    board: &mut Board,
    moves: &mut [Move],
    n: usize,
    tt_best: Option<Move>,
    book_hint: Option<BookHint>,
    scores: &mut [i32; MAX_LEGAL_MOVES],
    our_dist: u8,
    opp_dist: u8,
    bfs: &mut BfsScratch,
    cat: &CorridorAttention,
) {
    for i in 0..n {
        scores[i] = move_order_score(
            board, moves[i], tt_best, book_hint, our_dist, opp_dist, bfs, cat,
        );
    }
    let mut order: [usize; MAX_LEGAL_MOVES] = core::array::from_fn(|i| i);
    order[..n].sort_unstable_by(|&a, &b| scores[b].cmp(&scores[a]));
    let mut tmp = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    tmp[..n].copy_from_slice(&moves[..n]);
    for i in 0..n {
        moves[i] = tmp[order[i]];
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::{Board, Player};
    use crate::util::grid::{set_wall, wall_touch_squares};

    #[test]
    fn blocks_cross_gap_detects_shifted_prevention() {
        let mut board = Board::new();
        set_wall(&mut board, 2, 4, WallOrientation::Vertical, true);
        set_wall(&mut board, 4, 4, WallOrientation::Vertical, true);
        assert!(blocks_cross_gap_wall(
            &board,
            3,
            3,
            WallOrientation::Horizontal
        ));
        assert!(blocks_cross_gap_wall(
            &board,
            3,
            5,
            WallOrientation::Horizontal
        ));
        assert!(!blocks_cross_gap_wall(
            &board,
            3,
            4,
            WallOrientation::Horizontal
        ));
    }

    #[test]
    fn wall_shape_bonus_only_for_hot_cross_gap() {
        let mut board = Board::new();
        set_wall(&mut board, 2, 4, WallOrientation::Vertical, true);
        set_wall(&mut board, 4, 4, WallOrientation::Vertical, true);
        let cold_cat = CorridorAttention::default();
        let cross = Move::Wall {
            row: 3,
            col: 4,
            orientation: WallOrientation::Horizontal,
        };
        assert_eq!(
            wall_shape_attention_bonus(&board, cross, &cold_cat),
            0,
            "cold CAT should not revive unrelated shape bonus"
        );

        let mut bfs = BfsScratch::new();
        let hot_cat = bfs.build_corridor_attention(&board);
        assert!(
            wall_shape_attention_bonus(&board, cross, &hot_cat) >= WALL_CROSS_GAP_CM,
            "hot corridor cross-gap should get a tiny ordering nudge"
        );
    }

    #[test]
    fn cross_gap_wall_detects_perpendicular_through_gap() {
        let mut board = Board::new();
        set_wall(&mut board, 2, 4, WallOrientation::Vertical, true);
        set_wall(&mut board, 4, 4, WallOrientation::Vertical, true);
        assert!(is_cross_gap_wall(&board, 3, 4, WallOrientation::Horizontal));
        assert!(!is_cross_gap_wall(&board, 3, 4, WallOrientation::Vertical));
    }

    #[test]
    fn cross_gap_ignores_adjacent_chain_t_junction() {
        let mut board = Board::new();
        set_wall(&mut board, 2, 4, WallOrientation::Vertical, true);
        set_wall(&mut board, 3, 4, WallOrientation::Vertical, true);
        assert!(!is_cross_gap_wall(
            &board,
            3,
            4,
            WallOrientation::Horizontal
        ));
    }

    #[test]
    fn left_chain_keeps_gap_tactics() {
        let mut board = Board::new();
        // Vertical chain on c|d (col 2) with deliberate gaps between segments.
        for row in [0u8, 2, 4, 6] {
            set_wall(&mut board, row, 2, WallOrientation::Vertical, true);
        }
        board.pawns = [(3, 0), (5, 0)]; // a4, a6 — left pocket
        board.hash = crate::core::zobrist::hash_board(&board);

        let mut bfs = BfsScratch::new();
        let cat = bfs.build_corridor_attention(&board);
        let reachable = bfs.both_reachable_mask(&board);
        let our_dist = bfs
            .shortest_distance(&board, Player::One)
            .unwrap_or(DIST_PENALTY);
        let opp_dist = bfs
            .shortest_distance(&board, Player::Two)
            .unwrap_or(DIST_PENALTY);
        let mut opp_path = [0u8; 81];
        let opp_path_len = get_shortest_path(&board, Player::Two, &mut bfs, &mut opp_path);
        let gap_zone = gap_play_zone_mask(reachable);

        let cross_gap = Move::Wall {
            row: 3,
            col: 2,
            orientation: WallOrientation::Horizontal,
        };
        let mut cross_board = board.clone();
        assert!(
            is_cross_gap_wall(&cross_board, 3, 2, WallOrientation::Horizontal),
            "H through V|gap|V should be detected"
        );
        assert!(
            wall_should_search(
                cross_gap,
                &cat,
                reachable,
                gap_zone,
                &mut cross_board,
                our_dist,
                opp_dist,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "horizontal through chain gap must stay searchable"
        );

        let flank_block = Move::Wall {
            row: 3,
            col: 3,
            orientation: WallOrientation::Horizontal,
        };
        let mut flank_board = board.clone();
        assert!(
            blocks_cross_gap_wall(&flank_board, 3, 3, WallOrientation::Horizontal),
            "shifted block beside gap should be detected"
        );
        assert!(
            wall_should_search(
                flank_block,
                &cat,
                reachable,
                gap_zone,
                &mut flank_board,
                our_dist,
                opp_dist,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "flank block preventing half-protrusion into void must stay searchable"
        );
    }

    #[test]
    fn gap_mouth_keeps_t_junction_tactics_prunes_deep_void() {
        let mut board = Board::new();
        // Three walls around a T mouth; fourth slot open at (3,4) horizontal.
        set_wall(&mut board, 2, 4, WallOrientation::Vertical, true);
        set_wall(&mut board, 3, 3, WallOrientation::Vertical, true);
        set_wall(&mut board, 3, 5, WallOrientation::Vertical, true);
        board.pawns = [(4, 4), (6, 4)];
        board.hash = crate::core::zobrist::hash_board(&board);

        let mut bfs = BfsScratch::new();
        let cat = bfs.build_corridor_attention(&board);
        let reachable = bfs.both_reachable_mask(&board);
        let gap_zone = gap_play_zone_mask(reachable);
        assert!(
            gap_zone != 0,
            "T mouth should produce a non-empty gap play zone"
        );
        let our_dist = bfs
            .shortest_distance(&board, Player::One)
            .unwrap_or(DIST_PENALTY);
        let opp_dist = bfs
            .shortest_distance(&board, Player::Two)
            .unwrap_or(DIST_PENALTY);
        let mut opp_path = [0u8; 81];
        let opp_path_len = get_shortest_path(&board, Player::Two, &mut bfs, &mut opp_path);

        let mouth_wall = Move::Wall {
            row: 3,
            col: 4,
            orientation: WallOrientation::Horizontal,
        };
        let mut mouth_board = board.clone();
        assert!(
            wall_should_search(
                mouth_wall,
                &cat,
                reachable,
                gap_zone,
                &mut mouth_board,
                our_dist,
                opp_dist,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "wall at T-junction gap mouth must stay searchable"
        );

        // Fully sealed pocket on the far left — interior wall does not touch gap mouth.
        let mut pocket = Board::new();
        for &(row, col, orient) in &[
            (1, 0, WallOrientation::Vertical),
            (1, 1, WallOrientation::Horizontal),
            (2, 0, WallOrientation::Horizontal),
        ] {
            set_wall(&mut pocket, row, col, orient, true);
        }
        let reachable_pocket = bfs.both_reachable_mask(&pocket);
        let gap_zone_pocket = gap_play_zone_mask(reachable_pocket);
        let inner = Move::Wall {
            row: 0,
            col: 0,
            orientation: WallOrientation::Vertical,
        };
        let mut inner_board = pocket.clone();
        assert!(
            !wall_should_search(
                inner,
                &CorridorAttention::default(),
                reachable_pocket,
                gap_zone_pocket,
                &mut inner_board,
                DIST_PENALTY,
                DIST_PENALTY,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "interior sealed void wall away from gap mouth must be pruned"
        );
    }

    #[test]
    fn dead_zone_prunes_walls_in_unreachable_void() {
        let board = Board::new();
        let mut bfs = BfsScratch::new();
        let cat = CorridorAttention::default();
        let reachable = bfs.both_reachable_mask(&board);
        let mut opp_path = [0u8; 81];
        let opp_path_len = get_shortest_path(&board, Player::Two, &mut bfs, &mut opp_path);

        // Fully buried inner wall — every touched square is outside both floods.
        let mut pocket = Board::new();
        for &(row, col, orient) in &[
            (1, 0, WallOrientation::Vertical),
            (1, 1, WallOrientation::Horizontal),
            (2, 0, WallOrientation::Horizontal),
        ] {
            set_wall(&mut pocket, row, col, orient, true);
        }
        let inner_t = Move::Wall {
            row: 0,
            col: 0,
            orientation: WallOrientation::Vertical,
        };
        let reachable_pocket = bfs.both_reachable_mask(&pocket);
        let gap_zone_pocket = gap_play_zone_mask(reachable_pocket);
        let mut pocket_board = pocket.clone();
        assert!(
            !wall_should_search(
                inner_t,
                &cat,
                reachable_pocket,
                gap_zone_pocket,
                &mut pocket_board,
                DIST_PENALTY,
                DIST_PENALTY,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "walls in a sealed void cannot affect play on the live side of the chain"
        );
    }

    #[test]
    fn wall_search_prunes_enclosed_t_keeps_corridor_blocks() {
        let board = Board::new();
        let mut bfs = BfsScratch::new();
        let cat = bfs.build_corridor_attention(&board);
        let reachable = bfs.both_reachable_mask(&board);
        let our_dist = bfs
            .shortest_distance(&board, Player::One)
            .unwrap_or(DIST_PENALTY);
        let opp_dist = bfs
            .shortest_distance(&board, Player::Two)
            .unwrap_or(DIST_PENALTY);
        let mut opp_path = [0u8; 81];
        let opp_path_len = get_shortest_path(&board, Player::Two, &mut bfs, &mut opp_path);
        let gap_zone = gap_play_zone_mask(reachable);

        let passive_corner = Move::Wall {
            row: 0,
            col: 0,
            orientation: WallOrientation::Horizontal,
        };
        let mut passive_board = board.clone();
        assert!(
            !wall_should_search(
                passive_corner,
                &cat,
                reachable,
                gap_zone,
                &mut passive_board,
                our_dist,
                opp_dist,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "passive corner T-wall should be pruned"
        );

        let corridor_wall = Move::Wall {
            row: 3,
            col: 4,
            orientation: WallOrientation::Horizontal,
        };
        let mut corridor_board = board.clone();
        assert!(
            wall_should_search(
                corridor_wall,
                &cat,
                reachable,
                gap_zone,
                &mut corridor_board,
                our_dist,
                opp_dist,
                &opp_path,
                opp_path_len,
                &mut bfs,
            ),
            "central corridor wall should stay searchable"
        );

        let mut pocket = Board::new();
        for &(row, col, orient) in &[
            (1, 0, WallOrientation::Vertical),
            (1, 1, WallOrientation::Horizontal),
            (2, 0, WallOrientation::Horizontal),
        ] {
            set_wall(&mut pocket, row, col, orient, true);
        }
        let cat_pocket = bfs.build_corridor_attention(&pocket);
        let reachable_pocket = bfs.both_reachable_mask(&pocket);
        let gap_zone_pocket = gap_play_zone_mask(reachable_pocket);
        let our_dist_pocket = bfs
            .shortest_distance(&pocket, Player::One)
            .unwrap_or(DIST_PENALTY);
        let opp_dist_pocket = bfs
            .shortest_distance(&pocket, Player::Two)
            .unwrap_or(DIST_PENALTY);
        let mut opp_path_pocket = [0u8; 81];
        let opp_path_len_pocket =
            get_shortest_path(&pocket, Player::Two, &mut bfs, &mut opp_path_pocket);

        let inner_t = Move::Wall {
            row: 0,
            col: 0,
            orientation: WallOrientation::Vertical,
        };
        let mut inner_board = pocket.clone();
        assert!(
            !wall_should_search(
                inner_t,
                &cat_pocket,
                reachable_pocket,
                gap_zone_pocket,
                &mut inner_board,
                our_dist_pocket,
                opp_dist_pocket,
                &opp_path_pocket,
                opp_path_len_pocket,
                &mut bfs,
            ),
            "fully buried inner T-wall should be pruned"
        );
    }

    #[test]
    fn useless_t_junction_gets_no_shape_bonus() {
        let mut board = Board::new();
        set_wall(&mut board, 0, 5, WallOrientation::Vertical, true);
        set_wall(&mut board, 1, 5, WallOrientation::Vertical, true);
        let mut bfs = BfsScratch::new();
        let cat = bfs.build_corridor_attention(&board);
        let t_junction = Move::Wall {
            row: 1,
            col: 5,
            orientation: WallOrientation::Horizontal,
        };
        assert_eq!(
            wall_shape_attention_bonus(&board, t_junction, &cat),
            0,
            "far-side T junction should not get shape attention"
        );
    }

    #[test]
    fn sprint_line_orders_wall_before_lateral_pawn() {
        use crate::core::board::Board;
        use crate::util::perft::format_move;

        let seq = ["e2", "e8", "d2", "e7", "d3", "e6", "d4", "e5", "c4", "e4"];
        let mut board = Board::new();
        for m in seq {
            board.apply_algebraic(m);
        }
        let mut bfs = BfsScratch::new();
        let our = bfs
            .shortest_distance(&board, Player::One)
            .unwrap_or(DIST_PENALTY);
        let opp = bfs
            .shortest_distance(&board, Player::Two)
            .unwrap_or(DIST_PENALTY);
        let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n = collect_search_moves(&mut board, &mut buf, &mut bfs, false, true);
        let cat = bfs.build_corridor_attention(&board);
        let mut scores = [0i32; MAX_LEGAL_MOVES];
        order_moves(
            &mut board,
            &mut buf,
            n,
            None,
            None,
            &mut scores,
            our,
            opp,
            &mut bfs,
            &cat,
        );
        let mut ranked: Vec<_> = (0..n).map(|i| (scores[i], format_move(buf[i]))).collect();
        ranked.sort_by(|a, b| b.0.cmp(&a.0));
        for (s, m) in ranked.iter().take(8) {
            eprintln!("  {s} {m}");
        }
        let top_has_wall = ranked
            .iter()
            .take(6)
            .any(|(_, m)| m.ends_with('h') || m.ends_with('v'));
        assert!(
            top_has_wall,
            "a blocking wall should rank in top 6, top={ranked:?}"
        );
    }

    #[test]
    fn sprint_line_includes_blocker_wall_when_behind() {
        use crate::core::board::Board;
        use crate::util::perft::format_move;

        let seq = ["e2", "e8", "d2", "e7", "d3", "e6", "d4", "e5", "c4", "e4"];
        let mut board = Board::new();
        for m in seq {
            board.apply_algebraic(m);
        }
        let mut bfs = BfsScratch::new();
        let our = bfs
            .shortest_distance(&board, Player::One)
            .unwrap_or(DIST_PENALTY);
        let opp = bfs
            .shortest_distance(&board, Player::Two)
            .unwrap_or(DIST_PENALTY);
        assert!(our > opp, "white should be behind in race, W{our} B{opp}");

        let mut buf = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
        let n = collect_search_moves(&mut board, &mut buf, &mut bfs, false, true);
        let walls: Vec<String> = buf[..n]
            .iter()
            .filter(|mv| matches!(mv, Move::Wall { .. }))
            .map(|&mv| format_move(mv))
            .collect();
        eprintln!("searchable walls ({n} total moves): {walls:?}");
        assert!(
            !walls.is_empty(),
            "must keep at least one blocking wall when losing the sprint"
        );
    }
}
