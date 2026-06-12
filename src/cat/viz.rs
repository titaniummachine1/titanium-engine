//! CAT v3 visualization snapshot — square heat, wall heat, prune mask for the web UI.

use std::collections::HashSet;

use crate::cat::constants::{
    BOTTLENECK_BONUS_CM, CAT_COLD_CM, CAT_CORRIDOR_CM, CAT_HOT_CM, DIST_PENALTY,
};
use crate::cat::prune::{collect_search_moves, gap_play_zone_mask, wall_completely_skipped};
use crate::core::board::{Board, Move, Player};
use crate::movegen::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::path::BfsScratch;
use crate::util::perft::format_move;

/// JSON payload for `titanium cat` and `/api/titanium/cat`.
pub fn cat_snapshot_json(board: &mut Board) -> String {
    let mut bfs = BfsScratch::new();
    let cat = bfs.build_corridor_attention(board);

    let mut legal = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let legal_n = generate_legal_moves_slice(board, &mut legal, &mut bfs);

    let mut opp_path = [0u8; 81];
    let opp_path_len = crate::cat::prune::get_shortest_path(
        board,
        board.side().opposite(),
        &mut bfs,
        &mut opp_path,
    );
    let opp_dist =
        crate::cat::prune::path_distance(board.side().opposite(), &opp_path, opp_path_len);
    let our_dist_stm = bfs
        .shortest_distance(board, board.side())
        .unwrap_or(DIST_PENALTY);
    let mut search = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let search_n = collect_search_moves(
        board,
        &mut search,
        &mut bfs,
        &cat,
        &opp_path,
        opp_path_len,
        our_dist_stm,
        opp_dist,
        false,
        true,
    );

    let searchable: HashSet<String> = search[..search_n]
        .iter()
        .filter_map(|&mv| match mv {
            Move::Wall { .. } => Some(format_move(mv)),
            _ => None,
        })
        .collect();

    let reachable = bfs.both_reachable_mask(board);
    let gap_zone = gap_play_zone_mask(reachable);

    // Board overlay: per-player max (not summed search heat — that floods mid-game).
    let display_squares = crate::cat::build::build_corridor_display_squares(&mut bfs, board);
    let square_parts: Vec<String> = display_squares.iter().map(|h| h.to_string()).collect();

    let reachable_parts: Vec<&str> = (0u8..81)
        .map(|sq| {
            if reachable & (1u128 << sq) != 0 {
                "1"
            } else {
                "0"
            }
        })
        .collect();

    let mut wall_parts = Vec::new();
    for i in 0..legal_n {
        let mv = legal[i];
        let Move::Wall {
            row,
            col,
            orientation,
        } = mv
        else {
            continue;
        };
        let alg = format_move(mv);
        let heat = cat.wall_edge_heat(row, col, orientation);
        let search = searchable.contains(&alg);
        let skip = wall_completely_skipped(mv, board, reachable, gap_zone);
        wall_parts.push(format!(
            "{{\"alg\":\"{}\",\"heat\":{},\"search\":{},\"skip\":{}}}",
            alg, heat, search, skip
        ));
    }

    let white_dist = bfs
        .shortest_distance(board, Player::One)
        .unwrap_or(DIST_PENALTY);
    let black_dist = bfs
        .shortest_distance(board, Player::Two)
        .unwrap_or(DIST_PENALTY);

    let skipped_squares = reachable_parts.iter().filter(|&&b| b == "0").count();

    format!(
        "{{\"squares\":[{}],\"reachable\":[{}],\"walls\":[{}],\"whiteDist\":{},\"blackDist\":{},\"skippedSquares\":{},\"hotCm\":{},\"coldCm\":{},\"maxCm\":{}}}",
        square_parts.join(","),
        reachable_parts.join(","),
        wall_parts.join(","),
        white_dist,
        black_dist,
        skipped_squares,
        CAT_HOT_CM,
        CAT_COLD_CM,
        // Display squares are per-player max (not the summed search table), so the
        // color ceiling is one player's full corridor + bottleneck bonus.
        CAT_CORRIDOR_CM + BOTTLENECK_BONUS_CM,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::grid::square_index;

    #[test]
    fn snapshot_uses_sparse_corridor_not_full_board_flood() {
        let mut board = Board::new();
        board.apply_algebraic("e2");
        board.apply_algebraic("e8");
        board.apply_algebraic("e3");
        let json = cat_snapshot_json(&mut board);
        let values = parse_snapshot_squares(&json);
        let warm = values.iter().filter(|&&v| v >= CAT_COLD_CM).count();
        assert!(
            warm < 45,
            "corridor CAT should not flood the board, got {warm} warm squares"
        );
        let e3 = values[square_index(2, 4) as usize];
        let a1 = values[square_index(0, 0) as usize];
        assert!(e3 > a1, "pawn corridor hotter than far corner");
        assert!(a1 < CAT_COLD_CM, "far corner stays cold fringe");
    }

    #[test]
    fn snapshot_midgame_corridor_stays_focused() {
        let moves = [
            "e2", "e8", "e3", "e7", "d7v", "e4", "d8v", "f3", "e5", "e6", "b5v",
        ];
        let mut board = Board::new();
        for mv in moves {
            board.apply_algebraic(mv);
        }
        let json = cat_snapshot_json(&mut board);
        let values = parse_snapshot_squares(&json);
        let warm = values.iter().filter(|&&v| v >= CAT_COLD_CM).count();
        assert!(
            warm < 35,
            "mid-game CAT overlay should stay corridor-focused, got {warm} warm squares"
        );
        let e6 = values[square_index(5, 4) as usize];
        let a1 = values[square_index(0, 0) as usize];
        assert!(e6 >= CAT_COLD_CM, "white pawn corridor visible");
        assert!(a1 < CAT_COLD_CM, "far corner cold");
    }

    fn parse_snapshot_squares(json: &str) -> Vec<u16> {
        let edge = "\"squares\":[";
        let start = json.find(edge).unwrap() + edge.len();
        let end = json.find("],\"reachable\"").unwrap();
        json[start..end]
            .split(',')
            .filter_map(|s| s.trim().parse::<u16>().ok())
            .collect()
    }
}
