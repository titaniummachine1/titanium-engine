//! CAT v3 visualization snapshot — square heat, wall heat, prune mask for the web UI.

use std::collections::HashSet;

use crate::cat::constants::{CAT_COLD_CM, CAT_HOT_CM, DIST_PENALTY};
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

    let mut search = [Move::Pawn { row: 0, col: 0 }; MAX_LEGAL_MOVES];
    let search_n = collect_search_moves(board, &mut search, &mut bfs, false, true);

    let searchable: HashSet<String> = search[..search_n]
        .iter()
        .filter_map(|&mv| match mv {
            Move::Wall { .. } => Some(format_move(mv)),
            _ => None,
        })
        .collect();

    let reachable = bfs.both_reachable_mask(board);
    let gap_zone = gap_play_zone_mask(reachable);

    let square_parts: Vec<String> = (0u8..9)
        .flat_map(|row| (0u8..9).map(move |col| cat.square_heat(row, col).to_string()))
        .collect();

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

    let max_cm = square_parts
        .iter()
        .filter_map(|s| s.parse::<u16>().ok())
        .max()
        .unwrap_or(400)
        .max(1);

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
        max_cm,
    )
}
