//! Distance-based CAT heat (legacy / tests only).
//!
//! Live board overlay uses `build::build_corridor_display_squares` in `viz.rs`
//! (per-player max corridor heat). Search uses the same max-merge in `build_corridor_attention`.

use crate::core::board::{Board, Player};
use crate::path::distance::fill_dist_from_sq;
use crate::path::masks::DirMasks;
use crate::path::BfsScratch;
use crate::util::grid::square_index;

/// Fixed scale for UI overlays: two players × 100 cm at dist 0.
pub const VIZ_MAX_CM: u16 = 200;

const VIZ_BASE_CM: u16 = 100;
const VIZ_DIST_PENALTY_CM: u16 = 3;
const VIZ_MAX_PENALTY_CM: u16 = 30;

/// `100 - clamp(dist × 3, 0, 30)` — see docs/video/CAT-SPEC.md.
#[inline]
pub fn attention_weight_cm(dist: u8) -> u16 {
    if dist == u8::MAX {
        return 0;
    }
    let penalty = u16::from(dist)
        .saturating_mul(VIZ_DIST_PENALTY_CM)
        .min(VIZ_MAX_PENALTY_CM);
    VIZ_BASE_CM.saturating_sub(penalty)
}

/// Combined two-player distance attention for board tinting.
pub fn build_viz_attention(scratch: &mut BfsScratch, board: &Board) -> [u16; 81] {
    let masks = DirMasks::from_board(board);
    let mut out = [0u16; 81];

    for player in [Player::One, Player::Two] {
        let (sr, sc) = board.pawn(player);
        let start = square_index(sr, sc);
        let (dist_from, _) = scratch.dist_scratch_mut();
        fill_dist_from_sq(start, masks, dist_from);

        for sq in 0usize..81 {
            let d = dist_from[sq];
            if d != u8::MAX {
                out[sq] = out[sq].saturating_add(attention_weight_cm(d));
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::grid::square_index;

    #[test]
    fn pawn_squares_hottest_at_startpos() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let viz = build_viz_attention(&mut scratch, &board);
        let white = viz[square_index(0, 4) as usize];
        let black = viz[square_index(8, 4) as usize];
        let e5 = viz[square_index(4, 4) as usize];
        // Each pawn: 100 cm (dist 0) + opponent BFS weight at dist 8 (76 cm).
        assert_eq!(white, 176);
        assert_eq!(black, 176);
        assert!(white >= 100, "pawn square includes own dist-0 weight");
        assert!(e5 >= 140 && e5 <= VIZ_MAX_CM, "e5 warm but not above scale");
    }

    #[test]
    fn mid_board_varies_by_column_at_startpos() {
        let board = Board::new();
        let mut scratch = BfsScratch::new();
        let viz = build_viz_attention(&mut scratch, &board);
        let d5 = viz[square_index(4, 3) as usize];
        let e5 = viz[square_index(4, 4) as usize];
        let a5 = viz[square_index(4, 0) as usize];
        assert!(a5 < e5, "edge files cooler than central corridor");
        assert!(d5 > 0 && e5 > 0);
    }
}
