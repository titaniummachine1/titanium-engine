//! Root LMR plan snapshot — what search would do before / at a given ID depth.

use crate::cat::constants::DIST_PENALTY;
use crate::cat::prune::{
    get_shortest_path, is_tactical_move, move_corridor_attention, order_moves,
};
use crate::cat::CorridorAttention;
use crate::core::board::{Board, Move};
use crate::movegen::{generate_legal_moves_slice, MAX_LEGAL_MOVES};
use crate::path::BfsScratch;
use crate::search::lmr_profile::{build_lmr_table, compute_stage_t, LmrProfile};
use crate::util::perft::format_move;

const LMR_MIN_DEPTH: u32 = 2;
const ROOT_WALL_CAP_OPENING: usize = 26;
const ROOT_WALL_CAP_MID: usize = 38;

#[derive(Debug, Clone)]
pub struct RootLmrPlan {
    pub mv: String,
    pub is_pawn: bool,
    pub order: usize,
    pub cat_cm: i32,
    pub tactical: bool,
    pub hot: bool,
    pub pruned: bool,
    pub reduction: u32,
    pub child_depth_full: u32,
    pub child_depth_used: u32,
    pub in_full_window: bool,
}

fn cap_root_wall_moves(buf: &mut [Move], n: &mut usize, cat: &CorridorAttention, max_walls: usize) {
    let mut wall_count = 0usize;
    let mut write = 0usize;
    for i in 0..*n {
        if matches!(buf[i], Move::Wall { .. }) {
            wall_count += 1;
            if wall_count > max_walls {
                continue;
            }
        }
        if write != i {
            buf[write] = buf[i];
        }
        write += 1;
    }
    *n = write;
}

fn root_cat_heat_stats(
    board: &Board,
    moves: &[Move],
    n: usize,
    cat: &CorridorAttention,
) -> (u16, u16) {
    let mut heats = Vec::with_capacity(n);
    for mv in &moves[..n] {
        heats.push(move_corridor_attention(board, *mv, cat).max(0) as u16);
    }
    if heats.is_empty() {
        return (0, 0);
    }
    heats.sort_by(|a, b| b.cmp(a));
    let max = heats[0];
    let p75_idx = (heats.len() * 3 / 4).min(heats.len() - 1);
    (max, heats[p75_idx])
}

/// Planned root LMR for `id_depth` at `pierce_fraction` elapsed (0 = pierce peak).
pub fn plan_root_lmr(
    board: &mut Board,
    bfs: &mut BfsScratch,
    id_depth: u32,
    time_ms: u64,
    pierce_fraction: f32,
) -> (LmrProfile, Vec<RootLmrPlan>) {
    let root_side = board.side();
    let opp_side = root_side.opposite();
    let our_dist = bfs
        .shortest_distance(board, root_side)
        .unwrap_or(DIST_PENALTY);
    let opp_dist = bfs
        .shortest_distance(board, opp_side)
        .unwrap_or(DIST_PENALTY);
    let endgame_race = our_dist.min(opp_dist) <= 4;

    let mut buf = [Move::Pawn { row: 1, col: 4 }; MAX_LEGAL_MOVES];
    let n0 = generate_legal_moves_slice(board, &mut buf, bfs);
    let cat = bfs.build_corridor_attention(board);
    let (cat_max, cat_p75) = root_cat_heat_stats(board, &buf, n0, &cat);
    let stage_t = compute_stage_t(board, our_dist, opp_dist, cat_max, cat_p75);

    let mut profile = LmrProfile::from_stage(stage_t, endgame_race, false);
    profile.apply_time_budget(time_ms);
    profile.apply_pierce_schedule(pierce_fraction, time_ms);

    let mut n = n0;
    if id_depth >= 3 {
        let cap = profile.root_wall_cap().min(if profile.stage_t < 0.40 {
            ROOT_WALL_CAP_OPENING
        } else {
            ROOT_WALL_CAP_MID
        });
        cap_root_wall_moves(&mut buf, &mut n, &cat, cap);
    }

    let mut scores = [0i32; MAX_LEGAL_MOVES];
    let mut opp_path = [0u8; 81];
    let opp_path_len = get_shortest_path(board, opp_side, bfs, &mut opp_path);
    order_moves(
        board,
        &mut buf,
        n,
        None,
        None,
        &mut scores,
        our_dist,
        opp_dist,
        &opp_path,
        opp_path_len,
        bfs,
        &cat,
    );

    let lmr_table = build_lmr_table(profile.aggression);
    let depth = id_depth.max(1);
    let child_depth_full = depth.saturating_sub(1);

    let mut plans = Vec::with_capacity(n);
    let mut moves_searched = 0usize;

    for i in 0..n {
        let mv = buf[i];
        let cat_cm = move_corridor_attention(board, mv, &cat);
        let heat_ratio_hot = cat_max > 0
            && (cat_cm.max(0) as u32) * 100 >= (cat_max as u32) * u32::from(profile.hot_ratio_pct);
        let corridor_relevant = cat_cm >= i32::from(profile.cold_cm);
        let full_depth_slots = profile.move_window.max(profile.lmr_after_move);
        let in_full_window = moves_searched < full_depth_slots;
        let is_tactical =
            if moves_searched == 0 || depth < LMR_MIN_DEPTH || in_full_window || heat_ratio_hot {
                true
            } else if matches!(mv, Move::Wall { .. })
                && !crate::cat::prune::wall_intersects_path(mv, &opp_path, opp_path_len)
            {
                false
            } else {
                is_tactical_move(board, mv, our_dist, opp_dist, bfs)
            };

        let pruned = false;

        let reduction =
            if (moves_searched == 0) || is_tactical || depth < LMR_MIN_DEPTH || heat_ratio_hot {
                0u32
            } else {
                let d = (depth as usize).min(63);
                let m = (i + 1).min(63);
                let base_r = lmr_table[d][m];
                let gap = cat_max.saturating_sub(cat_cm.max(0) as u16);
                let cat_extra = (gap as f32 * profile.cat_heat_lmr_slope).round() as u32;
                let wall_extra = if matches!(mv, Move::Wall { .. }) && cat_cm == 0 {
                    4u32
                } else if matches!(mv, Move::Wall { .. })
                    && !crate::cat::prune::wall_intersects_path(mv, &opp_path, opp_path_len)
                    && !corridor_relevant
                {
                    3u32
                } else if cat_cm < i32::from(profile.cold_cm) {
                    if profile.stage_t < 0.35 {
                        3u32
                    } else {
                        1u32
                    }
                } else {
                    0u32
                };
                (base_r + wall_extra + cat_extra).min(depth.saturating_sub(1))
            };

        let child_depth_used = if pruned {
            0
        } else {
            child_depth_full.saturating_sub(reduction)
        };

        plans.push(RootLmrPlan {
            mv: format_move(mv),
            is_pawn: matches!(mv, Move::Pawn { .. }),
            order: moves_searched,
            cat_cm,
            tactical: is_tactical,
            hot: heat_ratio_hot,
            pruned,
            reduction,
            child_depth_full,
            child_depth_used,
            in_full_window,
        });

        if !pruned {
            moves_searched += 1;
        }
    }

    (profile, plans)
}

pub fn lmr_profile_fields(profile: &LmrProfile, id_depth: u32) -> String {
    format!(
        "{{\"stageT\":{:.3},\"aggression\":{:.2},\"pierceT\":{:.3},\"moveWindow\":{},\"lmrAfter\":{},\"hotPct\":{},\"idDepth\":{}}}",
        profile.stage_t,
        profile.aggression,
        profile.pierce_t,
        profile.move_window,
        profile.lmr_after_move,
        profile.hot_ratio_pct,
        id_depth,
    )
}

pub fn format_lmr_plans_json(plans: &[RootLmrPlan]) -> String {
    let mut out = String::new();
    for (i, p) in plans.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"move\":\"{}\",\"kind\":\"{}\",\"order\":{},\"catCm\":{},\"tactical\":{},\"hot\":{},\"pruned\":{},\"reduction\":{},\"childDepthFull\":{},\"childDepthUsed\":{},\"inFullWindow\":{}}}",
            p.mv,
            if p.is_pawn { "pawn" } else { "wall" },
            p.order,
            p.cat_cm,
            p.tactical,
            p.hot,
            p.pruned,
            p.reduction,
            p.child_depth_full,
            p.child_depth_used,
            p.in_full_window,
        ));
    }
    out
}

/// Pre-search LMR plan — static profile at pierce peak, depth 2 (first ply reductions apply).
/// No negamax run; mirrors what the first real ID iterations will slice.
pub fn lmr_snapshot_json(board: &mut Board, time_ms: u64) -> String {
    let mut bfs = BfsScratch::new();
    const SHALLOW_PLAN_DEPTH: u32 = 2;
    let (profile, plans) = plan_root_lmr(board, &mut bfs, SHALLOW_PLAN_DEPTH, time_ms, 0.0);
    format!(
        "{{\"source\":\"shallow\",\"idDepth\":{},\"timeMs\":{},\"lmrProfile\":{},\"moves\":[{}]}}",
        SHALLOW_PLAN_DEPTH,
        time_ms,
        lmr_profile_fields(&profile, SHALLOW_PLAN_DEPTH),
        format_lmr_plans_json(&plans),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::Board;
    use crate::search::lmr_profile::TIME_REFERENCE_MS;

    #[test]
    fn shallow_snapshot_has_legal_moves() {
        let mut board = Board::new();
        let mut bfs = BfsScratch::new();
        let (_, plans) = plan_root_lmr(&mut board, &mut bfs, 1, TIME_REFERENCE_MS, 0.0);
        assert!(plans.len() >= 4);
        assert!(plans[0].tactical);
    }
}
