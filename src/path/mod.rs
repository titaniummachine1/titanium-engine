//! Reachability — bitwise (bitboard) flood fill and `BfsScratch` (no CAT logic here; see `cat`).
//!
//! `parallel::pbff_*` = binary flood fill path-to-goal helpers for wall-legality trials.

pub mod bfs;
pub mod distance;
pub mod flood;
pub mod masks;
pub mod parallel;

pub use bfs::{
    both_players_reach_goals, both_players_reach_goals_with_masks, can_reach_goal,
    shortest_distance, BfsScratch,
};
pub use masks::DirMasks;
pub use parallel::{
    pbff_ks_to_goal, pbff_ks_to_goal_cached, pbff_ks_wall_legal, pbff_to_goal, pbff_to_goal_cached,
    pbff_wall_legal, pbff_wall_legal_board, wall_delta, WallGrids,
};

#[cfg(test)]
mod tests;
