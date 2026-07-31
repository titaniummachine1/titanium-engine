//! Shared pathfinding algorithms.
//!
//! - `bff`: Binary Flood Fill primitives and wall-legality floods.
//! - `bfs`: distances, exact wavefront layers, and shortest paths.
//! - `incremental`: goal-seeded layer memoization for incremental wall trials.
//! - `masks`: the directional board representation shared by both algorithms.

pub mod bff;
pub mod bfs;
pub mod incremental;
pub mod masks;

pub use bff::wall::{
    bff_ks_to_goal, bff_ks_to_goal_cached, bff_ks_wall_legal, bff_to_goal, bff_to_goal_cached,
    bff_wall_legal, bff_wall_legal_board, wall_delta, WallGrids,
};
pub use bfs::{
    both_players_reach_goals, both_players_reach_goals_with_masks, can_reach_goal,
    shortest_distance, shortest_path, BfsScratch,
};
pub use incremental::{GoalField, Probe};
pub use masks::DirMasks;

#[cfg(test)]
mod tests;
