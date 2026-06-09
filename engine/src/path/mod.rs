//! Reachability — bitwise flood fill and `BfsScratch` (no CAT logic here; see `cat`).

pub mod bfs;
pub mod distance;
pub mod flood;
pub mod masks;

pub use bfs::{
    both_players_reach_goals, can_reach_goal, shortest_distance, BfsScratch,
};
pub use masks::DirMasks;

#[cfg(test)]
mod tests;
