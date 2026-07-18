//! NNUE per-cell field plane names — read this before touching eval or training.
//!
//! ## Philosophy (pre-training)
//! The NN must not learn Quoridor topology from scratch. Search/BFS gives exact geometry;
//! the NN compresses it into priors so H=32 capacity goes to strategy (wall timing,
//! sacrifices, traps) — not rediscovering graph branching.
//!
//! **NN = geometric prior · Search = tactical proof**
//!
//! Do NOT add: per-wall delta floods, extra BFS, raw board planes (those belong in search).
//! Target blob ~552 KB, H=32 fixed.
//!
//! All planes use ACE cell index (row 0 = top). Both players always fed (no STM mirror).
//! Values in eval are divided by 16 before multiply with weights.
//!
//! | Weight (net.rs)      | JSON key (eval --json)        | Meaning |
//! |----------------------|-------------------------------|---------|
//! | goal_inv_p0          | goal_inv_p0_field             | Inverse BFS: steps from cell → P0 goal row |
//! | goal_inv_p1          | goal_inv_p1_field             | Same for P1 |
//! | pawn_fwd_p0          | pawn_fwd_p0_field             | Forward BFS: steps from P0 pawn |
//! | pawn_fwd_p1          | pawn_fwd_p1_field             | Same for P1 |
//! | corridor_delta_p0    | corridor_delta_p0_field       | from+to−shortest; route flexibility |
//! | corridor_delta_p1    | corridor_delta_p1_field       | Same for P1 |
//! | path_cross_p0        | path_cross_p0_field           | Route overlap: many paths pass here |
//! | path_cross_p1        | path_cross_p1_field           | Same for P1 |
//! | choke_p0             | choke_p0_field                | Forcedness: 1/(1+forward_continuations) |
//! | choke_p1             | choke_p1_field                | Same for P1 |
//! | contested            | contested_field               | Shared: 1/(1+delta_p0+delta_p1) |
//!
//! **path_cross** vs **choke**: highway has high cross / low choke; one-lane bridge has
//! low cross / high choke. Both needed.
//!
//! Legacy JSON aliases (trainer still accepts): d0_field, player0_field, delta0_field, …
//!
//! Scalars (not planes): d0/d1 = shortest path length at each pawn. All 16 ws slots used.

//! ## Type layers (do not confuse these)
//!
//! | Layer | Types | Role |
//! |-------|-------|------|
//! | Board / flood | `u128` frontier, `hw[64]`+`vw[64]` walls | Exact combinatorics, fast parallel BFS |
//! | Feature geometry | `u8` per cell (÷16 in eval) | BFS output compressed to scalars |
//! | NN accumulators | `f64` weights × feature | Stable weighted sum into H=32 hidden |
//!
//! W1C "128" = 64 horizontal + 64 vertical **wall slot indices**, not a single 128-bit feature.
//! Field planes are **81-cell vectors** (one scalar per square), not bitboards.

pub const GOAL_INV_P0: &str = "goal_inv_p0_field";
pub const GOAL_INV_P1: &str = "goal_inv_p1_field";
pub const PAWN_FWD_P0: &str = "pawn_fwd_p0_field";
pub const PAWN_FWD_P1: &str = "pawn_fwd_p1_field";
pub const CORRIDOR_DELTA_P0: &str = "corridor_delta_p0_field";
pub const CORRIDOR_DELTA_P1: &str = "corridor_delta_p1_field";
pub const PATH_CROSS_P0: &str = "path_cross_p0_field";
pub const PATH_CROSS_P1: &str = "path_cross_p1_field";
pub const CHOKE_P0: &str = "choke_p0_field";
pub const CHOKE_P1: &str = "choke_p1_field";
pub const CONTESTED: &str = "contested_field";
