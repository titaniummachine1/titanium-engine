//! Endgame reasoning — Layer 2.
//!
//! Owns race proofs, certify, and ExactDP. Search may call race/certify only.

pub mod cert_bridge;
pub mod certify;
pub mod tb_layers;
pub mod tb_zero;
pub mod exact_dp;
pub mod race;
