//! Endgame reasoning — Layer 2.
//!
//! Owns race proofs, certify, and ExactDP. Search may call race/certify only.

pub mod cert_bridge;
pub mod certify;
pub mod exact_dp;
pub mod race;
