//! Endgame reasoning — Layer 2.
//!
//! Owns race proofs, certify, and ExactDP. Search may call race/certify only.
//!
//! The exact zero-wall tablebase does NOT live here. It is a validation oracle
//! for `cert_bridge::hands_empty_race_stm_wins`, never a search component, so
//! it sits in `titanium::research` with the rest of the offline analysis.

pub mod cert_bridge;
pub mod certify;
pub mod exact_dp;
pub mod race;
