//! Research / under investigation — not production search.
//!
//! Prefer graduating to a real module or deleting; do not leave forever.

/// Exact zero-wall retrograde tablebase. Offline (`tb-*` CLI) and test-only:
/// the test suite uses it as an independent oracle proving
/// `cert_bridge::hands_empty_race_stm_wins` exact. Live search never probes it
/// — see `tests/repo_hygiene.rs::tablebase_is_not_reachable_from_live_search`.
pub mod tb_layers;
pub mod tb_zero;
pub mod wall_ignore_cert;
pub mod wall_ignore_cert_tests;
pub mod wall_ignore_corridor;
