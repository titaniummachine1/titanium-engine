//! Engine context — **thread-ready layout**, single-thread by default.
//!
//! ```text
//! SharedState   → TT (later: Arc, shared across Lazy SMP workers)
//! WorkerContext → BfsScratch (one per thread)
//! Engine        → see `engine.rs`
//! ```

use crate::path::BfsScratch;
use crate::search::tt::TranspositionTable;

/// Shared across workers — transposition table today, search metadata later.
pub struct SharedState {
    pub tt: TranspositionTable,
}

impl Default for SharedState {
    fn default() -> Self {
        Self::new()
    }
}

impl SharedState {
    pub fn new() -> Self {
        Self {
            tt: TranspositionTable::new(),
        }
    }

    pub fn clear_tt(&mut self) {
        self.tt.clear();
    }
}

/// Per-thread scratch — never share mutably across threads.
#[derive(Clone)]
pub struct WorkerContext {
    pub bfs: BfsScratch,
}

impl Default for WorkerContext {
    fn default() -> Self {
        Self::new()
    }
}

impl WorkerContext {
    pub fn new() -> Self {
        Self {
            bfs: BfsScratch::new(),
        }
    }
}

/// Run configuration — `threads = 1` keeps CI deterministic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineLimits {
    pub threads: usize,
}

impl Default for EngineLimits {
    fn default() -> Self {
        Self { threads: 1 }
    }
}

impl EngineLimits {
    pub fn single_threaded() -> Self {
        Self { threads: 1 }
    }

    pub fn with_threads(threads: usize) -> Self {
        Self {
            threads: threads.max(1),
        }
    }
}

/// Result for `thread-bench` / video episode — 1 thread vs N at same node count.
#[derive(Debug, Clone, PartialEq)]
pub struct ThreadBenchResult {
    pub depth: u32,
    pub nodes: u64,
    pub threads_one_secs: f64,
    pub threads_n_secs: f64,
    pub threads_n: usize,
}

impl ThreadBenchResult {
    pub fn speedup(&self) -> f64 {
        if self.threads_n_secs > 0.0 {
            self.threads_one_secs / self.threads_n_secs
        } else {
            0.0
        }
    }
}
