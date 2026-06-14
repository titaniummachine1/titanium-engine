//! Transposition table — perft node cache now, αβ search later.
//!
//! Stockfish-style **clustered buckets** (4 slots per index) to cut collisions.

const TT_CLUSTER: usize = 4;
/// Cluster size in bytes: 4 entries × 24 B = 96 B.
const TT_CLUSTER_BYTES: usize = TT_CLUSTER * 24;

/// Size/performance table (native, perft, `tt_speedup` bench):
///   | bits | RAM    | perft(3) | perft(4) | perft(5) |  fits in        |
///   |   9  |  48 KB |  105 ms  |  1.05 s  |    —     |  L1/core (64K) | ← start
///   |  11  | 192 KB |  103 ms  |  0.83 s  |    —     |  L2/core (256K)| ← L1→L2
///   |  16  |   6 MB |  111 ms  |  0.44 s  |    —     |  L3 (8 MB)     | ← L2→L3
///   |  18  |  24 MB |  119 ms  |  0.37 s  | 20.7 s   |                | ← L3→18
///   |  22  | 384 MB |  119 ms  |  0.63 s  | 12.6 s   |                | ← 18→22
///   |  24  | 1.5 GB |    —     |  0.76 s  | 13.4 s   |                |
///
/// Working-set knee per depth: d3≈11, d4≈18, d5≈22 (~7 bits per ply).
///
/// **Adaptive mode (default) — overflow-driven cache-tier jumps:**
///
///   L1  (bits=9,  48 KB) — start here; d1/d2 never overflow, zero page-fault cost.
///   L2  (bits=11, 192 KB) — on L1 overflow; rehashes 48 KB. d3 working set fits here.
///   L3  (bits=16,  6 MB) — on L2 overflow; rehashes 192 KB.
///   d4  (bits=18, 24 MB) — on L3 overflow; rehashes 6 MB.   d4 optimal landing.
///   d5  (bits=22, 384 MB) — on d4 overflow; rehashes 24 MB. d5 optimal landing.
///   +1  past d5           — +1 bit per overflow (d6+ territory).
///
/// Each overflow jumps to the next calibrated level, rehashing only the
/// CURRENT (small) table. Cleared TT retains its size — in game search the
/// TT grows once per session and subsequent searches reuse it at no cost.
///
/// NOTE: isolated perft calls create fresh TTs, so each run pays the grow
/// cost from L1. Pin a static size with `TT_BITS=N` for dedicated perft
/// benchmarking (`TT_BITS=18` for d4, `TT_BITS=22` for d5).
///
/// Override env vars (all accept 8..=27):
///   `TT_BITS=N`      — pin static size, disable growth
///   `TT_START_BITS`  — L1-phase start (default 9)
///   `TT_L2_BITS`     — L1→L2 jump target (default 11)
///   `TT_L3_BITS`     — L2→L3 jump target (default 16)
///   `TT_D4_BITS`     — L3→d4 jump target (default 18)
///   `TT_D5_BITS`     — d4→d5 jump target (default 22)
///   `TT_MAX_BITS`    — growth ceiling (default 25, ~3.2 GB)

// Hardware calibration (i7-4900MQ, 4 cores):
//   L1/core = 64 KB → 2^9 clusters × 96 B = 48 KB fits
//   L2/core = 256 KB → 2^11 × 96 B = 192 KB fits
//   L3      = 8 MB  → 2^16 × 96 B = 6 MB fits
//   d4 measured optimal = bits=18 (24 MB)
//   d5 measured optimal = bits=22 (384 MB)
const DEFAULT_START_BITS: usize = 9;
const DEFAULT_L2_BITS: usize = 11;
const DEFAULT_L3_BITS: usize = 16;
const DEFAULT_D4_BITS: usize = 18;
const DEFAULT_D5_BITS: usize = 22;
const DEFAULT_MAX_BITS: usize = 25;

// NOTE: 16-byte packed layout tried at perft(4) and (5) — no speedup. Engine
// is compute-bound on TT-miss nodes, not memory-bound. See `benches/tt_speedup.rs`.
//
// COLLISION SAFETY: 64-bit `key` alone can't prove board identity. `verify` is an
// independent 32-bit hash (`Board::tt_verify`); false hit needs BOTH (~2^-96/pair).
// FREE: {key:8, nodes:8, verify:4, depth:1} = 21 B padded to 24 B — no cache cost.
//
// EVICTION: depth-only (shallowest entry in a full cluster). walls_total-primary
// policy MEASURED and REJECTED: regressed d5 ~10%.
#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    nodes: u64,
    verify: u32,
    depth: u8,
}

#[derive(Clone, Copy)]
struct Cluster {
    entries: [Entry; TT_CLUSTER],
}

impl Default for Cluster {
    fn default() -> Self {
        Self { entries: [Entry::default(); TT_CLUSTER] }
    }
}

pub struct TranspositionTable {
    clusters: Vec<Cluster>,
    mask: usize,
    bits: usize,
    /// Non-empty slots consumed (grow trigger).
    filled: usize,
    l2_bits: usize,
    l3_bits: usize,
    d4_bits: usize,
    d5_bits: usize,
    max_bits: usize,
    /// False when `TT_BITS` was pinned (static size, no growth).
    adaptive: bool,
}

impl Default for TranspositionTable {
    fn default() -> Self { Self::new() }
}

fn env_bits(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|&b| (8..=27).contains(&b))
}

impl TranspositionTable {
    pub fn new() -> Self {
        if let Some(bits) = env_bits("TT_BITS") {
            return Self::make(bits, bits, bits, bits, bits, bits, false);
        }
        let start = env_bits("TT_START_BITS").unwrap_or(DEFAULT_START_BITS);
        let l2 = env_bits("TT_L2_BITS").unwrap_or(DEFAULT_L2_BITS).max(start);
        let l3 = env_bits("TT_L3_BITS").unwrap_or(DEFAULT_L3_BITS).max(l2);
        let d4 = env_bits("TT_D4_BITS").unwrap_or(DEFAULT_D4_BITS).max(l3);
        let d5 = env_bits("TT_D5_BITS").unwrap_or(DEFAULT_D5_BITS).max(d4);
        let max = env_bits("TT_MAX_BITS").unwrap_or(DEFAULT_MAX_BITS).max(d5);
        Self::make(start, l2, l3, d4, d5, max, true)
    }

    fn make(
        bits: usize,
        l2_bits: usize, l3_bits: usize,
        d4_bits: usize, d5_bits: usize,
        max_bits: usize,
        adaptive: bool,
    ) -> Self {
        let size = 1usize << bits;
        Self {
            clusters: vec![Cluster::default(); size],
            mask: size - 1,
            bits,
            filled: 0,
            l2_bits, l3_bits, d4_bits, d5_bits,
            max_bits,
            adaptive,
        }
    }

    pub fn clear(&mut self) {
        self.clusters.fill(Cluster::default());
        self.filled = 0;
        // Size NOT reset — game search TT grows once per session and stays.
    }

    pub fn size_bytes(&self) -> usize {
        self.clusters.len() * TT_CLUSTER_BYTES
    }

    #[inline]
    fn should_grow(&self) -> bool {
        self.adaptive
            && self.bits < self.max_bits
            && self.filled * 2 >= self.clusters.len() * TT_CLUSTER
    }

    #[inline]
    pub fn probe(&self, key: u64, verify: u32, depth: u8) -> Option<u64> {
        let cluster = &self.clusters[(key as usize) & self.mask];
        for entry in &cluster.entries {
            if entry.key == key && entry.verify == verify && entry.depth == depth {
                return Some(entry.nodes);
            }
        }
        None
    }

    #[inline]
    pub fn store(&mut self, key: u64, verify: u32, depth: u8, nodes: u64) {
        if Self::insert_into(&mut self.clusters, self.mask, key, verify, depth, nodes) {
            self.filled += 1;
            if self.should_grow() {
                self.grow();
            }
        }
    }

    #[inline]
    fn insert_into(
        clusters: &mut [Cluster],
        mask: usize,
        key: u64,
        verify: u32,
        depth: u8,
        nodes: u64,
    ) -> bool {
        let cluster = &mut clusters[(key as usize) & mask];
        let mut replace = 0usize;
        let mut shallowest = u8::MAX;

        for (i, entry) in cluster.entries.iter().enumerate() {
            if entry.key == key && entry.verify == verify {
                if entry.depth <= depth {
                    cluster.entries[i] = Entry { key, nodes, verify, depth };
                }
                return false;
            }
            if entry.key == 0 {
                cluster.entries[i] = Entry { key, nodes, verify, depth };
                return true;
            }
            if entry.depth < shallowest {
                shallowest = entry.depth;
                replace = i;
            }
        }
        cluster.entries[replace] = Entry { key, nodes, verify, depth };
        false
    }

    /// Overflow-driven jump chain (each rehashes only the current tier table):
    ///   L1(9) → L2(11):  rehash  48 KB
    ///   L2    → L3(16):  rehash 192 KB
    ///   L3    → d4(18):  rehash   6 MB  (d4 measured optimal)
    ///   d4    → d5(22):  rehash  24 MB  (d5 measured optimal)
    ///   past d5: +1 bit per overflow
    fn grow(&mut self) {
        let new_bits = if self.bits < self.l2_bits {
            self.l2_bits
        } else if self.bits < self.l3_bits {
            self.l3_bits
        } else if self.bits < self.d4_bits {
            self.d4_bits
        } else if self.bits < self.d5_bits {
            self.d5_bits
        } else {
            self.bits + 1
        }
        .min(self.max_bits);

        let new_size = 1usize << new_bits;
        let new_mask = new_size - 1;
        let mut next = vec![Cluster::default(); new_size];
        for cluster in &self.clusters {
            for e in &cluster.entries {
                if e.key != 0 {
                    Self::insert_into(&mut next, new_mask, e.key, e.verify, e.depth, e.nodes);
                }
            }
        }
        self.clusters = next;
        self.mask = new_mask;
        self.bits = new_bits;
    }
}
