//! Transposition table — perft node cache now, αβ search later.
//!
//! Stockfish-style **clustered buckets** (4 slots per index) to cut collisions.

const TT_CLUSTER: usize = 4;
const TT_BITS: usize = 18;
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: usize = TT_SIZE - 1;

#[derive(Clone, Copy, Default)]
struct Entry {
    key: u64,
    depth: u8,
    nodes: u64,
}

#[derive(Clone, Copy)]
struct Cluster {
    entries: [Entry; TT_CLUSTER],
}

impl Default for Cluster {
    fn default() -> Self {
        Self {
            entries: [Entry::default(); TT_CLUSTER],
        }
    }
}

pub struct TranspositionTable {
    clusters: Vec<Cluster>,
}

impl Default for TranspositionTable {
    fn default() -> Self {
        Self::new()
    }
}

impl TranspositionTable {
    pub fn new() -> Self {
        Self {
            clusters: vec![Cluster::default(); TT_SIZE],
        }
    }

    pub fn clear(&mut self) {
        self.clusters.fill(Cluster::default());
    }

    #[inline]
    pub fn probe(&self, key: u64, depth: u8) -> Option<u64> {
        let cluster = &self.clusters[(key as usize) & TT_MASK];
        for entry in &cluster.entries {
            if entry.key == key && entry.depth == depth {
                return Some(entry.nodes);
            }
        }
        None
    }

    #[inline]
    pub fn store(&mut self, key: u64, depth: u8, nodes: u64) {
        let cluster = &mut self.clusters[(key as usize) & TT_MASK];
        let mut replace = 0usize;
        let mut shallowest = u8::MAX;

        for (i, entry) in cluster.entries.iter().enumerate() {
            if entry.key == key {
                if entry.depth <= depth {
                    cluster.entries[i] = Entry { key, depth, nodes };
                }
                return;
            }
            if entry.key == 0 {
                cluster.entries[i] = Entry { key, depth, nodes };
                return;
            }
            if entry.depth < shallowest {
                shallowest = entry.depth;
                replace = i;
            }
        }

        cluster.entries[replace] = Entry { key, depth, nodes };
    }
}
