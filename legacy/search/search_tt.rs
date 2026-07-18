//! αβ transposition table — persists across plies in a [`GameSearchSession`](super::session::GameSearchSession).

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtBound {
    Exact,
    Lower,
    Upper,
}

pub const SEARCH_TT_BITS: usize = 20;
pub const SEARCH_TT_SIZE: usize = 1 << SEARCH_TT_BITS;
pub const SEARCH_TT_MASK: usize = SEARCH_TT_SIZE - 1;

#[derive(Clone, Copy, Default)]
pub struct SearchTtEntry {
    pub key: u64,
    pub depth: i8,
    pub score: i32,
    pub bound: u8,
    pub best: u32,
}

#[derive(Default)]
pub struct SearchTt {
    entries: Vec<SearchTtEntry>,
}

impl SearchTt {
    pub fn new() -> Self {
        Self {
            entries: vec![SearchTtEntry::default(); SEARCH_TT_SIZE],
        }
    }

    pub fn clear(&mut self) {
        self.entries.fill(SearchTtEntry::default());
    }

    pub fn probe(&self, key: u64) -> Option<SearchTtEntry> {
        let e = &self.entries[key as usize & SEARCH_TT_MASK];
        if e.key == key {
            Some(*e)
        } else {
            None
        }
    }

    pub fn store(&mut self, key: u64, depth: i8, score: i32, bound: TtBound, best: u32) {
        let slot = &mut self.entries[key as usize & SEARCH_TT_MASK];
        if slot.key != 0 && slot.key != key && slot.depth > depth {
            return;
        }
        *slot = SearchTtEntry {
            key,
            depth,
            score,
            bound: bound as u8,
            best,
        };
    }
}
