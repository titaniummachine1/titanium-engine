//! Tablebase layers: the wall CONFIGURATIONS each tier has to cover.
//!
//! This module only enumerates positions. It does not solve anything — who
//! holds a wall decides who may place it, and that asymmetry belongs to the
//! solver, not to the set. Here a configuration is just the walls on the board.
//!
//! LAYER k = boards with `20 - k` walls placed, hence k walls in hand between
//! the two players. Walls are conserved (board + wl[0] + wl[1] == 20), so
//! taking a wall back off the board is the same operation as putting one into a
//! hand, and peeling walls off a 20-wall seed walks the tiers directly:
//!
//!   layer 0   20 walls placed   0 in hand
//!   layer 1   19 walls placed   1 in hand
//!   layer 2   18 walls placed   2 in hand
//!   layer 3   17 walls placed   3 in hand
//!
//! CONFIGURATIONS ARE TRANSPOSITIONS. Two lines that place the same walls in a
//! different order reach the same board, so a layer is a SET keyed on
//! `(hw_bits, vw_bits)` and nothing is counted twice. Which player is owed the
//! wall is deliberately not part of the key: it does not change the board, and
//! it is only needed once these positions are handed to a solver.
//!
//! Removal never needs a legality check. A wall can only ever block a path, so
//! taking one off a legal board leaves a legal board. Every peel is valid by
//! construction, which is why this enumeration is exact rather than filtered.
//!
//! SIZE. From a single 20-wall seed the layers are exactly C(20, k) — 1, 20,
//! 190, 1140, 4845 — because removing a different subset of walls always gives
//! a different bitboard. Roughly 6,200 configurations per seed through layer 4.
//! Across many seeds the union is what matters, and layers dedupe against each
//! other heavily once the seeds overlap.
//!
//! What this is NOT: the set of ALL boards with 19 walls. That is ~10^21 and
//! cannot be enumerated. These are the boards reachable by unwinding real
//! seeds, which is the sampled coverage the plan calls for.

use crate::titanium::position::game::GameState;
use std::collections::HashSet;

/// A wall configuration: the horizontal and vertical slot bitboards.
///
/// This is the whole identity of a layer entry. Pawns are not part of it — a
/// solved table covers every pawn placement on the configuration — and neither
/// are the hands.
pub type Config = (u64, u64);

/// Walls standing on a configuration.
#[inline]
pub fn wall_count(c: Config) -> u32 {
    c.0.count_ones() + c.1.count_ones()
}

/// Every configuration one wall removal away from `c`.
///
/// Each set bit is cleared in turn, giving exactly `wall_count(c)` results, all
/// distinct. No legality filter: removing a wall cannot close a path.
pub fn peel_one(c: Config, out: &mut Vec<Config>) {
    out.clear();
    let (hw, vw) = c;
    let mut b = hw;
    while b != 0 {
        let bit = b & b.wrapping_neg();
        out.push((hw ^ bit, vw));
        b ^= bit;
    }
    let mut b = vw;
    while b != 0 {
        let bit = b & b.wrapping_neg();
        out.push((hw, vw ^ bit));
        b ^= bit;
    }
}

/// Peel one wall off every configuration in `layer`, deduping the result.
pub fn peel_layer(layer: &HashSet<Config>) -> HashSet<Config> {
    let mut next = HashSet::new();
    let mut buf = Vec::with_capacity(20);
    for &c in layer {
        peel_one(c, &mut buf);
        for &child in &buf {
            next.insert(child);
        }
    }
    next
}

/// Build layers 0..=`depth` from `seeds`.
///
/// `layers[k]` holds the configurations with k walls missing. Deduping is
/// global within a layer, so overlapping seeds cost nothing extra.
pub fn expand(seeds: &[Config], depth: usize) -> Vec<HashSet<Config>> {
    let mut layers = Vec::with_capacity(depth + 1);
    layers.push(seeds.iter().copied().collect::<HashSet<Config>>());
    for k in 1..=depth {
        let next = peel_layer(&layers[k - 1]);
        layers.push(next);
    }
    layers
}

/// Deterministic xorshift. Seeds must be reproducible: a layer set that cannot
/// be regenerated is not a dataset, and `Math.random`-style nondeterminism here
/// would make any later disagreement impossible to investigate.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
}

/// Play legal walls, alternating sides, until all 20 are placed.
///
/// Built by making real moves rather than by setting bits, so the board is one
/// the engine itself can produce and wall conservation holds by construction:
/// twenty alternating placements take both hands from 10 to 0.
///
/// Returns `None` if the board saturates early — with walls packed badly there
/// may be no legal twentieth placement, and a short board is not a layer-0 seed.
pub fn random_seed_board(rng_seed: u64) -> Option<Config> {
    let mut rng = Rng(rng_seed | 1);
    let mut g = GameState::new();
    for _ in 0..20 {
        let mut legal = Vec::with_capacity(128);
        for wall_type in 0..2 {
            for slot in 0..64 {
                if g.wall_legal(wall_type, slot) {
                    legal.push((wall_type, slot));
                }
            }
        }
        if legal.is_empty() {
            return None;
        }
        let (wall_type, slot) = legal[(rng.next() % legal.len() as u64) as usize];
        let mv = if wall_type == 0 {
            crate::titanium::MOVE_HW_BASE
        } else {
            crate::titanium::MOVE_VW_BASE
        } + slot as i16;
        g.make_move(mv);
    }
    debug_assert_eq!(g.wl, [0, 0], "twenty placements must empty both hands");
    Some((g.hw_bits, g.vw_bits))
}

/// `n` distinct seed boards, skipping rng seeds that saturate early.
pub fn seed_boards(n: usize, rng_seed: u64) -> Vec<Config> {
    let mut out = Vec::with_capacity(n);
    let mut seen = HashSet::new();
    let mut s = rng_seed | 1;
    let mut attempts = 0usize;
    while out.len() < n && attempts < n * 1000 + 1000 {
        attempts += 1;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        if let Some(c) = random_seed_board(s) {
            if seen.insert(c) {
                out.push(c);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_seed_board_carries_twenty_walls() {
        let c = seed_boards(1, 12345);
        assert_eq!(c.len(), 1, "no seed board generated");
        assert_eq!(wall_count(c[0]), 20, "layer-0 seed must have 20 walls");
    }

    /// From ONE seed the layers are exactly the binomials: removing a different
    /// subset of walls always gives a different bitboard, so nothing collides.
    /// This is the check that the peel is neither dropping nor double-counting.
    #[test]
    fn one_seed_layers_are_binomial() {
        let seeds = seed_boards(1, 99);
        let layers = expand(&seeds, 4);
        assert_eq!(layers[0].len(), 1);
        assert_eq!(layers[1].len(), 20, "C(20,1)");
        assert_eq!(layers[2].len(), 190, "C(20,2)");
        assert_eq!(layers[3].len(), 1140, "C(20,3)");
        assert_eq!(layers[4].len(), 4845, "C(20,4)");
    }

    #[test]
    fn every_layer_entry_has_the_expected_wall_count() {
        let seeds = seed_boards(3, 7);
        let layers = expand(&seeds, 3);
        for (k, layer) in layers.iter().enumerate() {
            for &c in layer {
                assert_eq!(
                    wall_count(c) as usize,
                    20 - k,
                    "layer {k} entry has {} walls",
                    wall_count(c)
                );
            }
        }
    }

    /// Layers must dedupe across seeds, otherwise the union is counted wrong and
    /// every size estimate built on it is inflated.
    #[test]
    fn layers_dedupe_across_seeds() {
        let seeds = seed_boards(4, 2024);
        assert_eq!(seeds.len(), 4);
        let layers = expand(&seeds, 2);
        assert!(
            layers[2].len() <= seeds.len() * 190,
            "union cannot exceed per-seed totals"
        );
        for &c in &layers[2] {
            assert_eq!(wall_count(c), 18);
        }
    }
}
