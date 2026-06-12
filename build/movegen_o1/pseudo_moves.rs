//! Pseudo-legal pawn catalog + localized key ingredients.
//!
//! Pipeline (no Monte Carlo):
//! 1. **Catalog** — 12 theoretical destinations per source square (on-board slots only).
//! 2. **Enemy** — at most 4 cardinal adjacencies that can change jump/slide; else "far".
//! 3. **Walls** — only slots read by `can_step` on our steps, jumps, and diagonal slides.
//! 4. **Table** — `key(walls, enemy) → bitmask ⊆ catalog` of truly legal moves.

pub const PSEUDO_SLOTS: usize = 12;
pub const OFF_BOARD: u8 = 255;

/// Semantic pseudo-legal destinations from `(sr, sc)`.
/// 0–3 steps, 4–7 jump-through, 8–11 diagonal slide when jump blocked.
pub fn pseudo_catalog(sr: u8, sc: u8) -> [u8; PSEUDO_SLOTS] {
    let sq = |r: i16, c: i16| -> u8 {
        if (0..=8).contains(&r) && (0..=8).contains(&c) {
            super::geometry::square_index(r as u8, c as u8)
        } else {
            OFF_BOARD
        }
    };
    let r = sr as i16;
    let c = sc as i16;
    [
        sq(r + 1, c),
        sq(r - 1, c),
        sq(r, c + 1),
        sq(r, c - 1),
        sq(r + 2, c),
        sq(r - 2, c),
        sq(r, c + 2),
        sq(r, c - 2),
        sq(r - 1, c + 1),
        sq(r - 1, c - 1),
        sq(r + 1, c + 1),
        sq(r + 1, c - 1),
    ]
}

pub fn on_board_pseudo_count(catalog: &[u8; PSEUDO_SLOTS]) -> u8 {
    catalog.iter().filter(|&&sq| sq != OFF_BOARD).count() as u8
}

/// Map true legal destination squares → subset bitmask over the pseudo catalog.
pub fn legal_subset_mask(catalog: &[u8; PSEUDO_SLOTS], legal_dests: &[u8], n: usize) -> u16 {
    let mut mask = 0u16;
    for &d in &legal_dests[..n] {
        for (slot, &sq) in catalog.iter().enumerate() {
            if sq != OFF_BOARD && sq == d {
                mask |= 1 << slot;
                break;
            }
        }
    }
    mask
}

/// Opponent modes that can affect our pawn: not adjacent, or on one in-bounds cardinal.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EnemyMode {
    /// Opponent placed far away — only cardinal steps / no jump interaction.
    Far,
    /// Opponent on `cardinals[i]` offset from our pawn.
    Cardinal(usize),
}

pub const CARDINAL_OFFSETS: [(i8, i8); 4] = [(1, 0), (-1, 0), (0, 1), (0, -1)];

pub fn in_bounds_cardinals(sr: u8, sc: u8) -> Vec<(i8, i8)> {
    CARDINAL_OFFSETS
        .iter()
        .copied()
        .filter(|&(dr, dc)| {
            let nr = sr as i16 + dr as i16;
            let nc = sc as i16 + dc as i16;
            (0..=8).contains(&nr) && (0..=8).contains(&nc)
        })
        .collect()
}

pub fn enemy_modes(sr: u8, sc: u8) -> Vec<EnemyMode> {
    let cardinals = in_bounds_cardinals(sr, sc);
    let mut modes = vec![EnemyMode::Far];
    for i in 0..cardinals.len() {
        modes.push(EnemyMode::Cardinal(i));
    }
    modes
}
