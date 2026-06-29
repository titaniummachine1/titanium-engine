//! Corridor Attention Table (CAT) — per-square heat for search ordering (not eval).

use crate::core::board::WallOrientation;
use crate::util::grid::square_index;
use std::ops::Index;

/// Per-square attention scores for move ordering / LMR (centi-units, not eval).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CorridorAttention {
    pub(crate) square_heat: [u16; 81],
    pub(crate) route_flex: [u8; 81],
    pub(crate) bottleneck_heat: [u16; 81],
}

impl Default for CorridorAttention {
    fn default() -> Self {
        Self {
            square_heat: [0; 81],
            route_flex: [0; 81],
            bottleneck_heat: [0; 81],
        }
    }
}

impl Index<usize> for CorridorAttention {
    type Output = u16;

    fn index(&self, index: usize) -> &Self::Output {
        &self.square_heat[index]
    }
}

impl CorridorAttention {
    pub fn square_heat(&self, row: u8, col: u8) -> u16 {
        self.square_heat[square_index(row, col) as usize]
    }

    pub fn route_flex(&self, row: u8, col: u8) -> u8 {
        self.route_flex[square_index(row, col) as usize]
    }

    pub fn wall_edge_heat(&self, row: u8, col: u8, orientation: WallOrientation) -> u16 {
        let edge_heat = |a: (u8, u8), b: (u8, u8)| -> u16 {
            let ai = square_index(a.0, a.1) as usize;
            let bi = square_index(b.0, b.1) as usize;
            let ha = self.square_heat[ai];
            let hb = self.square_heat[bi];
            let hi = ha.max(hb);
            if hi == 0 {
                // both cells off-path: this wall touches no corridor
                return 0;
            }
            // A wall fully on the corridor (both cells hot) reads full; a wall
            // that only *touches* the corridor on one side still registers — at
            // lo + 40% of the gap — instead of collapsing to ~0 under min().
            // Walls touching the contested path are tactically live even when
            // they don't block the exact current edge.
            let lo = ha.min(hb);
            let corridor = lo + (hi - lo) * 2 / 5;
            let bottleneck = self.bottleneck_heat[ai].min(self.bottleneck_heat[bi]);
            corridor.saturating_add(bottleneck)
        };

        let (a, b) = match orientation {
            WallOrientation::Horizontal => (
                edge_heat((row, col), (row + 1, col)),
                edge_heat((row, col + 1), (row + 1, col + 1)),
            ),
            WallOrientation::Vertical => (
                edge_heat((row, col), (row, col + 1)),
                edge_heat((row + 1, col), (row + 1, col + 1)),
            ),
        };
        a.max(b).saturating_add(a.min(b) / 4)
    }
}
