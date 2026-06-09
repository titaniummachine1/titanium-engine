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
            let corridor = self.square_heat[ai].min(self.square_heat[bi]);
            if corridor == 0 {
                return 0;
            }
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
