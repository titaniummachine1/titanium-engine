//! Packed move keys for TT, killers, and history tables.

use crate::core::board::{Move, WallOrientation};

#[inline]
pub fn pack_move(mv: Move) -> u32 {
    match mv {
        Move::Pawn { row, col } => 1 | (u32::from(row) << 8) | (u32::from(col) << 16),
        Move::Wall {
            row,
            col,
            orientation,
        } => {
            let o = match orientation {
                WallOrientation::Horizontal => 0u32,
                WallOrientation::Vertical => 1,
            };
            2 | (u32::from(row) << 8) | (u32::from(col) << 16) | (o << 24)
        }
    }
}

#[inline]
pub fn unpack_move(packed: u32) -> Option<Move> {
    match packed & 0xFF {
        0 => None,
        1 => Some(Move::Pawn {
            row: ((packed >> 8) & 0xFF) as u8,
            col: ((packed >> 16) & 0xFF) as u8,
        }),
        2 => Some(Move::Wall {
            row: ((packed >> 8) & 0xFF) as u8,
            col: ((packed >> 16) & 0xFF) as u8,
            orientation: if (packed >> 24) & 1 == 0 {
                WallOrientation::Horizontal
            } else {
                WallOrientation::Vertical
            },
        }),
        _ => None,
    }
}
