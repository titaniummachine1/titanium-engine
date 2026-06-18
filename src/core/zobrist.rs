//! Zobrist hashing — incremental position keys for TT and search.
//!
//! Keys are generated at compile time (`const fn` splitmix64 stream), so
//! lookups are plain static loads — no `OnceLock` atomic on the make/unmake
//! hot path. Move hashing is exposed as fused single-xor deltas.

use crate::core::board::{Board, Player, WallOrientation};

pub struct ZobristKeys {
    pub pawn: [[u64; 81]; 2],
    pub horizontal_wall: [u64; 64],
    pub vertical_wall: [u64; 64],
    pub walls_left: [[u64; 11]; 2],
    pub side_to_move: u64,
}

impl ZobristKeys {
    /// Same key-stream order as the original runtime generator:
    /// pawns (2×81), wall slots interleaved H/V (64×2), walls_left (2×11), side.
    const fn new() -> Self {
        let mut seed = 0xA24BAED4963EE407u64;

        let mut pawn = [[0u64; 81]; 2];
        let mut player = 0;
        while player < 2 {
            let mut sq = 0;
            while sq < 81 {
                seed = splitmix64(seed);
                pawn[player][sq] = seed;
                sq += 1;
            }
            player += 1;
        }

        let mut horizontal_wall = [0u64; 64];
        let mut vertical_wall = [0u64; 64];
        let mut slot = 0;
        while slot < 64 {
            seed = splitmix64(seed);
            horizontal_wall[slot] = seed;
            seed = splitmix64(seed);
            vertical_wall[slot] = seed;
            slot += 1;
        }

        let mut walls_left = [[0u64; 11]; 2];
        let mut player = 0;
        while player < 2 {
            let mut count = 0;
            while count < 11 {
                seed = splitmix64(seed);
                walls_left[player][count] = seed;
                count += 1;
            }
            player += 1;
        }

        seed = splitmix64(seed);
        Self {
            pawn,
            horizontal_wall,
            vertical_wall,
            walls_left,
            side_to_move: seed,
        }
    }
}

const fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const fn wall_slot(row: u8, col: u8) -> usize {
    (row as usize) * 8 + col as usize
}

const fn pawn_sq(row: u8, col: u8) -> usize {
    (row as usize) * 9 + col as usize
}

static KEYS: ZobristKeys = ZobristKeys::new();

pub fn keys() -> &'static ZobristKeys {
    &KEYS
}

pub fn hash_board(board: &Board) -> u64 {
    let k = keys();
    let mut h = 0u64;

    for player in 0..2 {
        let (row, col) = board.pawns[player];
        h ^= k.pawn[player][pawn_sq(row, col)];
        h ^= k.walls_left[player][board.walls_remaining[player] as usize];
    }

    h ^= wall_bits_hash(board.horizontal_walls, &k.horizontal_wall);
    h ^= wall_bits_hash(board.vertical_walls, &k.vertical_wall);

    if board.side_to_move == Player::Two {
        h ^= k.side_to_move;
    }

    h
}

fn wall_bits_hash(bits: u64, table: &[u64; 64]) -> u64 {
    let mut h = 0u64;
    let mut b = bits;
    while b != 0 {
        let slot = b.trailing_zeros() as usize;
        h ^= table[slot];
        b &= b - 1;
    }
    h
}

/// Full hash delta for a pawn move: from-square out, to-square in, side flip.
#[inline]
pub fn pawn_move_delta(player: usize, from: (u8, u8), to: (u8, u8)) -> u64 {
    let k = keys();
    k.pawn[player][pawn_sq(from.0, from.1)] ^ k.pawn[player][pawn_sq(to.0, to.1)] ^ k.side_to_move
}

/// Full hash delta for a wall move: slot in, walls_left `n → n-1`, side flip.
#[inline]
pub fn wall_move_delta(
    orientation: WallOrientation,
    row: u8,
    col: u8,
    player: usize,
    walls_before: u8,
) -> u64 {
    let k = keys();
    let slot = wall_slot(row, col);
    let wall_key = match orientation {
        WallOrientation::Horizontal => k.horizontal_wall[slot],
        WallOrientation::Vertical => k.vertical_wall[slot],
    };
    wall_key
        ^ k.walls_left[player][walls_before as usize]
        ^ k.walls_left[player][walls_before as usize - 1]
        ^ k.side_to_move
}

#[inline]
pub fn xor_side(hash: &mut u64) {
    *hash ^= keys().side_to_move;
}
