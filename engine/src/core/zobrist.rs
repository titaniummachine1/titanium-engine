//! Zobrist hashing — incremental position keys for TT and search.

use crate::core::board::{Board, Player, WallOrientation};

pub struct ZobristKeys {
    pub pawn: [[u64; 81]; 2],
    pub horizontal_wall: [u64; 64],
    pub vertical_wall: [u64; 64],
    pub walls_left: [[u64; 11]; 2],
    pub side_to_move: u64,
}

impl ZobristKeys {
    fn new() -> Self {
        let mut seed = 0xA24BAED4963EE407u64;
        let mut next = || {
            seed = splitmix64(seed);
            seed
        };

        let mut pawn = [[0u64; 81]; 2];
        for player in 0..2 {
            for sq in 0..81 {
                pawn[player][sq] = next();
            }
        }

        let mut horizontal_wall = [0u64; 64];
        let mut vertical_wall = [0u64; 64];
        for slot in 0..64 {
            horizontal_wall[slot] = next();
            vertical_wall[slot] = next();
        }

        let mut walls_left = [[0u64; 11]; 2];
        for player in 0..2 {
            for count in 0..11 {
                walls_left[player][count] = next();
            }
        }

        Self {
            pawn,
            horizontal_wall,
            vertical_wall,
            walls_left,
            side_to_move: next(),
        }
    }
}

fn splitmix64(x: u64) -> u64 {
    let x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = x;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn wall_slot(row: u8, col: u8) -> usize {
    (row as usize) * 8 + col as usize
}

fn pawn_sq(row: u8, col: u8) -> usize {
    (row as usize) * 9 + col as usize
}

static KEYS: std::sync::OnceLock<ZobristKeys> = std::sync::OnceLock::new();

pub fn keys() -> &'static ZobristKeys {
    KEYS.get_or_init(ZobristKeys::new)
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

#[inline]
pub fn xor_pawn(hash: &mut u64, player: usize, row: u8, col: u8) {
    *hash ^= keys().pawn[player][pawn_sq(row, col)];
}

#[inline]
pub fn xor_wall(hash: &mut u64, orientation: WallOrientation, row: u8, col: u8) {
    let slot = wall_slot(row, col);
    match orientation {
        WallOrientation::Horizontal => *hash ^= keys().horizontal_wall[slot],
        WallOrientation::Vertical => *hash ^= keys().vertical_wall[slot],
    }
}

#[inline]
pub fn xor_walls_left(hash: &mut u64, player: usize, count: u8) {
    *hash ^= keys().walls_left[player][count as usize];
}

#[inline]
pub fn xor_side(hash: &mut u64) {
    *hash ^= keys().side_to_move;
}
