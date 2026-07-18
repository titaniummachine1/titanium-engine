//! ACE v7 game state — 1:1 port of the JS `Quoridor` object.
//!
//! Coordinates are ACE-native: cell = r*9+c with r=0 the TOP row.
//! Player 0 starts at 76 (bottom) and races to row 0; player 1 starts at 4
//! and races to row 8. Moves: 0..80 pawn target, 81+slot hw, 145+slot vw.

pub const DELTA: [i16; 4] = [-9, 9, -1, 1];
pub const DIRBIT: [u8; 4] = [1, 2, 4, 8];

const fn ace_goal_bits(row: usize) -> u128 {
    let mut bits = 0u128;
    let mut c = 0usize;
    while c < 9 {
        bits |= crate::util::grid::FLOOD_BIT_BY_SQ[row * 9 + c];
        c += 1;
    }
    bits
}

const ACE_P0_GOAL_BITS: u128 = ace_goal_bits(0);
const ACE_P1_GOAL_BITS: u128 = ace_goal_bits(8);

#[derive(Clone, Copy)]
struct BitfillPathCheck {
    open: bool,
    p0_goal_depth: u8,
    p1_seeded_proof_depth: u8,
}

// ── Zobrist (exact JS xorshift sequence so hashes match the reference) ───────

pub struct Zobrist {
    pub pawn_lo: [[u32; 81]; 2],
    pub pawn_hi: [[u32; 81]; 2],
    pub hw_lo: [u32; 64],
    pub hw_hi: [u32; 64],
    pub vw_lo: [u32; 64],
    pub vw_hi: [u32; 64],
    pub turn_lo: u32,
    pub turn_hi: u32,
}

const fn zrand(seed: u32) -> u32 {
    let mut s = seed;
    s ^= s << 13;
    s ^= s >> 17;
    s ^= s << 5;
    s
}

const fn build_zobrist() -> Zobrist {
    let mut z = Zobrist {
        pawn_lo: [[0; 81]; 2],
        pawn_hi: [[0; 81]; 2],
        hw_lo: [0; 64],
        hw_hi: [0; 64],
        vw_lo: [0; 64],
        vw_hi: [0; 64],
        turn_lo: 0,
        turn_hi: 0,
    };
    let mut seed: u32 = 0x9e3779b9;
    let mut zi = 0;
    while zi < 2 {
        let mut zj = 0;
        while zj < 81 {
            seed = zrand(seed);
            z.pawn_lo[zi][zj] = seed;
            seed = zrand(seed);
            z.pawn_hi[zi][zj] = seed;
            zj += 1;
        }
        zi += 1;
    }
    let mut zs = 0;
    while zs < 64 {
        seed = zrand(seed);
        z.hw_lo[zs] = seed;
        seed = zrand(seed);
        z.hw_hi[zs] = seed;
        seed = zrand(seed);
        z.vw_lo[zs] = seed;
        seed = zrand(seed);
        z.vw_hi[zs] = seed;
        zs += 1;
    }
    seed = zrand(seed);
    z.turn_lo = seed;
    seed = zrand(seed);
    z.turn_hi = seed;
    z
}

pub static ZOBRIST: Zobrist = build_zobrist();

const fn build_border() -> [u8; 81] {
    let mut border = [0u8; 81];
    let mut bc = 0;
    while bc < 81 {
        let br = bc / 9;
        let bcl = bc % 9;
        border[bc] = (if br == 0 { 1 } else { 0 })
            | (if br == 8 { 2 } else { 0 })
            | (if bcl == 0 { 4 } else { 0 })
            | (if bcl == 8 { 8 } else { 0 });
        bc += 1;
    }
    border
}

pub static BORDER: [u8; 81] = build_border();

// ── Game state ────────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct GameState {
    pub pawn: [usize; 2],
    pub wl: [i32; 2],
    pub turn: usize,
    pub hw: [u8; 64],
    pub vw: [u8; 64],
    /// Packed horizontal wall slots (bit `slot` = wall at slot).
    pub hw_bits: u64,
    /// Packed vertical wall slots.
    pub vw_bits: u64,
    /// Wall-blocked direction bits per cell: N=1 S=2 W=4 E=8 (bounds via BORDER).
    pub blocked: [u8; 81],
    pub hash_lo: u32,
    pub hash_hi: u32,
    pub hist_m: [i16; 1024],
    pub hist_from: [i16; 1024],
    pub hist_lw: [i16; 1024],
    pub hashes_u: [u32; 2048],
    pub hist_len: usize,
    /// Repetition can only reach back to the last wall placement.
    pub last_wall_ply: usize,
    /// Bumped on every wall make/unmake; dist fields depend only on walls.
    pub wall_stamp: i32,
}

impl Default for GameState {
    fn default() -> Self {
        Self::new()
    }
}

impl GameState {
    pub fn new() -> Self {
        let z = &ZOBRIST;
        Self {
            pawn: [76, 4],
            wl: [10, 10],
            turn: 0,
            hw: [0; 64],
            vw: [0; 64],
            hw_bits: 0,
            vw_bits: 0,
            blocked: [0; 81],
            hash_lo: z.pawn_lo[0][76] ^ z.pawn_lo[1][4],
            hash_hi: z.pawn_hi[0][76] ^ z.pawn_hi[1][4],
            hist_m: [0; 1024],
            hist_from: [0; 1024],
            hist_lw: [0; 1024],
            hashes_u: [0; 2048],
            hist_len: 0,
            last_wall_ply: 0,
            wall_stamp: 0,
        }
    }

    #[inline(always)]
    pub fn can_step(&self, cell: usize, dir: usize) -> bool {
        crate::bench_instr::count(
            |b| &mut b.can_step,
            || ((self.blocked[cell] | BORDER[cell]) & DIRBIT[dir]) == 0,
        )
    }

    pub fn winner(&self) -> i32 {
        if self.pawn[0] < 9 {
            return 0;
        }
        if self.pawn[1] >= 72 {
            return 1;
        }
        -1
    }

    // ── Wall mechanics ──────────────────────────────────────────────────────

    #[inline]
    fn pack_wall_bits(arr: &[u8; 64]) -> u64 {
        let mut bits = 0u64;
        for (s, &on) in arr.iter().enumerate() {
            if on != 0 {
                bits |= 1u64 << s;
            }
        }
        bits
    }

    #[cfg(debug_assertions)]
    pub fn assert_wall_bits_sync(&self) {
        debug_assert_eq!(self.hw_bits, Self::pack_wall_bits(&self.hw));
        debug_assert_eq!(self.vw_bits, Self::pack_wall_bits(&self.vw));
    }

    /// Convert a dense wall slot (0..63) to the corresponding `Board` wall bit index.
    /// Board bits are indexed by `(row * 8 + col)` with row 0 at the top;
    /// Dense wall slots are `(7 - row) * 8 + col`.
    #[inline]
    pub fn ace_slot_to_board_bit(slot: usize) -> usize {
        (7 - slot / 8) * 8 + slot % 8
    }

    /// Convert ACE-slot-ordered wall bits to Board-ordered bits.
    #[inline]
    pub fn ace_wall_bits_to_board(bits: u64) -> u64 {
        let mut out = 0u64;
        let mut b = bits;
        while b != 0 {
            let slot = b.trailing_zeros() as usize;
            b &= b - 1;
            out |= 1u64 << Self::ace_slot_to_board_bit(slot);
        }
        out
    }

    /// Convert Board-ordered wall bits to ACE-slot-ordered bits.
    #[inline]
    pub fn board_wall_bits_to_ace(bits: u64) -> u64 {
        let mut out = 0u64;
        let mut b = bits;
        while b != 0 {
            let bit = b.trailing_zeros() as usize;
            b &= b - 1;
            let slot = Self::ace_slot_to_board_bit(bit); // inverse is the same reflection
            out |= 1u64 << slot;
        }
        out
    }

    pub fn set_wall_bits(&mut self, wall_type: usize, slot: usize, on: bool) {
        let r = slot / 8;
        let c = slot % 8;
        if wall_type == 0 {
            let a = r * 9 + c;
            let b = a + 1;
            let cc = a + 9;
            let dd = b + 9;
            if on {
                self.blocked[a] |= 2;
                self.blocked[b] |= 2;
                self.blocked[cc] |= 1;
                self.blocked[dd] |= 1;
            } else {
                self.blocked[a] &= !2;
                self.blocked[b] &= !2;
                self.blocked[cc] &= !1;
                self.blocked[dd] &= !1;
            }
        } else {
            let a = r * 9 + c;
            let b = a + 9;
            let cc = a + 1;
            let dd = b + 1;
            if on {
                self.blocked[a] |= 8;
                self.blocked[b] |= 8;
                self.blocked[cc] |= 4;
                self.blocked[dd] |= 4;
            } else {
                self.blocked[a] &= !8;
                self.blocked[b] &= !8;
                self.blocked[cc] &= !4;
                self.blocked[dd] &= !4;
            }
        }
    }

    pub fn wall_fits(&self, wall_type: usize, slot: usize) -> bool {
        let r = slot / 8;
        let c = slot % 8;
        if self.hw[slot] != 0 || self.vw[slot] != 0 {
            return false;
        }
        if wall_type == 0 {
            if c > 0 && self.hw[slot - 1] != 0 {
                return false;
            }
            if c < 7 && self.hw[slot + 1] != 0 {
                return false;
            }
        } else {
            if r > 0 && self.vw[slot - 8] != 0 {
                return false;
            }
            if r < 7 && self.vw[slot + 8] != 0 {
                return false;
            }
        }
        true
    }

    /// O(1) topo shift: skip BFS when the wall cannot touch enough topology to seal.
    pub fn wall_needs_path_check(&self, wall_type: usize, slot: usize) -> bool {
        let board_h = Self::ace_wall_bits_to_board(self.hw_bits);
        let board_v = Self::ace_wall_bits_to_board(self.vw_bits);
        crate::movegen::wall_masks::wall_slot_needs_flood(
            board_h,
            board_v,
            wall_type == 0,
            Self::ace_slot_to_board_bit(slot),
        )
    }

    pub fn has_path(&self, player: usize) -> bool {
        use crate::pathfinding::bff::flood_to_goal_with_depth;
        use crate::pathfinding::masks::DirMasks;

        let masks = DirMasks::from_ace_game(self);
        let goal = if player == 0 {
            ACE_P0_GOAL_BITS
        } else {
            ACE_P1_GOAL_BITS
        };
        flood_to_goal_with_depth(self.pawn[player] as u8, masks, goal).0
    }

    #[inline]
    fn both_paths_open_bitfill(&self) -> BitfillPathCheck {
        use crate::pathfinding::bff::{
            flood_component_with_goal_depth, flood_to_goal_seeded_with_depth,
        };
        use crate::pathfinding::masks::DirMasks;

        let masks = DirMasks::from_ace_game(self);
        let (ok0, p0_reached, p0_goal_depth) =
            flood_component_with_goal_depth(self.pawn[0] as u8, masks, ACE_P0_GOAL_BITS);
        if !ok0 {
            return BitfillPathCheck {
                open: false,
                p0_goal_depth,
                p1_seeded_proof_depth: 0,
            };
        }
        let (p1_open, p1_seeded_proof_depth) = flood_to_goal_seeded_with_depth(
            self.pawn[1] as u8,
            p0_reached,
            masks,
            ACE_P1_GOAL_BITS,
        );
        BitfillPathCheck {
            open: p1_open,
            p0_goal_depth,
            p1_seeded_proof_depth,
        }
    }

    pub fn wall_legal(&mut self, wall_type: usize, slot: usize) -> bool {
        if self.wl[self.turn] <= 0 {
            return false;
        }
        if !self.wall_fits(wall_type, slot) {
            return false;
        }
        if !self.wall_needs_path_check(wall_type, slot) {
            return true;
        }
        self.set_wall_bits(wall_type, slot, true);
        let checked = self.both_paths_open_bitfill();
        let _p0_goal_depth = checked.p0_goal_depth;
        let _p1_seeded_proof_depth = checked.p1_seeded_proof_depth;
        let ok = checked.open;
        self.set_wall_bits(wall_type, slot, false);
        ok
    }

    // ── Pawn moves ──────────────────────────────────────────────────────────

    pub fn gen_pawn_moves(&self, out: &mut [i16], mut n: usize) -> usize {
        let me = self.turn;
        let s = self.pawn[me];
        let o = self.pawn[1 - me];
        for d in 0..4 {
            if !self.can_step(s, d) {
                continue;
            }
            let t = (s as i16 + DELTA[d]) as usize;
            if t != o {
                out[n] = t as i16;
                n += 1;
                continue;
            }
            if self.can_step(o, d) {
                out[n] = o as i16 + DELTA[d];
                n += 1;
                continue;
            }
            let p1 = if d < 2 { 2 } else { 0 };
            let p2 = if d < 2 { 3 } else { 1 };
            if self.can_step(o, p1) {
                let w1 = (o as i16 + DELTA[p1]) as usize;
                if w1 != s {
                    out[n] = w1 as i16;
                    n += 1;
                }
            }
            if self.can_step(o, p2) {
                let w2 = (o as i16 + DELTA[p2]) as usize;
                if w2 != s {
                    out[n] = w2 as i16;
                    n += 1;
                }
            }
        }
        n
    }

    // ── Make / unmake (allocation-free) ─────────────────────────────────────

    pub fn make_move(&mut self, m: i16) {
        let z = &ZOBRIST;
        let hl = self.hist_len;
        self.hist_m[hl] = m;
        self.hist_lw[hl] = self.last_wall_ply as i16;
        if crate::titanium::is_pawn_move(m) {
            let p = self.turn;
            let to = m as usize;
            self.hist_from[hl] = self.pawn[p] as i16;
            self.hash_lo ^= z.pawn_lo[p][self.pawn[p]] ^ z.pawn_lo[p][to];
            self.hash_hi ^= z.pawn_hi[p][self.pawn[p]] ^ z.pawn_hi[p][to];
            self.pawn[p] = to;
        } else if crate::titanium::is_hwall_move(m) {
            let s0 = crate::titanium::wall_slot(m);
            self.hw[s0] = 1;
            self.hw_bits |= 1u64 << s0;
            self.set_wall_bits(0, s0, true);
            self.wl[self.turn] -= 1;
            self.wall_stamp += 1;
            self.hash_lo ^= z.hw_lo[s0];
            self.hash_hi ^= z.hw_hi[s0];
            self.last_wall_ply = hl + 1;
        } else {
            let s1 = crate::titanium::wall_slot(m);
            self.vw[s1] = 1;
            self.vw_bits |= 1u64 << s1;
            self.set_wall_bits(1, s1, true);
            self.wl[self.turn] -= 1;
            self.wall_stamp += 1;
            self.hash_lo ^= z.vw_lo[s1];
            self.hash_hi ^= z.vw_hi[s1];
            self.last_wall_ply = hl + 1;
        }
        self.turn ^= 1;
        self.hash_lo ^= z.turn_lo;
        self.hash_hi ^= z.turn_hi;
        self.hashes_u[hl * 2] = self.hash_lo;
        self.hashes_u[hl * 2 + 1] = self.hash_hi;
        self.hist_len = hl + 1;
        #[cfg(debug_assertions)]
        self.assert_wall_bits_sync();
    }

    pub fn unmake_move(&mut self) {
        let z = &ZOBRIST;
        self.hist_len -= 1;
        let hl = self.hist_len;
        let m = self.hist_m[hl];
        self.last_wall_ply = self.hist_lw[hl] as usize;
        self.turn ^= 1;
        self.hash_lo ^= z.turn_lo;
        self.hash_hi ^= z.turn_hi;
        if crate::titanium::is_pawn_move(m) {
            let p = self.turn;
            let from = self.hist_from[hl] as usize;
            let to = m as usize;
            self.hash_lo ^= z.pawn_lo[p][from] ^ z.pawn_lo[p][to];
            self.hash_hi ^= z.pawn_hi[p][from] ^ z.pawn_hi[p][to];
            self.pawn[p] = from;
        } else if crate::titanium::is_hwall_move(m) {
            let s0 = crate::titanium::wall_slot(m);
            self.hw[s0] = 0;
            self.hw_bits &= !(1u64 << s0);
            self.set_wall_bits(0, s0, false);
            self.wl[self.turn] += 1;
            self.wall_stamp -= 1;
            self.hash_lo ^= z.hw_lo[s0];
            self.hash_hi ^= z.hw_hi[s0];
        } else {
            let s1 = crate::titanium::wall_slot(m);
            self.vw[s1] = 0;
            self.vw_bits &= !(1u64 << s1);
            self.set_wall_bits(1, s1, false);
            self.wl[self.turn] += 1;
            self.wall_stamp -= 1;
            self.hash_lo ^= z.vw_lo[s1];
            self.hash_hi ^= z.vw_hi[s1];
        }
        #[cfg(debug_assertions)]
        self.assert_wall_bits_sync();
    }

    // ── Distance fields ─────────────────────────────────────────────────────

    pub fn compute_dist(&self, player: usize, dist: &mut [u8; 81]) {
        use crate::pathfinding::masks::DirMasks;
        use crate::titanium::dist::{
            fill_ace_dist_to_goal_with_masks_p0, fill_ace_dist_to_goal_with_masks_p1,
        };

        let masks = DirMasks::from_ace_game(self);
        if player == 0 {
            fill_ace_dist_to_goal_with_masks_p0(masks, dist);
        } else {
            fill_ace_dist_to_goal_with_masks_p1(masks, dist);
        }
    }

    /// Bitboard flood steps from `start` cell (255 = unreachable).
    pub fn compute_steps_from(&self, start: usize, dist: &mut [u8; 81]) {
        use crate::pathfinding::masks::DirMasks;
        use crate::titanium::dist::fill_ace_dist_from_pawn_with_masks;

        if start < 81 {
            fill_ace_dist_from_pawn_with_masks(start, DirMasks::from_ace_game(self), dist);
        } else {
            dist.fill(255);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::{Board, WallOrientation};
    use crate::movegen::legal::can_wall_block_topology;
    use crate::titanium::algebraic_to_move_id;

    fn pos(moves: &[&str]) -> GameState {
        let mut g = GameState::new();
        for m in moves {
            g.make_move(algebraic_to_move_id(m));
        }
        g
    }

    fn board_from_game(g: &GameState) -> Board {
        let mut b = Board::new();
        b.horizontal_walls = GameState::ace_wall_bits_to_board(g.hw_bits);
        b.vertical_walls = GameState::ace_wall_bits_to_board(g.vw_bits);
        b
    }

    #[test]
    fn wall_needs_path_check_matches_topo_oracle() {
        let positions = [
            pos(&[]),
            pos(&["e2", "e8", "e3", "e7", "d3h", "d6h", "f3h", "f6h"]),
            pos(&[
                "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e6h", "c3h", "c6h", "g3h", "g6h",
                "a3h", "e4v", "h3v",
            ]),
        ];
        for g in positions {
            let board = board_from_game(&g);
            for wall_type in [0usize, 1] {
                for slot in 0..64usize {
                    // Dense slot row is reflected relative to Board wall row.
                    let row = (7 - slot / 8) as u8;
                    let col = (slot % 8) as u8;
                    let horizontal = wall_type == 0;
                    let orientation = if horizontal {
                        WallOrientation::Horizontal
                    } else {
                        WallOrientation::Vertical
                    };
                    assert_eq!(
                        g.wall_needs_path_check(wall_type, slot),
                        can_wall_block_topology(&board, row, col, orientation),
                        "topo parity wall_type={wall_type} slot={slot}"
                    );
                }
            }
        }
    }

    #[test]
    fn topo_skip_implies_paths_stay_open() {
        let positions = [
            pos(&[]),
            pos(&["e2", "e8", "e3", "e7", "c3h", "c6h", "e4v"]),
        ];
        for base in positions {
            for wall_type in [0usize, 1] {
                for slot in 0..64usize {
                    let mut g = base.clone();
                    if !g.wall_fits(wall_type, slot) {
                        continue;
                    }
                    if g.wall_needs_path_check(wall_type, slot) {
                        continue;
                    }
                    g.set_wall_bits(wall_type, slot, true);
                    assert!(
                        g.has_path(0),
                        "p0 open after topo skip wt={wall_type} slot={slot}"
                    );
                    assert!(
                        g.has_path(1),
                        "p1 open after topo skip wt={wall_type} slot={slot}"
                    );
                    g.set_wall_bits(wall_type, slot, false);
                }
            }
        }
    }

    #[test]
    fn bitfill_wall_legal_preserves_path_invariants() {
        let positions = [
            pos(&[]),
            pos(&["e2", "e8", "e3", "e7", "d3h", "d6h", "f3h", "f6h"]),
            pos(&[
                "e2", "e8", "e3", "e7", "e4", "e6", "d3h", "d6h", "f3h", "f6h", "d5v", "h3v",
                "e4h", "h6h",
            ]),
            pos(&[
                "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e6h", "c3h", "c6h", "g3h", "g6h",
                "a3h", "e4v", "h3v",
            ]),
        ];

        for base in positions {
            for wall_type in [0usize, 1] {
                for slot in 0..64usize {
                    let mut g = base.clone();
                    if !g.wall_legal(wall_type, slot) {
                        continue;
                    }
                    g.set_wall_bits(wall_type, slot, true);
                    assert!(g.has_path(0), "p0 path wall_type={wall_type} slot={slot}");
                    assert!(g.has_path(1), "p1 path wall_type={wall_type} slot={slot}");
                    g.set_wall_bits(wall_type, slot, false);
                }
            }
        }
    }

    /// Exhaustive dense slot ↔ Board wall-bit mapping over all 128 placements.
    #[test]
    fn ace_board_wall_bit_mapping_exhaustive() {
        for wall_type in [0usize, 1] {
            for slot in 0..64usize {
                let board_bit = GameState::ace_slot_to_board_bit(slot);
                // Reflection is its own inverse.
                assert_eq!(
                    GameState::ace_slot_to_board_bit(board_bit),
                    slot,
                    "round-trip slot wt={wall_type} slot={slot} board_bit={board_bit}"
                );

                let ace_mask = 1u64 << slot;
                let board_mask = GameState::ace_wall_bits_to_board(ace_mask);
                assert_eq!(board_mask.count_ones(), 1);
                assert_eq!(
                    GameState::board_wall_bits_to_ace(board_mask),
                    ace_mask,
                    "bits round-trip wt={wall_type} slot={slot}"
                );
                assert_eq!(
                    board_mask.trailing_zeros() as usize,
                    board_bit,
                    "single-bit board index wt={wall_type} slot={slot}"
                );

                // make/unmake round-trip keeps packed ACE bits in sync.
                let mut g2 = GameState::new();
                if !g2.wall_fits(wall_type, slot) {
                    continue;
                }
                let mid = if wall_type == 0 {
                    crate::titanium::MOVE_HW_BASE + slot as i16
                } else {
                    crate::titanium::MOVE_VW_BASE + slot as i16
                };
                g2.make_move(mid);
                let packed = if wall_type == 0 {
                    g2.hw_bits
                } else {
                    g2.vw_bits
                };
                assert_eq!(
                    packed & ace_mask,
                    ace_mask,
                    "make wt={wall_type} slot={slot}"
                );
                g2.unmake_move();
                let packed2 = if wall_type == 0 {
                    g2.hw_bits
                } else {
                    g2.vw_bits
                };
                assert_eq!(packed2 & ace_mask, 0, "unmake wt={wall_type} slot={slot}");

                // Topo skip on empty board matches scraped oracle (Board row/col).
                let row = (7 - slot / 8) as u8;
                let col = (slot % 8) as u8;
                let orientation = if wall_type == 0 {
                    WallOrientation::Horizontal
                } else {
                    WallOrientation::Vertical
                };
                let board = Board::new();
                let needs = GameState::new().wall_needs_path_check(wall_type, slot);
                let oracle = can_wall_block_topology(&board, row, col, orientation);
                assert_eq!(needs, oracle, "topo wt={wall_type} slot={slot}");
            }
        }

        // Corner rows/cols: reflection preserves column; row flips 0↔7.
        for (label, slot) in [
            ("top-left", 0usize),
            ("top-right", 7),
            ("bottom-left", 56),
            ("bottom-right", 63),
        ] {
            let bb = GameState::ace_slot_to_board_bit(slot);
            assert_eq!(bb % 8, slot % 8, "{label} col preserved");
            assert_eq!(bb / 8 + slot / 8, 7, "{label} row reflection");
        }
    }
}
