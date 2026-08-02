//! Zero-wall tablebase: exact retrograde solve of the subgame where neither
//! player has walls left and none are on the board, so the only actions are
//! pawn steps and jumps.
//!
//! The live state space is `(p0, p1, stm)` with `p0 != p1` — at most
//! `81 * 80 * 2 = 12_960` states — small enough to solve completely and embed.
//!
//! Why this exists: `cert_bridge::hands_empty_race_stm_wins` already answers
//! the same question exactly, but it *computes* the answer per query (a tempo
//! classifier, then a `race_minimax` fallback on the volatile overlapping
//! band). This table answers in O(1) and additionally yields distance-to-mate
//! and the optimal move, which the classifier cannot provide.
//!
//! Move generation deliberately reuses [`GameState::gen_pawn_moves`] rather
//! than reimplementing the jump rules, so the table can never disagree with
//! the engine's own movegen.

use crate::titanium::position::game::GameState;
use std::sync::OnceLock;

/// Result from the side-to-move's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TbResult {
    Loss,
    Draw,
    Win,
}

#[derive(Debug, Clone, Copy)]
pub struct TbEntry {
    pub result: TbResult,
    /// Plies to the result for Win/Loss; `-1` for Draw.
    pub distance: i16,
    /// Optimal pawn destination for Win/Loss; `-1` when there is none.
    pub best_move: i16,
}

impl Default for TbEntry {
    fn default() -> Self {
        TbEntry {
            result: TbResult::Draw,
            distance: -1,
            best_move: -1,
        }
    }
}

const NCELLS: usize = 81;
const NSTATES: usize = NCELLS * NCELLS * 2;

#[inline]
pub fn tb_index(p0: usize, p1: usize, stm: usize) -> usize {
    (p0 * NCELLS + p1) * 2 + stm
}

/// A position is in the zero-wall subgame when no walls are placed and neither
/// side has any left to place.
#[inline]
pub fn applies(g: &GameState) -> bool {
    g.hw_bits == 0 && g.vw_bits == 0 && g.wl[0] == 0 && g.wl[1] == 0
}

pub struct ZeroWallTb {
    tbl: Vec<TbEntry>,
    live: usize,
}

impl ZeroWallTb {
    /// Solve the whole subgame by fixpoint iteration.
    ///
    /// Backward induction with cycles: a state is a WIN as soon as one child is
    /// a LOSS, and a LOSS only once *every* child is a WIN. States still
    /// unresolved at the fixpoint are draws by repetition — that is the correct
    /// verdict here, since neither side can be compelled to make progress.
    pub fn build() -> ZeroWallTb {
        let mut tbl = vec![TbEntry::default(); NSTATES];
        let mut live = 0usize;

        // Scratch game used only to enumerate pawn moves. No walls are ever
        // placed on it, so `blocked` stays border-only for the whole build.
        let mut g = GameState::new();

        // Precompute the move list for every live (p0, p1, stm).
        let mut moves: Vec<Vec<i16>> = vec![Vec::new(); NSTATES];
        let mut terminal = vec![false; NSTATES];

        for p0 in 0..NCELLS {
            for p1 in 0..NCELLS {
                if p0 == p1 {
                    continue;
                }
                for stm in 0..2 {
                    let idx = tb_index(p0, p1, stm);
                    live += 1;
                    // Terminal: a pawn already stands on its goal row.
                    if p0 < 9 {
                        terminal[idx] = true;
                        tbl[idx] = TbEntry {
                            result: if stm == 0 { TbResult::Win } else { TbResult::Loss },
                            distance: 0,
                            best_move: -1,
                        };
                        continue;
                    }
                    if p1 >= 72 {
                        terminal[idx] = true;
                        tbl[idx] = TbEntry {
                            result: if stm == 1 { TbResult::Win } else { TbResult::Loss },
                            distance: 0,
                            best_move: -1,
                        };
                        continue;
                    }
                    g.pawn[0] = p0;
                    g.pawn[1] = p1;
                    g.turn = stm;
                    let mut buf = [0i16; 160];
                    let n = g.gen_pawn_moves(&mut buf, 0);
                    moves[idx] = buf[..n].to_vec();
                }
            }
        }

        // Fixpoint. Each pass can only resolve states whose verdict follows
        // from already-resolved children, so it terminates.
        let mut solved = vec![false; NSTATES];
        for (i, t) in terminal.iter().enumerate() {
            solved[i] = *t;
        }

        loop {
            let mut changed = false;
            for p0 in 0..NCELLS {
                for p1 in 0..NCELLS {
                    if p0 == p1 {
                        continue;
                    }
                    for stm in 0..2 {
                        let idx = tb_index(p0, p1, stm);
                        if solved[idx] || moves[idx].is_empty() {
                            continue;
                        }
                        let mut best_win: Option<(i16, i16)> = None; // (dist, move)
                        let mut worst_loss: Option<(i16, i16)> = None;
                        let mut all_children_win = true;

                        for &mv in &moves[idx] {
                            let dest = mv as usize;
                            let (c0, c1) = if stm == 0 { (dest, p1) } else { (p0, dest) };
                            let cidx = tb_index(c0, c1, 1 - stm);
                            if !solved[cidx] {
                                all_children_win = false;
                                continue;
                            }
                            match tbl[cidx].result {
                                // Child loses for the mover there => we win here.
                                TbResult::Loss => {
                                    let d = tbl[cidx].distance + 1;
                                    if best_win.map_or(true, |(bd, _)| d < bd) {
                                        best_win = Some((d, mv));
                                    }
                                }
                                TbResult::Win => {
                                    let d = tbl[cidx].distance + 1;
                                    if worst_loss.map_or(true, |(bd, _)| d > bd) {
                                        worst_loss = Some((d, mv));
                                    }
                                }
                                TbResult::Draw => {
                                    all_children_win = false;
                                }
                            }
                        }

                        if let Some((d, mv)) = best_win {
                            tbl[idx] = TbEntry {
                                result: TbResult::Win,
                                distance: d,
                                best_move: mv,
                            };
                            solved[idx] = true;
                            changed = true;
                        } else if all_children_win {
                            if let Some((d, mv)) = worst_loss {
                                tbl[idx] = TbEntry {
                                    result: TbResult::Loss,
                                    distance: d,
                                    best_move: mv,
                                };
                                solved[idx] = true;
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }

        ZeroWallTb { tbl, live }
    }

    #[inline]
    pub fn probe_raw(&self, p0: usize, p1: usize, stm: usize) -> TbEntry {
        self.tbl[tb_index(p0, p1, stm)]
    }

    /// Probe for a real position. Returns `None` unless [`applies`] holds.
    #[inline]
    pub fn probe(&self, g: &GameState) -> Option<TbEntry> {
        if !applies(g) {
            return None;
        }
        Some(self.probe_raw(g.pawn[0], g.pawn[1], g.turn))
    }

    // ── Serialization ───────────────────────────────────────────────────────
    //
    // Format `TBZW` v1: 16-byte header then one 5-byte record per state in
    // `tb_index` order (result u8, distance i16 LE, best_move i16 LE).
    //
    //   0..4   magic  b"TBZW"
    //   4..6   version u16 LE
    //   6..8   reserved
    //   8..16  content hash u64 LE (over the record bytes)

    pub const MAGIC: &'static [u8; 4] = b"TBZW";
    pub const VERSION: u16 = 1;
    pub const HEADER_LEN: usize = 16;
    pub const RECORD_LEN: usize = 5;

    /// FNV-1a over the record bytes. Used to version the table and to detect a
    /// corrupt or stale file at load time.
    pub fn content_hash(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for e in &self.tbl {
            for b in Self::encode_entry(e) {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    fn encode_entry(e: &TbEntry) -> [u8; 5] {
        let r = match e.result {
            TbResult::Loss => 0u8,
            TbResult::Draw => 1,
            TbResult::Win => 2,
        };
        let d = e.distance.to_le_bytes();
        let m = e.best_move.to_le_bytes();
        [r, d[0], d[1], m[0], m[1]]
    }

    fn decode_entry(b: &[u8]) -> Result<TbEntry, String> {
        let result = match b[0] {
            0 => TbResult::Loss,
            1 => TbResult::Draw,
            2 => TbResult::Win,
            other => return Err(format!("bad result byte {other}")),
        };
        Ok(TbEntry {
            result,
            distance: i16::from_le_bytes([b[1], b[2]]),
            best_move: i16::from_le_bytes([b[3], b[4]]),
        })
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(Self::HEADER_LEN + NSTATES * Self::RECORD_LEN);
        out.extend_from_slice(Self::MAGIC);
        out.extend_from_slice(&Self::VERSION.to_le_bytes());
        out.extend_from_slice(&[0u8, 0]);
        out.extend_from_slice(&self.content_hash().to_le_bytes());
        for e in &self.tbl {
            out.extend_from_slice(&Self::encode_entry(e));
        }
        out
    }

    /// Parse a table produced by [`to_bytes`], rejecting a wrong magic,
    /// version, length, or content hash.
    pub fn from_bytes(bytes: &[u8]) -> Result<ZeroWallTb, String> {
        let want = Self::HEADER_LEN + NSTATES * Self::RECORD_LEN;
        if bytes.len() != want {
            return Err(format!("length {} != expected {want}", bytes.len()));
        }
        if &bytes[0..4] != Self::MAGIC {
            return Err("bad magic (not a TBZW file)".into());
        }
        let ver = u16::from_le_bytes([bytes[4], bytes[5]]);
        if ver != Self::VERSION {
            return Err(format!("version {ver} != supported {}", Self::VERSION));
        }
        let stored_hash = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let mut tbl = Vec::with_capacity(NSTATES);
        for i in 0..NSTATES {
            let off = Self::HEADER_LEN + i * Self::RECORD_LEN;
            tbl.push(Self::decode_entry(&bytes[off..off + Self::RECORD_LEN])?);
        }
        let mut live = 0usize;
        for p0 in 0..NCELLS {
            for p1 in 0..NCELLS {
                if p0 != p1 {
                    live += 2;
                }
            }
        }
        let tb = ZeroWallTb { tbl, live };
        if tb.content_hash() != stored_hash {
            return Err("content hash mismatch (file is corrupt or stale)".into());
        }
        Ok(tb)
    }

    pub fn live_states(&self) -> usize {
        self.live
    }

    /// Count of states by verdict — used by the build test as a cheap
    /// regression signature.
    pub fn census(&self) -> (usize, usize, usize) {
        let (mut w, mut l, mut d) = (0, 0, 0);
        for p0 in 0..NCELLS {
            for p1 in 0..NCELLS {
                if p0 == p1 {
                    continue;
                }
                for stm in 0..2 {
                    match self.tbl[tb_index(p0, p1, stm)].result {
                        TbResult::Win => w += 1,
                        TbResult::Loss => l += 1,
                        TbResult::Draw => d += 1,
                    }
                }
            }
        }
        (w, l, d)
    }
}

static ZERO_WALL_TB: OnceLock<ZeroWallTb> = OnceLock::new();

/// Probe the process-wide exact zero-wall tablebase, building it once on first use.
#[inline]
pub fn probe_global(g: &GameState) -> Option<TbEntry> {
    ZERO_WALL_TB.get_or_init(ZeroWallTb::build).probe(g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::titanium::cert_bridge::hands_empty_race_stm_wins_oracle;

    fn tb() -> ZeroWallTb {
        ZeroWallTb::build()
    }

    #[test]
    fn build_covers_every_live_state_and_is_decisive() {
        let t = tb();
        assert_eq!(t.live_states(), 81 * 80 * 2, "live state count");
        let (w, l, d) = t.census();
        println!("zero-wall TB census: win={w} loss={l} draw={d}");
        assert_eq!(w + l + d, 81 * 80 * 2);
        // A wall-free race is always decisive: the trailing side can never
        // force repetition, so no state should be left unresolved.
        assert_eq!(d, 0, "unresolved (draw) states in a pure race");
    }

    /// The table must agree with the existing exact solver on EVERY zero-wall
    /// position. This is the soundness gate: `hands_empty_race_stm_wins` is
    /// already trusted by `certify`, so any disagreement is a real bug in one
    /// of the two and must block the change.
    #[test]
    fn tb_agrees_with_hands_empty_race_oracle_on_all_states() {
        let t = tb();
        let mut checked = 0usize;
        let mut mismatches = Vec::new();
        for p0 in 9..81 {
            for p1 in 0..72 {
                if p0 == p1 {
                    continue;
                }
                for stm in 0..2 {
                    let mut g = GameState::new();
                    g.pawn[0] = p0;
                    g.pawn[1] = p1;
                    g.turn = stm;
                    g.wl = [0, 0];
                    let Some(oracle_stm_wins) = hands_empty_race_stm_wins_oracle(&mut g) else {
                        continue;
                    };
                    let e = t.probe_raw(p0, p1, stm);
                    let tb_stm_wins = e.result == TbResult::Win;
                    checked += 1;
                    if tb_stm_wins != oracle_stm_wins {
                        if mismatches.len() < 20 {
                            mismatches.push((p0, p1, stm, tb_stm_wins, oracle_stm_wins, e.distance));
                        }
                    }
                }
            }
        }
        println!("checked {checked} zero-wall states, {} mismatches", mismatches.len());
        assert!(
            mismatches.is_empty(),
            "TB disagrees with hands_empty_race oracle on {} states, first: {:?}",
            mismatches.len(),
            &mismatches[..mismatches.len().min(10)]
        );
    }
}
