//! Zero-wall tablebase: exact retrograde solve of the pawn-only subgame.
//!
//! WRONG SCOPE -- READ BEFORE USING. This table models a BARE board: no walls
//! in hand AND none placed. That combination is physically impossible in a real
//! game. Each player starts with 10 walls; if both hands are empty then twenty
//! walls are on the board, so `hw_bits == 0` cannot hold. `applies()` is
//! therefore unsatisfiable in real play and `probe()` returns None for every
//! position that actually occurs.
//!
//! The real subgame worth solving is "neither player has walls LEFT", with
//! whatever walls are on the board treated as fixed scenery. That is where the
//! game becomes a pure race, and it is common.
//!
//! Do NOT "fix" this by relaxing `applies()` to drop the board-empty test. The
//! table is indexed by `(p0, p1, stm)` with no wall dimension and was built on
//! bare-board adjacency, so it would answer real positions with distances
//! computed for a board that is not the one being played -- confidently wrong
//! instead of silent. The fix is to index by wall configuration and solve one
//! table per configuration; the retrograde below is reusable as-is.
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
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, OnceLock};

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

/// The subgame is decided by the HANDS, asymmetrically: neither side has a wall
/// left to place. Whatever is on the board stays there and is pure scenery,
/// affecting adjacency only.
///
/// This deliberately does NOT test `hw_bits == 0 && vw_bits == 0`. Walls are
/// conserved -- board + wl[0] + wl[1] == 20 -- so requiring both hands empty
/// AND a bare board describes a position summing to 0 instead of 20, which
/// cannot occur. That is what this used to require, which is why the table
/// never fired and every broke endgame fell through to the per-query race
/// computation instead.
#[inline]
pub fn applies(g: &GameState) -> bool {
    g.wl[0] == 0 && g.wl[1] == 0
}

pub struct ZeroWallTb {
    tbl: Vec<TbEntry>,
    live: usize,
}

impl ZeroWallTb {
    /// Solve the whole subgame by retrograde analysis.
    ///
    /// Backward induction with cycles: a state is a WIN as soon as one child is
    /// a LOSS, and a LOSS only once *every* child is a WIN. States still
    /// unresolved at the end are draws by repetition — that is the correct
    /// verdict here, since neither side can be compelled to make progress.
    pub fn build() -> ZeroWallTb {
        Self::build_for(&GameState::new())
    }

    /// Solve the subgame for the wall configuration `template` is carrying.
    ///
    /// The template is cloned and only `pawn[0]`, `pawn[1]` and `turn` are
    /// varied, so the walls -- and every derived structure the engine keeps
    /// beside them -- are exactly those of the real position. Reconstructing
    /// the wall state by hand here would be a second implementation of the
    /// board and a second chance to disagree with movegen.
    ///
    /// Both hands are empty in this subgame, so no wall can ever be placed and
    /// the configuration is fixed for the whole solve.
    pub fn build_for(template: &GameState) -> ZeroWallTb {
        let mut tbl = vec![TbEntry::default(); NSTATES];
        let mut live = 0usize;

        // Scratch game used only to enumerate pawn moves. Walls come from the
        // template and never change: neither side has one left to place.
        let mut g = template.clone();

        // Which squares each pawn may legally stand on. Walls may legally seal
        // an empty pocket -- `wall_legal` only protects a path from where the
        // pawns STAND -- so a pawn dropped in one could never arrive. Left in,
        // such a state has no winning line, shuffles forever, and is filed as a
        // repetition draw: a confident exact label for a position no game can
        // reach. Roughly 7% of a walled board's states are this.
        let reach = [
            crate::titanium::endgame::tb_layers::goal_reachable(template, 0),
            crate::titanium::endgame::tb_layers::goal_reachable(template, 1),
        ];

        let mut moves: Vec<Vec<i16>> = vec![Vec::new(); NSTATES];
        let mut preds: Vec<Vec<u32>> = vec![Vec::new(); NSTATES];
        let mut remaining = vec![0u32; NSTATES];
        let mut finalized = vec![false; NSTATES];
        // Terminals all sit at distance 0 and every state entered afterwards is
        // one ply further out, so a FIFO queue visits states in nondecreasing
        // distance order and no bucket or heap is needed.
        let mut queue: VecDeque<usize> = VecDeque::new();

        for p0 in 0..NCELLS {
            if !reach[0][p0] {
                continue;
            }
            for p1 in 0..NCELLS {
                if p0 == p1 || !reach[1][p1] {
                    continue;
                }
                for stm in 0..2 {
                    let idx = tb_index(p0, p1, stm);
                    live += 1;
                    // Terminal: a pawn already stands on its goal row.
                    if p0 < 9 {
                        tbl[idx] = TbEntry {
                            result: if stm == 0 { TbResult::Win } else { TbResult::Loss },
                            distance: 0,
                            best_move: -1,
                        };
                        finalized[idx] = true;
                        queue.push_back(idx);
                        continue;
                    }
                    if p1 >= 72 {
                        tbl[idx] = TbEntry {
                            result: if stm == 1 { TbResult::Win } else { TbResult::Loss },
                            distance: 0,
                            best_move: -1,
                        };
                        finalized[idx] = true;
                        queue.push_back(idx);
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

        // Invert the move relation. A destination is always in the same
        // component as the square moved from, so a move out of a legal square
        // can only land on another legal square and no child needs filtering.
        for idx in 0..NSTATES {
            if moves[idx].is_empty() {
                continue;
            }
            let stm = idx & 1;
            let p0 = idx / 2 / NCELLS;
            let p1 = (idx / 2) % NCELLS;
            for &mv in &moves[idx] {
                let dest = mv as usize;
                let (c0, c1) = if stm == 0 { (dest, p1) } else { (p0, dest) };
                preds[tb_index(c0, c1, 1 - stm)].push(idx as u32);
                remaining[idx] += 1;
            }
        }

        // Retrograde analysis in distance order.
        //
        // This must NOT be a relaxation fixpoint. The obvious version marks a
        // state won the moment any one child is known lost and freezes that
        // distance -- but a later pass can resolve a different child as lost and
        // nearer, and nothing goes back to shorten it. That is silent: the
        // verdict stays right while the distance-to-mate reads long, which is
        // invisible to any test comparing only win/loss against an oracle, and
        // wrong as a training label. Processing in nondecreasing distance makes
        // the FIRST time a state is reached its true distance, so it is
        // finalised once and never revised.
        while let Some(idx) = queue.pop_front() {
            let (result, d) = (tbl[idx].result, tbl[idx].distance);
            // The move that led here, from the predecessor's point of view: the
            // side that moved is the one NOT to move in this state.
            let ct = idx & 1;
            let c0 = idx / 2 / NCELLS;
            let c1 = (idx / 2) % NCELLS;
            let dest = if ct == 1 { c0 as i16 } else { c1 as i16 };

            for pi in 0..preds[idx].len() {
                let p = preds[idx][pi] as usize;
                if finalized[p] {
                    continue;
                }
                match result {
                    // A child the opponent loses in makes this a win, and the
                    // first one found is the shortest.
                    TbResult::Loss => {
                        tbl[p] = TbEntry {
                            result: TbResult::Win,
                            distance: d + 1,
                            best_move: dest,
                        };
                        finalized[p] = true;
                        queue.push_back(p);
                    }
                    // One more child known won. Only when every child is won is
                    // the state lost, and the child that got it there last is
                    // the longest resistance -- so this is also the right move
                    // to record.
                    TbResult::Win => {
                        remaining[p] -= 1;
                        if remaining[p] == 0 {
                            tbl[p] = TbEntry {
                                result: TbResult::Loss,
                                distance: d + 1,
                                best_move: dest,
                            };
                            finalized[p] = true;
                            queue.push_back(p);
                        }
                    }
                    TbResult::Draw => {}
                }
            }
        }

        // Anything still unfinalised is a draw by repetition, which is the
        // `TbEntry` default. Illegal states keep that default too, so a caller
        // dumping labels must re-run the flood fill and skip them -- the table
        // deliberately does not carry a legality mask, since at millions of
        // tables the memory matters more than the convenience.
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

/// Solved tables by wall configuration `(hw_bits, vw_bits)`.
///
/// Once both hands are empty no wall can be placed, so a whole search sees a
/// single configuration and this holds one entry in practice. The cap only
/// guards pathological drivers that hop between unrelated positions.
static TB_BY_WALLS: OnceLock<Mutex<HashMap<(u64, u64), Arc<ZeroWallTb>>>> = OnceLock::new();
const TB_CACHE_CAP: usize = 8;

/// Table for this position's wall configuration, building it on first use.
pub fn table_for(g: &GameState) -> Arc<ZeroWallTb> {
    let key = (g.hw_bits, g.vw_bits);
    let cache = TB_BY_WALLS.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(tb) = cache.lock().expect("tb cache poisoned").get(&key) {
        return Arc::clone(tb);
    }
    // Built outside the lock: ~ms of work, and holding it would serialise every
    // Lazy SMP worker behind the first one to ask.
    let built = Arc::new(ZeroWallTb::build_for(g));
    let mut guard = cache.lock().expect("tb cache poisoned");
    if guard.len() >= TB_CACHE_CAP {
        guard.clear();
    }
    Arc::clone(guard.entry(key).or_insert(built))
}

/// Probe the process-wide exact zero-wall tablebase, building it once on first use.
#[inline]
pub fn probe_global(g: &GameState) -> Option<TbEntry> {
    if !applies(g) {
        return None;
    }
    if g.hw_bits == 0 && g.vw_bits == 0 {
        // Bare board: the shared table is already correct, skip the cache.
        return ZERO_WALL_TB.get_or_init(ZeroWallTb::build).probe(g);
    }
    Some(table_for(g).probe_raw(g.pawn[0], g.pawn[1], g.turn))
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
    /// The table now answers positions that have walls on them, so it must
    /// agree with the per-query oracle THERE, not just on a bare board. Before
    /// this, `applies()` demanded an empty board as well as empty hands -- a
    /// state that violates wall conservation and never occurs -- so nothing was
    /// ever answered and nothing was ever wrong. Now it can be wrong, so it is
    /// checked.
    #[test]
    fn walled_board_table_agrees_with_oracle() {
        // A handful of real wall layouts, applied through the engine rather
        // than by writing bits, so the board is one movegen actually produces.
        let layouts: [&[&str]; 3] = [
            &["e3h", "c5v", "f6h"],
            &["b2h", "d4v", "g7h", "e5v"],
            &["a1h", "c3h", "e5h", "g7h", "d2v", "f6v"],
        ];
        let mut total = 0usize;
        for walls in layouts {
            let mut base = GameState::new();
            for w in walls {
                base.make_move(crate::titanium::algebraic_to_move_id(w));
            }
            base.wl = [0, 0]; // broke: the subgame this table is for
            let t = ZeroWallTb::build_for(&base);
            let mut checked = 0usize;
            for p0 in (9..81).step_by(7) {
                for p1 in (0..72).step_by(7) {
                    if p0 == p1 {
                        continue;
                    }
                    for stm in 0..2 {
                        let mut g = base.clone();
                        g.pawn[0] = p0;
                        g.pawn[1] = p1;
                        g.turn = stm;
                        let Some(oracle) = hands_empty_race_stm_wins_oracle(&mut g) else {
                            continue;
                        };
                        let e = t.probe_raw(p0, p1, stm);
                        assert_eq!(
                            e.result == TbResult::Win,
                            oracle,
                            "walled tb disagrees with oracle: walls={walls:?}                              p0={p0} p1={p1} stm={stm} tb={:?}",
                            e.result
                        );
                        checked += 1;
                    }
                }
            }
            assert!(checked > 0, "no comparable states for {walls:?}");
            total += checked;
        }
        assert!(total > 100, "too few states compared: {total}");
    }

    /// Distance-to-mate must be locally consistent: a win is exactly one ply
    /// past its NEAREST losing child, a loss exactly one ply past its FURTHEST
    /// winning child.
    ///
    /// This is the check the suite lacked. Every other test here compares only
    /// win/loss against the oracle, so the old relaxation fixpoint could — and
    /// did — report a win at distance 5 whose children proved it at 3, and
    /// nothing failed. Verdicts are what search needs; distances are what a
    /// TRAINING LABEL needs, and a wrong one is taught as fact.
    #[test]
    fn distance_to_mate_is_exact() {
        let layouts: [&[&str]; 3] = [
            &["e3h", "c5v", "f6h"],
            &["b2h", "d4v", "g7h", "e5v"],
            &["a1h", "c3h", "e5h", "g7h", "d2v", "f6v"],
        ];
        for walls in layouts {
            let mut base = GameState::new();
            for w in walls {
                base.make_move(crate::titanium::algebraic_to_move_id(w));
            }
            base.wl = [0, 0];
            let t = ZeroWallTb::build_for(&base);
            let reach = [
                crate::titanium::endgame::tb_layers::goal_reachable(&base, 0),
                crate::titanium::endgame::tb_layers::goal_reachable(&base, 1),
            ];

            let mut g = base.clone();
            let mut checked = 0usize;
            for p0 in 9..81 {
                if !reach[0][p0] {
                    continue;
                }
                for p1 in 0..72 {
                    if p0 == p1 || !reach[1][p1] {
                        continue;
                    }
                    for stm in 0..2 {
                        let e = t.probe_raw(p0, p1, stm);
                        if e.result == TbResult::Draw {
                            continue;
                        }
                        g.pawn[0] = p0;
                        g.pawn[1] = p1;
                        g.turn = stm;
                        let mut buf = [0i16; 160];
                        let n = g.gen_pawn_moves(&mut buf, 0);

                        let mut nearest_loss: Option<i16> = None;
                        let mut furthest_win: Option<i16> = None;
                        for &mv in &buf[..n] {
                            let dest = mv as usize;
                            let (c0, c1) = if stm == 0 { (dest, p1) } else { (p0, dest) };
                            let c = t.probe_raw(c0, c1, 1 - stm);
                            match c.result {
                                TbResult::Loss => {
                                    if nearest_loss.map_or(true, |b| c.distance < b) {
                                        nearest_loss = Some(c.distance);
                                    }
                                }
                                TbResult::Win => {
                                    if furthest_win.map_or(true, |b| c.distance > b) {
                                        furthest_win = Some(c.distance);
                                    }
                                }
                                TbResult::Draw => {}
                            }
                        }

                        let want = match e.result {
                            TbResult::Win => nearest_loss.map(|d| d + 1),
                            TbResult::Loss => furthest_win.map(|d| d + 1),
                            TbResult::Draw => None,
                        };
                        if let Some(want) = want {
                            assert_eq!(
                                e.distance, want,
                                "walls={walls:?} p0={p0} p1={p1} stm={stm} {:?}: \
                                 distance {} but children imply {want}",
                                e.result, e.distance
                            );
                            checked += 1;
                        }
                    }
                }
            }
            assert!(checked > 500, "too few decisive states checked: {checked}");
        }
    }

    /// Stranded pawn squares must be left out of the table entirely.
    ///
    /// Without this the count silently returns to 12,960 the moment the flood
    /// fill stops being applied, and the fabricated repetition draws come back
    /// with it.
    #[test]
    fn unreachable_pawn_squares_are_excluded() {
        use crate::titanium::endgame::tb_layers;
        let stranding = tb_layers::seed_boards(40, 4242)
            .into_iter()
            .find(|&c| tb_layers::live_state_count(c) < 81 * 80 * 2)
            .expect("no stranding board among the seeds");
        let g = tb_layers::state_from_config(stranding, [0, 0]);
        let t = ZeroWallTb::build_for(&g);
        assert_eq!(
            t.live_states(),
            tb_layers::live_state_count(stranding),
            "table must cover exactly the legal pawn states"
        );
        assert!(
            t.live_states() < 81 * 80 * 2,
            "this board strands squares, so the table must be smaller than the full grid"
        );
    }

    #[test]
    fn applies_accepts_broke_hands_with_walls_on_board() {
        let mut g = GameState::new();
        for w in ["e3h", "c5v"] {
            g.make_move(crate::titanium::algebraic_to_move_id(w));
        }
        g.wl = [0, 0];
        assert!(g.hw_bits != 0 || g.vw_bits != 0, "test needs walls on board");
        assert!(
            applies(&g),
            "broke hands with walls on board is the subgame this table exists for"
        );
        g.wl = [1, 0];
        assert!(!applies(&g), "a wall still in hand leaves the subgame");
    }

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
