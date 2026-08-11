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
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

/// Result from the side-to-move's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TbResult {
    Loss,
    /// Neither side can force a win.
    ///
    /// NOT the threefold-repetition rule. This table is keyed on
    /// `(p0, p1, stm)` and carries no move history, so it could not count
    /// repetitions even in principle. What it holds is the game VALUE: under
    /// perfect play the position never resolves, and because the state space is
    /// finite that shows up over the board as a repetition. The rule is how the
    /// value becomes a result; the value is what a tablebase stores.
    ///
    /// This is the entry for mutual zugzwang, and it falls out of the retrograde
    /// rather than being special-cased: a state becomes a LOSS only once every
    /// child is a WIN, so one drawing child leaves it unresolved forever. When
    /// each side would lose by committing — run for the goal and the opponent's
    /// wall traps you, and the same is true in reverse — neither can be
    /// compelled to break first, and both shuffle.
    ///
    /// Illegal pawn states also end up here, since `TbEntry::default()` is a
    /// draw and the flood fill never visits them. A caller counting real draws
    /// must subtract the excluded states — see `tb_layers::live_state_count`.
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
        Self::build_core(template, &[])
    }

    /// Solve one tier.
    ///
    /// `wall_children` are the placements available to a side that still holds a
    /// wall, each carrying the already-solved table one tier down that it leads
    /// to. An empty slice is the `(0,0)` case, where no wall can be placed and
    /// the configuration is fixed for the whole solve.
    ///
    /// Tier 0 and tier 1 deliberately share this one path. Two solvers would be
    /// two chances to disagree, and the tier that is easy to verify is exactly
    /// the one that would not catch it.
    fn build_core(template: &GameState, wall_children: &[WallChild]) -> ZeroWallTb {
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
        let mut wall_kids = vec![0u32; NSTATES];
        let mut finalized = vec![false; NSTATES];

        // Events, popped in nondecreasing distance:
        //
        //   WIN_AT      this state is won at `d`
        //   CHILD_WON   one of this state's children is won at `d`, so the state
        //               is one step nearer to having every child won, which is
        //               what makes it lost
        //
        // A heap rather than a FIFO because wall placements seed events at
        // arbitrary distances taken from lower-tier tables, not just at 0.
        //
        // The two kinds can never race for the same state: WIN_AT means some
        // child is lost, CHILD_WON reaching zero means every child is won, and
        // those cannot both hold.
        const WIN_AT: u8 = 0;
        const CHILD_WON: u8 = 1;
        let mut queue: BinaryHeap<Reverse<(i16, u8, u32, i16)>> = BinaryHeap::new();

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
                        continue;
                    }
                    if p1 >= 72 {
                        tbl[idx] = TbEntry {
                            result: if stm == 1 { TbResult::Win } else { TbResult::Loss },
                            distance: 0,
                            best_move: -1,
                        };
                        finalized[idx] = true;
                        continue;
                    }
                    g.pawn[0] = p0;
                    g.pawn[1] = p1;
                    g.turn = stm;
                    let mut buf = [0i16; 160];
                    let n = g.gen_pawn_moves(&mut buf, 0);
                    moves[idx] = buf[..n].to_vec();

                    // Wall placements by the side to move. Their tables are one
                    // tier down and already solved, so these values are known
                    // before the walk starts and simply seed the queue.
                    //
                    // Legality is re-tested per state because `wall_legal` runs
                    // the path check against THESE pawns; only where a placement
                    // LANDS is pawn-independent, which is what lets the child
                    // tables be shared across all states of the configuration.
                    for wc in wall_children.iter().filter(|w| w.stm == stm) {
                        if !g.wall_legal(wc.wall_type, wc.slot) {
                            continue;
                        }
                        wall_kids[idx] += 1;
                        let e = wc.table.probe_raw(p0, p1, 1 - stm);
                        match e.result {
                            // The opponent is lost there, so we win here.
                            TbResult::Loss => queue.push(Reverse((
                                e.distance + 1,
                                WIN_AT,
                                idx as u32,
                                wc.move_id,
                            ))),
                            TbResult::Win => queue.push(Reverse((
                                e.distance,
                                CHILD_WON,
                                idx as u32,
                                wc.move_id,
                            ))),
                            // Counted in `remaining` and never decremented: a
                            // side that can step into a draw is never forced to
                            // lose.
                            TbResult::Draw => {}
                        }
                    }
                }
            }
        }

        // Invert the move relation. A destination is always in the same
        // component as the square moved from, so a move out of a legal square
        // can only land on another legal square and no child needs filtering.
        for idx in 0..NSTATES {
            if moves[idx].is_empty() && wall_kids[idx] == 0 {
                continue;
            }
            let stm = idx & 1;
            let p0 = idx / 2 / NCELLS;
            let p1 = (idx / 2) % NCELLS;
            // Wall placements are children too, so they count toward "every
            // child is won" even though they live in another table.
            remaining[idx] = wall_kids[idx];
            for &mv in &moves[idx] {
                let dest = mv as usize;
                let (c0, c1) = if stm == 0 { (dest, p1) } else { (p0, dest) };
                preds[tb_index(c0, c1, 1 - stm)].push(idx as u32);
                remaining[idx] += 1;
            }
        }

        // Seed the walk from the terminals, which all sit at distance 0.
        for idx in 0..NSTATES {
            if !finalized[idx] {
                continue;
            }
            let ct = idx & 1;
            let dest = if ct == 1 {
                (idx / 2 / NCELLS) as i16
            } else {
                ((idx / 2) % NCELLS) as i16
            };
            for &p in &preds[idx] {
                match tbl[idx].result {
                    TbResult::Loss => queue.push(Reverse((1, WIN_AT, p, dest))),
                    TbResult::Win => queue.push(Reverse((0, CHILD_WON, p, dest))),
                    TbResult::Draw => {}
                }
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
        while let Some(Reverse((d, kind, target, mv))) = queue.pop() {
            let idx = target as usize;
            if finalized[idx] {
                continue;
            }
            let (result, dist) = match kind {
                // Reached from a lost child. Popping in increasing distance
                // means this is the NEAREST such child, so `d` is already the
                // true distance and never needs revising.
                WIN_AT => (TbResult::Win, d),
                // One more child won. The state is lost only once every child
                // is, and because these arrive in increasing distance the one
                // that empties the counter is the longest resistance — so it is
                // both the right distance and the right move to record.
                _ => {
                    remaining[idx] -= 1;
                    if remaining[idx] != 0 {
                        continue;
                    }
                    (TbResult::Loss, d + 1)
                }
            };
            tbl[idx] = TbEntry {
                result,
                distance: dist,
                best_move: mv,
            };
            finalized[idx] = true;

            // Propagate to the pawn-move predecessors inside this table. Wall
            // placements need no propagation: they only ever point DOWN a tier,
            // and that tier was solved before this one started.
            let ct = idx & 1;
            let dest = if ct == 1 {
                (idx / 2 / NCELLS) as i16
            } else {
                ((idx / 2) % NCELLS) as i16
            };
            for pi in 0..preds[idx].len() {
                let p = preds[idx][pi];
                if finalized[p as usize] {
                    continue;
                }
                match result {
                    TbResult::Loss => queue.push(Reverse((dist + 1, WIN_AT, p, dest))),
                    TbResult::Win => queue.push(Reverse((dist, CHILD_WON, p, dest))),
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
    /// v3 packs an entry into TWO bytes instead of five.
    ///
    ///   byte 0  i8 score: `+(k+1)` win in k plies, `-(k+1)` loss in k, `0` draw
    ///   byte 1  u8 move index, `NO_MOVE` when there is none
    ///
    /// The sign carries the result, so `result` stops being a stored field.
    ///
    /// WHY THE `k+1` OFFSET, given a terminal is decidable from the coordinates
    /// and could be rebuilt on decode instead. Because `0` would then carry
    /// THREE meanings, not two: draw, terminal, and ILLEGAL. Terminals are
    /// recoverable from `p0 < 9` / `p1 >= 72`; illegal states are not, because
    /// deciding those needs the wall configuration and the decoder does not have
    /// it. Rebuilding terminals unconditionally therefore relabels every
    /// excluded pawn state as a win -- exactly the fabricated-label bug the
    /// flood fill exists to prevent, reintroduced through the encoding.
    ///
    /// With the offset, `0` means one thing: not decisive. The cost is a ceiling
    /// of 126 rather than 127 plies, against a measured maximum of 35.
    ///
    /// Move ids fit a byte already: pawn destinations are 0..80 and wall moves
    /// 81..208, the same pseudo-legal index the rest of the engine uses.
    ///
    /// v2 (five bytes) is still READ so packs solved before this change stay
    /// usable -- tier 3 costs 19 minutes a configuration and throwing that away
    /// to change a file format would be its own bug.
    pub const VERSION: u16 = 3;
    pub const HEADER_LEN: usize = 16;
    pub const RECORD_LEN: usize = 2;
    /// Sentinel in the move byte for "no move". Real ids never reach it.
    pub const NO_MOVE: u8 = 0xFF;

    /// Record width of a given format version, so a pack can carry tables
    /// written before the width changed.
    pub fn record_len_for(version: u16) -> Option<usize> {
        match version {
            2 => Some(5),
            3 => Some(2),
            _ => None,
        }
    }

    /// FNV-1a over the record bytes. Used to version the table and to detect a
    /// corrupt or stale file at load time.
    pub fn content_hash(&self) -> u64 {
        self.try_content_hash().unwrap_or(0)
    }

    fn try_content_hash(&self) -> Result<u64, String> {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for e in &self.tbl {
            for b in Self::encode_entry(e)? {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        Ok(h)
    }

    /// Pack an entry into two bytes. Returns `Err` rather than truncating: a
    /// distance past the range would otherwise be written as a DIFFERENT, valid
    /// looking label, which is the one failure mode a training set cannot
    /// survive.
    fn encode_entry(e: &TbEntry) -> Result<[u8; 2], String> {
        let score: i16 = match e.result {
            TbResult::Draw => 0,
            TbResult::Win => e.distance + 1,
            TbResult::Loss => -(e.distance + 1),
        };
        if !(-127..=127).contains(&score) {
            return Err(format!(
                "distance {} does not fit a signed byte (score {score});                  the two-byte label format assumes forced wins stay under 127 plies",
                e.distance
            ));
        }
        let mv = if e.best_move < 0 || e.best_move > 254 {
            Self::NO_MOVE
        } else {
            e.best_move as u8
        };
        Ok([score as i8 as u8, mv])
    }

    fn decode_entry(b: &[u8]) -> Result<TbEntry, String> {
        let score = b[0] as i8 as i16;
        let mv = if b[1] == Self::NO_MOVE { -1 } else { b[1] as i16 };
        Ok(match score {
            0 => TbEntry { result: TbResult::Draw, distance: -1, best_move: mv },
            s if s > 0 => TbEntry { result: TbResult::Win, distance: s - 1, best_move: mv },
            s => TbEntry { result: TbResult::Loss, distance: -s - 1, best_move: mv },
        })
    }

    /// v2's five-byte record, kept so older packs still load.
    fn decode_entry_v2(b: &[u8]) -> Result<TbEntry, String> {
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
        self.try_to_bytes().expect("table does not fit the two-byte label format")
    }

    /// Serialise, refusing rather than truncating if an entry will not fit.
    pub fn try_to_bytes(&self) -> Result<Vec<u8>, String> {
        let mut out = Vec::with_capacity(Self::HEADER_LEN + NSTATES * Self::RECORD_LEN);
        out.extend_from_slice(Self::MAGIC);
        out.extend_from_slice(&Self::VERSION.to_le_bytes());
        out.extend_from_slice(&(self.live as u16).to_le_bytes());
        out.extend_from_slice(&self.try_content_hash()?.to_le_bytes());
        for e in &self.tbl {
            out.extend_from_slice(&Self::encode_entry(e)?);
        }
        Ok(out)
    }

    /// Parse a table produced by [`to_bytes`], rejecting a wrong magic,
    /// version, length, or content hash.
    pub fn from_bytes(bytes: &[u8]) -> Result<ZeroWallTb, String> {
        if bytes.len() < Self::HEADER_LEN {
            return Err("shorter than its header".into());
        }
        if &bytes[0..4] != Self::MAGIC {
            return Err("bad magic (not a TBZW file)".into());
        }
        let ver = u16::from_le_bytes([bytes[4], bytes[5]]);
        let rec = Self::record_len_for(ver)
            .ok_or_else(|| format!("version {ver} != supported 2 or 3"))?;
        let want = Self::HEADER_LEN + NSTATES * rec;
        if bytes.len() != want {
            return Err(format!("length {} != expected {want} for v{ver}", bytes.len()));
        }
        let live = u16::from_le_bytes([bytes[6], bytes[7]]) as usize;
        let stored_hash = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let mut tbl = Vec::with_capacity(NSTATES);
        for i in 0..NSTATES {
            let off = Self::HEADER_LEN + i * rec;
            let e = &bytes[off..off + rec];
            tbl.push(if ver == 2 {
                Self::decode_entry_v2(e)?
            } else {
                Self::decode_entry(e)?
            });
        }
        let tb = ZeroWallTb { tbl, live };
        // The hash is over v3 records, so a v2 file's stored hash is over a
        // different encoding and cannot be compared. Its integrity was checked
        // when it was written; re-checking here would reject every old pack.
        if ver == Self::VERSION && tb.content_hash() != stored_hash {
            return Err("content hash mismatch (file is corrupt or stale)".into());
        }
        Ok(tb)
    }

    /// Largest |distance| over decisive states, and how many exceed an i8.
    ///
    /// Decides whether a packed label can carry `+k`/`-k` in a single signed
    /// byte. Measured rather than assumed: if even one state overflows, the
    /// compact format silently corrupts that label instead of failing.
    pub fn distance_extremes(&self) -> (i16, usize) {
        let mut max = 0i16;
        let mut over = 0usize;
        for e in &self.tbl {
            if e.result == TbResult::Draw {
                continue;
            }
            if e.distance > max {
                max = e.distance;
            }
            if e.distance > 127 {
                over += 1;
            }
        }
        (max, over)
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

// ── Tier 1: one wall still in hand ──────────────────────────────────────────

/// A wall placement available in a tier, with the already-solved table it leads
/// to. Built once per configuration and shared across every pawn state, because
/// where a placement LANDS depends only on the walls.
struct WallChild {
    stm: usize,
    wall_type: usize,
    slot: usize,
    move_id: i16,
    table: Arc<ZeroWallTb>,
}

/// Cache key: the wall configuration plus the hand PAIR.
///
/// Keyed by the pair and never the total. `(2,0)` and `(1,1)` are different
/// games — the configuration is the same board either way, but who holds the
/// wall decides who may place it, so the solutions differ.
pub type TbKey = (u64, u64, i32, i32);

/// Solve a tier on demand, memoising the lower tiers it rests on.
///
/// Play only ever moves DOWN a tier: walls in hand cannot increase, so placing
/// the last wall lands in `(0,0)` and stays there. Taking walls back off a board
/// is our enumeration device for finding configurations, never a move.
///
/// The memo is what makes this finite. A held wall may be placed in ANY legal
/// slot, not back where it came from, so a 19-wall position's children are ~109
/// DIFFERENT 20-wall boards — only one of which is the board it was peeled from.
/// Peeling gives the positions to solve; it does not give the positions needed
/// to solve them.
pub struct TbSolver {
    cache: HashMap<TbKey, Arc<ZeroWallTb>>,
    hits: u64,
    misses: u64,
}

impl Default for TbSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl TbSolver {
    pub fn new() -> TbSolver {
        TbSolver {
            cache: HashMap::new(),
            hits: 0,
            misses: 0,
        }
    }

    pub fn key(g: &GameState) -> TbKey {
        (g.hw_bits, g.vw_bits, g.wl[0], g.wl[1])
    }

    /// Solve the tier `template` sits in, recursing into the tiers below.
    ///
    /// Terminates because every recursive call spends a wall, and `(0,0)` has
    /// none left to spend.
    pub fn solve(&mut self, template: &GameState) -> Arc<ZeroWallTb> {
        let key = Self::key(template);
        if let Some(t) = self.cache.get(&key) {
            self.hits += 1;
            return Arc::clone(t);
        }
        self.misses += 1;
        let built = Arc::new(self.build(template));
        self.cache.insert(key, Arc::clone(&built));
        built
    }

    fn build(&mut self, template: &GameState) -> ZeroWallTb {
        if template.wl[0] <= 0 && template.wl[1] <= 0 {
            return ZeroWallTb::build_for(template);
        }
        let children = self.children_of(template);
        ZeroWallTb::build_core(template, &children)
    }

    /// The wall placements available in `template`'s tier, each with its solved
    /// child table.
    fn children_of(&mut self, template: &GameState) -> Vec<WallChild> {
        let mut children = Vec::new();
        for stm in 0..2 {
            if template.wl[stm] <= 0 {
                continue;
            }
            for wall_type in 0..2 {
                for slot in 0..64 {
                    // `wall_fits` is the pawn-INDEPENDENT half of legality:
                    // overlap and crossing. The path check is the pawn-dependent
                    // half and is re-run per state in `build_core`, so this
                    // over-generates rather than under-. That is the safe
                    // direction — a child table no state can reach costs memory,
                    // a missing one would be a wrong answer.
                    if !template.wall_fits(wall_type, slot) {
                        continue;
                    }
                    let move_id = if wall_type == 0 {
                        crate::titanium::MOVE_HW_BASE
                    } else {
                        crate::titanium::MOVE_VW_BASE
                    } + slot as i16;
                    // Made as a real move rather than by editing bits, so the
                    // child's walls, hands and turn are what the engine would
                    // produce.
                    let mut c = template.clone();
                    c.turn = stm;
                    c.make_move(move_id);
                    let table = self.solve(&c);
                    children.push(WallChild {
                        stm,
                        wall_type,
                        slot,
                        move_id,
                        table,
                    });
                }
            }
        }
        children
    }

    /// Verify a solved tier by LOCAL CONSISTENCY: every state's verdict and
    /// distance must follow from its children's.
    ///
    /// This is a proof, not a spot check. If every state is locally consistent
    /// and the tier below is correct, the tier is correct by induction — and
    /// `(0,0)` is independently verified against
    /// `hands_empty_race_stm_wins_oracle`, so the induction has a base.
    ///
    /// It is also the only check available up here. `exact_dp` solves the
    /// hands-empty class only, so no external oracle exists for a tier holding a
    /// wall, and a plain negamax cannot be one either — the pawn-shuffle cycles
    /// are exactly what the retrograde exists to resolve.
    pub fn certify(&mut self, template: &GameState) -> Result<(), String> {
        let table = self.solve(template);
        let children = self.children_of(template);
        let reach = [
            crate::titanium::endgame::tb_layers::goal_reachable(template, 0),
            crate::titanium::endgame::tb_layers::goal_reachable(template, 1),
        ];
        let mut g = template.clone();

        for p0 in 9..NCELLS {
            if !reach[0][p0] {
                continue;
            }
            for p1 in 0..72 {
                if p0 == p1 || !reach[1][p1] {
                    continue;
                }
                for stm in 0..2 {
                    let got = table.probe_raw(p0, p1, stm);
                    g.pawn[0] = p0;
                    g.pawn[1] = p1;
                    g.turn = stm;

                    let mut vals: Vec<TbEntry> = Vec::with_capacity(16);
                    let mut buf = [0i16; 160];
                    let n = g.gen_pawn_moves(&mut buf, 0);
                    for &mv in &buf[..n] {
                        let dest = mv as usize;
                        let (c0, c1) = if stm == 0 { (dest, p1) } else { (p0, dest) };
                        vals.push(table.probe_raw(c0, c1, 1 - stm));
                    }
                    for wc in children.iter().filter(|w| w.stm == stm) {
                        if g.wall_legal(wc.wall_type, wc.slot) {
                            vals.push(wc.table.probe_raw(p0, p1, 1 - stm));
                        }
                    }
                    if vals.is_empty() {
                        continue;
                    }

                    let mut nearest_loss: Option<i16> = None;
                    let mut furthest_win: Option<i16> = None;
                    let mut all_win = true;
                    for e in &vals {
                        match e.result {
                            TbResult::Loss => {
                                if nearest_loss.map_or(true, |b| e.distance < b) {
                                    nearest_loss = Some(e.distance);
                                }
                            }
                            TbResult::Win => {
                                if furthest_win.map_or(true, |b| e.distance > b) {
                                    furthest_win = Some(e.distance);
                                }
                            }
                            TbResult::Draw => all_win = false,
                        }
                    }

                    let (want, want_d) = if let Some(d) = nearest_loss {
                        (TbResult::Win, d + 1)
                    } else if all_win {
                        (TbResult::Loss, furthest_win.unwrap_or(-2) + 1)
                    } else {
                        (TbResult::Draw, -1)
                    };

                    if got.result != want {
                        return Err(format!(
                            "(p0={p0}, p1={p1}, stm={stm}): table {:?}, children imply {want:?}",
                            got.result
                        ));
                    }
                    if want != TbResult::Draw && got.distance != want_d {
                        return Err(format!(
                            "(p0={p0}, p1={p1}, stm={stm}) {want:?}: distance {} but children imply {want_d}",
                            got.distance
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    // ── Pack file ───────────────────────────────────────────────────────────
    //
    // Every solved table, keyed by configuration and hands, in one file:
    //
    //   0..4    magic  b"TBPK"
    //   4..6    version u16 LE
    //   6..8    reserved
    //   8..16   record count u64 LE
    //   then, per record:
    //     hw u64 LE, vw u64 LE, wl0 i8, wl1 i8, then a whole TBZW table
    //
    // One file rather than a file per configuration: at millions of tables,
    // 65 KB each, a directory tree is what makes the store unusable on Windows.
    //
    // Why persist at all: without it each tier re-solves every tier beneath it
    // from scratch, so the ladder cannot be climbed one step at a time. With it,
    // tier 3 loads tier 2 and only pays for its own level.

    pub const PACK_MAGIC: &'static [u8; 4] = b"TBPK";
    pub const PACK_VERSION: u16 = 1;

    /// Write every held table to `path`.
    pub fn save(&self, path: &str) -> std::io::Result<usize> {
        use std::io::Write;
        let f = std::fs::File::create(path)?;
        let mut w = std::io::BufWriter::new(f);
        w.write_all(Self::PACK_MAGIC)?;
        w.write_all(&Self::PACK_VERSION.to_le_bytes())?;
        w.write_all(&[0u8, 0])?;
        w.write_all(&(self.cache.len() as u64).to_le_bytes())?;
        // Sorted, so the same cache always produces a byte-identical file and
        // two runs can be diffed when they disagree.
        let mut keys: Vec<&TbKey> = self.cache.keys().collect();
        keys.sort_unstable();
        for k in &keys {
            w.write_all(&k.0.to_le_bytes())?;
            w.write_all(&k.1.to_le_bytes())?;
            w.write_all(&[k.2 as u8, k.3 as u8])?;
            w.write_all(&self.cache[k].to_bytes())?;
        }
        w.flush()?;
        Ok(keys.len())
    }

    /// Load tables from `path` into the cache, returning how many were added.
    ///
    /// Existing entries win: anything already solved in this process is at least
    /// as trustworthy as what is on disk, and silently replacing it would make a
    /// stale file able to override a fresh solve.
    pub fn load(&mut self, path: &str) -> std::io::Result<usize> {
        let bytes = std::fs::read(path)?;
        let bad = |m: String| std::io::Error::new(std::io::ErrorKind::InvalidData, m);
        if bytes.len() < 16 {
            return Err(bad("pack shorter than its header".into()));
        }
        if &bytes[0..4] != Self::PACK_MAGIC {
            return Err(bad("bad magic (not a TBPK pack)".into()));
        }
        let ver = u16::from_le_bytes([bytes[4], bytes[5]]);
        if ver != Self::PACK_VERSION {
            return Err(bad(format!("pack version {ver} != {}", Self::PACK_VERSION)));
        }
        let count = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        // Table width comes from the FIRST embedded table's own version, not
        // from the current constant. A pack written before the two-byte format
        // holds five-byte tables, and computing the stride from today's value
        // would walk the file at the wrong pitch and reject it as corrupt.
        let table_ver = if count == 0 {
            ZeroWallTb::VERSION
        } else {
            u16::from_le_bytes([bytes[16 + 18 + 4], bytes[16 + 18 + 5]])
        };
        let entry_len = ZeroWallTb::record_len_for(table_ver)
            .ok_or_else(|| bad(format!("pack holds unsupported table version {table_ver}")))?;
        let table_len = ZeroWallTb::HEADER_LEN + NSTATES * entry_len;
        let rec_len = 18 + table_len;
        let want = 16 + count * rec_len;
        if bytes.len() != want {
            return Err(bad(format!("pack length {} != expected {want}", bytes.len())));
        }
        let mut added = 0usize;
        for i in 0..count {
            let off = 16 + i * rec_len;
            let hw = u64::from_le_bytes(bytes[off..off + 8].try_into().unwrap());
            let vw = u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap());
            let key = (hw, vw, bytes[off + 16] as i8 as i32, bytes[off + 17] as i8 as i32);
            if self.cache.contains_key(&key) {
                continue;
            }
            // `from_bytes` re-checks the content hash, so a truncated or edited
            // pack is rejected here rather than answering wrongly later.
            let tb = ZeroWallTb::from_bytes(&bytes[off + 18..off + 18 + table_len])
                .map_err(|e| bad(format!("record {i}: {e}")))?;
            self.cache.insert(key, Arc::new(tb));
            added += 1;
        }
        Ok(added)
    }

    /// Distinct configurations solved and held.
    pub fn cached(&self) -> usize {
        self.cache.len()
    }

    /// Share of `solve` calls answered from cache. This is the number that
    /// decides how far the ladder can climb — memory, not time, binds first.
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }

    /// Drop every held table. The tiers are solved bottom-up, so once a tier is
    /// finished the ones beneath it are dead weight.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    pub fn bytes(&self) -> usize {
        self.cache.len() * NSTATES * std::mem::size_of::<TbEntry>()
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

    /// Tier 1: one wall in hand, solved on top of the `(0,0)` tables its
    /// placements reach.
    ///
    /// Uses a REAL layer-1 configuration — a 20-wall board with one wall peeled
    /// back off — so conservation holds (board + both hands == 20) and the
    /// position is one a game could actually be in.
    #[test]
    fn tier_one_is_locally_consistent() {
        use crate::titanium::endgame::tb_layers;
        let seed = tb_layers::seed_boards(1, 20260811)[0];
        let layer1 = tb_layers::expand(&[seed], 1);
        // Whoever is owed the peeled wall is what makes (1,0) and (0,1)
        // different games on the same board, so check both.
        for hands in [[1, 0], [0, 1]] {
            let config = tb_layers::pick(&layer1[1], 0).expect("layer 1 is empty");
            let g = tb_layers::state_from_config(config, hands);
            assert_eq!(
                tb_layers::wall_count(config) as i32 + g.wl[0] + g.wl[1],
                20,
                "wall conservation"
            );
            let mut s = TbSolver::new();
            if let Err(e) = s.certify(&g) {
                panic!("tier-1 {hands:?} is not locally consistent: {e}");
            }
            println!(
                "tier-1 {hands:?}: {} configs solved, hit rate {:.1}%, {:.1} MB",
                s.cached(),
                s.hit_rate() * 100.0,
                s.bytes() as f64 / 1e6
            );
        }
    }

    /// A wall in hand can only help the side holding it: giving a player a wall
    /// must never turn a position they had won into one they lose.
    ///
    /// Independent of local consistency — that check would pass just as happily
    /// on a table that mixed the two players up, which is the mistake the
    /// pair-keying exists to prevent.
    #[test]
    fn holding_a_wall_never_hurts() {
        use crate::titanium::endgame::tb_layers;
        let seed = tb_layers::seed_boards(1, 555)[0];
        let config =
            tb_layers::pick(&tb_layers::expand(&[seed], 1)[1], 0).expect("layer 1 is empty");

        let mut s = TbSolver::new();
        // Same board, same pawns: once with the wall in hand, once after it is
        // spent. `(0,0)` on this 19-wall board is the same race with no wall to
        // place, so any position won there must still be won holding one.
        let with_wall = s.solve(&tb_layers::state_from_config(config, [1, 0]));
        let without = s.solve(&tb_layers::state_from_config(config, [0, 0]));
        let reach = [
            tb_layers::goal_reachable(&tb_layers::state_from_config(config, [0, 0]), 0),
            tb_layers::goal_reachable(&tb_layers::state_from_config(config, [0, 0]), 1),
        ];

        let mut checked = 0usize;
        for p0 in 9..81 {
            if !reach[0][p0] {
                continue;
            }
            for p1 in 0..72 {
                if p0 == p1 || !reach[1][p1] {
                    continue;
                }
                // Player 0 to move, holding the wall.
                let a = with_wall.probe_raw(p0, p1, 0);
                let b = without.probe_raw(p0, p1, 0);
                if b.result == TbResult::Win {
                    assert_ne!(
                        a.result,
                        TbResult::Loss,
                        "p0={p0} p1={p1}: won without a wall but lost while holding one"
                    );
                }
                checked += 1;
            }
        }
        assert!(checked > 500, "too few states compared: {checked}");
    }

    /// (2,0) and (0,2) are the same board and DIFFERENT games, so their tables
    /// must differ somewhere.
    ///
    /// This is the guard against the symmetric collapse: keying a table on the
    /// TOTAL walls in hand rather than the pair looks harmless — the board is
    /// identical either way — and silently answers "player 0 holds both walls"
    /// with the solution to "player 1 holds both". Their aggregate win/loss
    /// counts happen to come out equal, so a census check would not catch it;
    /// only a per-state comparison does.
    #[test]
    fn who_holds_the_wall_changes_the_answer() {
        use crate::titanium::endgame::tb_layers;
        let seed = tb_layers::seed_boards(1, 0x5eed)[0];
        let config =
            tb_layers::pick(&tb_layers::expand(&[seed], 2)[2], 0).expect("layer 2 is empty");
        let reach = [
            tb_layers::goal_reachable(&tb_layers::state_from_config(config, [0, 0]), 0),
            tb_layers::goal_reachable(&tb_layers::state_from_config(config, [0, 0]), 1),
        ];

        let mut s = TbSolver::new();
        let a = s.solve(&tb_layers::state_from_config(config, [2, 0]));
        let b = s.solve(&tb_layers::state_from_config(config, [0, 2]));

        let mut differ = 0usize;
        let mut compared = 0usize;
        for p0 in 9..81 {
            if !reach[0][p0] {
                continue;
            }
            for p1 in 0..72 {
                if p0 == p1 || !reach[1][p1] {
                    continue;
                }
                for stm in 0..2 {
                    compared += 1;
                    if a.probe_raw(p0, p1, stm).result != b.probe_raw(p0, p1, stm).result {
                        differ += 1;
                    }
                }
            }
        }
        println!("(2,0) vs (0,2): {differ}/{compared} states differ");
        assert!(
            differ > 0,
            "(2,0) and (0,2) are identical on all {compared} states — \
             the solver is keying on the wall TOTAL, not on who holds them"
        );
    }

    /// A saved pack must reload to tables identical to the ones solved, on a
    /// configuration where the flood fill actually excluded squares.
    ///
    /// That last part is the point: format v1 reconstructed `live` as
    /// `81 * 80 * 2` instead of storing it, so a pruned table came back claiming
    /// states it does not cover. A bare board would not have caught it.
    #[test]
    fn pack_roundtrips_a_pruned_table() {
        use crate::titanium::endgame::tb_layers;
        let stranding = tb_layers::seed_boards(40, 4242)
            .into_iter()
            .find(|&c| tb_layers::live_state_count(c) < 81 * 80 * 2)
            .expect("no stranding board among the seeds");

        let mut s = TbSolver::new();
        let original = s.solve(&tb_layers::state_from_config(stranding, [1, 0]));
        assert!(
            original.live_states() < 81 * 80 * 2,
            "test needs a table with excluded squares"
        );

        let path = std::env::temp_dir().join("titanium_tb_roundtrip.tbpk");
        let path = path.to_str().unwrap();
        let saved = s.save(path).expect("save failed");

        let mut back = TbSolver::new();
        let loaded = back.load(path).expect("load failed");
        assert_eq!(loaded, saved, "every saved table must reload");

        let again = back.solve(&tb_layers::state_from_config(stranding, [1, 0]));
        assert_eq!(again.live_states(), original.live_states(), "live count");
        for p0 in 0..81 {
            for p1 in 0..81 {
                if p0 == p1 {
                    continue;
                }
                for stm in 0..2 {
                    let a = original.probe_raw(p0, p1, stm);
                    let b = again.probe_raw(p0, p1, stm);
                    assert_eq!(a.result, b.result, "result at ({p0},{p1},{stm})");
                    assert_eq!(a.distance, b.distance, "distance at ({p0},{p1},{stm})");
                    assert_eq!(a.best_move, b.best_move, "best move at ({p0},{p1},{stm})");
                }
            }
        }
        let _ = std::fs::remove_file(path);
    }

    /// How long can a forced win actually take, in PLIES?
    ///
    /// This sizes the packed label format. `+k`/`-k` in a single signed byte
    /// caps at 127, and `distance` counts plies with the sides alternating, so a
    /// 64-move march is already 128 and overflows. Walls can build a serpentine
    /// corridor, so the bound is not obviously small — measured, not assumed,
    /// because an overflow would silently corrupt the label rather than fail.
    #[test]
    fn measure_max_distance_for_label_packing() {
        use crate::titanium::endgame::tb_layers;
        let mut worst = 0i16;
        let mut worst_where = (0u64, 0u64);
        let mut over_i8 = 0usize;

        // Tier 0 is the pure race: no wall can be spent to change the tempo, so
        // the longest forced marches live here.
        for &c in &tb_layers::seed_boards(60, 20260811) {
            let t = ZeroWallTb::build_for(&tb_layers::state_from_config(c, [0, 0]));
            let (max, over) = t.distance_extremes();
            over_i8 += over;
            if max > worst {
                worst = max;
                worst_where = c;
            }
        }
        // And a sample with walls still in hand, where a placement can extend
        // the defender's resistance.
        for &c in &tb_layers::seed_boards(8, 424242) {
            for hands in [[1, 0], [0, 1], [1, 1]] {
                let t = ZeroWallTb::build_for(&tb_layers::state_from_config(c, hands));
                let (max, over) = t.distance_extremes();
                over_i8 += over;
                if max > worst {
                    worst = max;
                    worst_where = c;
                }
            }
        }

        println!("max forced distance: {worst} plies (config {worst_where:?})");
        println!("states exceeding i8 (+/-127): {over_i8}");
        println!("  -> i8 label {} ", if over_i8 == 0 { "FITS" } else { "OVERFLOWS" });
        assert!(worst > 0, "no decisive states measured at all");
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
