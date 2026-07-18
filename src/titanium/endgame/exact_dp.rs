//! Exact hands-empty endgame DP — Rust port of v14.1 `solver_core.js`.
//!
//! **Ownership:** `titanium/endgame` (architecture v1.0).
//! **Users:** validation, tests, benches — **not** production Search.
//! **Why Search must not call this:** exponential reference solver; would steal
//! clock and is not a production proof path. Search uses race/certify instead.
//!
//! When BOTH players are out of walls (hands-empty), the walls on the board are
//! frozen and the game collapses to a pure pawn race. This solver retrograde-
//! solves ALL 81×81×2 = 13,122 `(p0, p1, turn)` states for that fixed wall
//! configuration, level by level, giving the EXACT game-theoretic verdict with
//! exact ply counts:
//!
//!   `+k` — side to move wins in `k` plies
//!   `-k` — side to move loses in `k` plies
//!    `0` — draw (forced repetition / no winning resource; matches the engine's
//!          repetition = score-0 rule)
//!
//! Unlike the v13 `certify_win` certificate (sound but INCOMPLETE — it returns
//! "not proven" on positions its restricted subgame can't crack), this oracle is
//! sound AND COMPLETE for the hands-empty class: every jump / body-block / tempo
//! interaction is played out exactly, so the jump-blind `dMe<=dOpp` failure mode
//! cannot occur. This is the "guaranteed solved endgame" piece v14.1 added on
//! top of the (v13-identical) certificate.
//!
//! Semantics are reachability-game values: "win" = mathematically forced to
//! reach goal first under exact turn alternation; infinite play = draw. Each
//! built table is optionally re-verified by [`oracle_certify`] (local
//! consistency of every state proves the whole table).

use crate::titanium::game::GameState;
use std::collections::HashMap;

/// Number of `(p0, p1, turn)` states: 81 × 81 × 2.
pub const ORACLE_STATES: usize = 81 * 81 * 2;
/// Max successors per state (≤5 pawn moves with jumps) + padding.
const SUCC_STRIDE: usize = 6;

#[inline]
fn state_id(p0: usize, p1: usize, t: usize) -> usize {
    (p0 * 81 + p1) * 2 + t
}

/// Retrograde-solve every live `(p0, p1, turn)` state on the board described by
/// `blocked` (wall-edge bitmask per cell). Returns the value table (`ORACLE_STATES`
/// entries; `+k`/`-k`/`0`).
pub fn oracle_solve_board(blocked: &[u8; 81]) -> Vec<i16> {
    let mut og = GameState::new();
    og.blocked = *blocked;

    let mut v = vec![0i16; ORACLE_STATES];
    let mut succ_flat = vec![-1i8; ORACLE_STATES * SUCC_STRIDE];
    let mut succ_cnt = vec![0u8; ORACLE_STATES];
    let mut buf = [0i16; 16];

    // Precompute successor (pawn-move) lists for every queryable state.
    for p0 in 0..81usize {
        for p1 in 0..81usize {
            for t in 0..2usize {
                let id = state_id(p0, p1, t);
                // illegal/terminal: same square, or a pawn already on its goal.
                if p0 == p1 || p0 < 9 || p1 >= 72 {
                    succ_cnt[id] = 0;
                    continue;
                }
                og.pawn[0] = p0;
                og.pawn[1] = p1;
                og.turn = t;
                let cnt = og.gen_pawn_moves(&mut buf, 0);
                succ_cnt[id] = cnt as u8;
                for i in 0..cnt {
                    succ_flat[id * SUCC_STRIDE + i] = buf[i] as i8;
                }
            }
        }
    }

    // Level-by-level retrograde fixpoint: pass `k` assigns states with |V| == k.
    for k in 1..=4000i16 {
        let mut changed = false;
        for p0 in 9..81usize {
            for p1 in 0..72usize {
                if p0 == p1 {
                    continue;
                }
                for t in 0..2usize {
                    let id = state_id(p0, p1, t);
                    if v[id] != 0 {
                        continue;
                    }
                    let cnt = succ_cnt[id] as usize;
                    if cnt == 0 {
                        continue; // stuck pawn: draw (non-win resource)
                    }
                    let base = id * SUCC_STRIDE;
                    let mut has_win = false;
                    let mut all_opp_win = true;
                    let mut max_opp = 0i16;
                    for i in 0..cnt {
                        let to = succ_flat[base + i] as usize;
                        if (t == 0 && to < 9) || (t == 1 && to >= 72) {
                            if k == 1 {
                                has_win = true;
                                break;
                            }
                            all_opp_win = false;
                            continue;
                        }
                        let nid = if t == 0 {
                            state_id(to, p1, 1)
                        } else {
                            state_id(p0, to, 0)
                        };
                        let vv = v[nid];
                        if vv < 0 && -vv == k - 1 {
                            has_win = true; // opp loses in k-1 ⇒ we win in k
                        }
                        if vv <= 0 {
                            all_opp_win = false;
                        } else if vv > max_opp {
                            max_opp = vv;
                        }
                    }
                    if has_win {
                        v[id] = k;
                        changed = true;
                    } else if all_opp_win && max_opp == k - 1 {
                        v[id] = -k;
                        changed = true;
                    }
                }
            }
        }
        if !changed {
            break;
        }
    }
    v
}

/// Verify local consistency of every live state — a passing certificate proves
/// the whole table. Returns `Err(reason)` on the first inconsistency.
pub fn oracle_certify(blocked: &[u8; 81], v: &[i16]) -> Result<(), String> {
    let mut og = GameState::new();
    og.blocked = *blocked;
    let mut buf = [0i16; 16];

    for p0 in 9..81usize {
        for p1 in 0..72usize {
            if p0 == p1 {
                continue;
            }
            for t in 0..2usize {
                let id = state_id(p0, p1, t);
                og.pawn[0] = p0;
                og.pawn[1] = p1;
                og.turn = t;
                let cnt = og.gen_pawn_moves(&mut buf, 0);
                let val = v[id];
                let mut goal = false;
                let mut win = false;
                let mut all_opp_win = true;
                let mut max_opp = 0i16;
                let mut has_loss_succ = false;
                let mut has_draw_succ = false;
                for i in 0..cnt {
                    let to = buf[i] as usize;
                    if (t == 0 && to < 9) || (t == 1 && to >= 72) {
                        goal = true;
                        continue;
                    }
                    let nid = if t == 0 {
                        state_id(to, p1, 1)
                    } else {
                        state_id(p0, to, 0)
                    };
                    let vv = v[nid];
                    if val > 1 && vv == -(val - 1) {
                        win = true;
                    }
                    if vv <= 0 {
                        all_opp_win = false;
                        if vv == 0 {
                            has_draw_succ = true;
                        } else {
                            has_loss_succ = true;
                        }
                    } else if vv > max_opp {
                        max_opp = vv;
                    }
                }
                if val == 1 {
                    if !goal {
                        return Err(format!("state {id}: +1 without goal move"));
                    }
                } else if val > 1 {
                    if !win {
                        return Err(format!("state {id}: +{val} without succ -{}", val - 1));
                    }
                } else if val < 0 {
                    if goal {
                        return Err(format!("state {id}: {val} but has goal move"));
                    }
                    if !all_opp_win || cnt == 0 {
                        return Err(format!(
                            "state {id}: {val} but a successor is not an opp-win"
                        ));
                    }
                    if max_opp != -val - 1 {
                        return Err(format!("state {id}: {val} but max succ is {max_opp}"));
                    }
                } else {
                    // draw
                    if goal {
                        return Err(format!("state {id}: 0 but has goal move"));
                    }
                    if has_loss_succ {
                        return Err(format!("state {id}: 0 but has winning move"));
                    }
                    if cnt > 0 && !has_draw_succ {
                        return Err(format!(
                            "state {id}: 0 with all succ opp-wins (should be loss)"
                        ));
                    }
                }
            }
        }
    }
    Ok(())
}

/// Pack a 64-slot wall array into a `u64` (one bit per slot) for cache keying.
fn pack_walls(arr: &[u8; 64]) -> u64 {
    let mut bits = 0u64;
    for (s, &on) in arr.iter().enumerate() {
        if on != 0 {
            bits |= 1u64 << s;
        }
    }
    bits
}

/// Per-wall-config solved tables, memoized (cleared wholesale on overflow —
/// rebuilds are cheap and distinct hands-empty wall configs per search are few).
pub struct Oracle {
    cap: usize,
    map: HashMap<(u64, u64), Vec<i16>>,
    /// Re-verify (certify) every Nth distinct build; 0 = never.
    certify_every: u64,
    pub builds: u64,
    pub hits: u64,
    pub certified: u64,
}

impl Default for Oracle {
    fn default() -> Self {
        Self::new(96, 64)
    }
}

impl Oracle {
    pub fn new(cap: usize, certify_every: u64) -> Self {
        Self {
            cap,
            map: HashMap::new(),
            certify_every,
            builds: 0,
            hits: 0,
            certified: 0,
        }
    }

    /// Solved table for `g`'s wall config (built + cached on first sight).
    /// Panics if a freshly built table fails its consistency certificate —
    /// that would mean the solver itself is wrong, never a runtime input error.
    fn table(&mut self, g: &GameState) -> &Vec<i16> {
        let key = (pack_walls(&g.hw), pack_walls(&g.vw));
        if self.map.contains_key(&key) {
            self.hits += 1;
            return self.map.get(&key).unwrap();
        }
        let table = oracle_solve_board(&g.blocked);
        self.builds += 1;
        if self.certify_every > 0 && self.builds % self.certify_every == 1 {
            if let Err(e) = oracle_certify(&g.blocked, &table) {
                panic!("ORACLE CERTIFICATE FAIL: {e}");
            }
            self.certified += 1;
        }
        if self.map.len() >= self.cap {
            self.map.clear();
        }
        self.map.entry(key).or_insert(table)
    }

    /// Exact verdict for the side to move at `g` (hands MUST be empty):
    /// `+k`/`-k`/`0`.
    pub fn query(&mut self, g: &GameState) -> i16 {
        let id = state_id(g.pawn[0], g.pawn[1], g.turn);
        self.table(g)[id]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::{Board, Player};
    use crate::titanium::cert_bridge::titanium_game_from_board;

    #[test]
    fn empty_board_table_is_self_consistent() {
        let blocked = [0u8; 81];
        let v = oracle_solve_board(&blocked);
        assert_eq!(v.len(), ORACLE_STATES);
        assert_eq!(oracle_certify(&blocked, &v), Ok(()));
    }

    #[test]
    fn mate_in_one_is_exact() {
        // One (7,4) is one step from goal row 8, One to move, hands empty ⇒
        // exact verdict +1 (win in one ply). (NB: startpos itself is a SAME-
        // column race, where the jump-tempo applies, so it is NOT a clean
        // tempo win — the oracle correctly declines to call it +.)
        let mut board = Board::new();
        board.pawns = [(7, 4), (1, 0)];
        board.walls_remaining = [0, 0];
        board.hash = crate::core::zobrist::hash_board(&board);
        let g = titanium_game_from_board(&board);
        let mut oracle = Oracle::default();
        assert_eq!(oracle.query(&g), 1, "mate in 1 must be exact +1");
    }

    #[test]
    fn same_column_jump_tempo_is_not_a_stm_win() {
        // THE edge case: One (3,4) and Two (5,4) SAME column, equal distance 5,
        // One to move, hands empty. The jump-blind `dMe<=dOpp` rule says One
        // wins — but Two's jump over One steals a tempo. The exact oracle must
        // NOT report a win for One (it is a loss or draw).
        let mut board = Board::new();
        board.pawns = [(3, 4), (5, 4)];
        board.walls_remaining = [0, 0];
        board.hash = crate::core::zobrist::hash_board(&board);
        let g = titanium_game_from_board(&board);
        let mut oracle = Oracle::default();
        let verdict = oracle.query(&g);
        assert!(
            verdict <= 0,
            "jump-aware oracle must not falsely call this a win for One, got {verdict}"
        );
        let _ = Player::One;
    }
}
