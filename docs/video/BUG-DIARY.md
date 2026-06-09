# Bug diary — plot twists for the video series

Chronological. Each entry: **symptom → cause → fix → lesson**.

---

## 1. Wall count zero at start (128 walls missing)

**Symptom:** `titanium moves` → 3 moves only (pawns). JS has 131.

**Cause:** Inverted `canWallBlock` logic. Scraped JS allows “floating” walls when topology check is false; we **rejected** them.

**Fix:** Only run path BFS when `can_wall_block_topology` is true; otherwise legal if no collision.

**Lesson:** Read the oracle, not your intuition about “useless” walls.

---

## 2. BFS panic on top row

**Symptom:** `path::tests::start_position_reachable` crashed in `has_vertical`.

**Cause:** Sideways step from row 8 used `js_row = 9` for wall lookup.

**Fix:** `has_horizontal` / `has_vertical` return false for out-of-range js_row (open border).

**Lesson:** Board edges are real edge cases — pawn grid is 9×9, wall slots are 8×8.

---

## 3. Perft depth 2 off by 2 nodes (THE first perft bug)

**Symptom:**

```
JS 16677  vs  Rust 16679
perft_diff → only d8v and e8v subtrees differ
```

**Cause:** Wrong vertical wall anchors for **lateral** `can_step`. After `d8v`, Black at `e9` could illegally step to `d9`.

**Fix:** Match scraped `pawnCanMove`:

- Right: check vertical at `from` and one row below `from`
- Left: check vertical at `to` and one row below `to`

**Regression:** `grid::tests::vertical_d8v_blocks_black_left_from_e9`, `perft_depth2_matches_js_oracle` = 16_677.

**Lesson:** Divide first. Two nodes at depth 2 = one pawn move wrong somewhere in the tree.

---

## 4. “Rust is fast… but perft 3 is already the ceiling” (design surprise)

**Symptom:** We picked Rust for speed. Perft 3 finishes in ~0.2s. Perft 4 looks hung.

**Cause:** Quoridor branching is ~**131** at the root, not ~20 like chess. Depth 3 = **2M nodes**. Depth 4 ≈ **100× more** ≈ hundreds of millions of clones + BFS wall checks. **Language speed does not save you** from exponential blow-up.

**What Rust actually bought us:**

- Depth **3** correctness check in a blink (JS oracles struggle at the same depth)
- Headroom for **search** at millions of nodes/sec — if we search _smart_

**What we still need (or perft/search past d3 is pointless):**

1. **Make/unmake** — stop cloning the whole board every node
2. **Tactical wall pruning** — search only walls that change shortest-path (perft keeps full legality)
3. **Zobrist TT** — reuse subtrees (gorisanson/pavlosdais both do this for _play_, not perft)
4. **Aspiration / ID** — don't re-walk the whole tree blind every ply

**Lesson for the video:** “Rust isn’t a cheat code — it’s a bigger engine bay. Past perft 3 you either get smart or you wait forever.”

---

## 5. “Perft 4 takes forever” (not a bug — same root as #4)

**Symptom:** `cargo run -- perft 4` runs minutes / appears hung.

**Cause:** ~2M nodes at depth 3 × ~100+ branching ≈ **hundreds of millions** of nodes. Naive clone-per-node perft.

**Mitigation (done):** make/unmake, Zobrist TT, in-place wall trials, `BfsScratch`, stack move buffer — see `PERFT-OPTIMIZATIONS.md`. Depth 4 now ~7s release.

**Lesson:** Quoridor ≠ chess perft tables. Depth 2 is unit test; depth 3 is correctness gate; depth 4 is stress test.

---

## 6. Shared move buffer panic in fast perft

**Symptom:** `index out of bounds: len is 0 but index is 3` in `perft_fast_ctx`.

**Cause:** `generate_legal_moves_into` clears `ctx.moves` on recursion; parent loop still indexing old length.

**Fix:** Snapshot moves per node — first `mem::take`, then stack `[Move; 140]` via `generate_legal_moves_slice`.

**Lesson:** Reused buffers need explicit ownership boundaries at recursion edges (Stockfish uses stack move lists per frame).

---

## 7. Gorisanson infinite spinner (coordinate bridge)

**Symptom:** Local MCTS never returns; board spinner runs forever.

**Cause:** `gorisansonBridge.js` used `row + 1` instead of flipping rows. UI row 1 = bottom; Gorisanson row 0 = top. Invalid moves → `applyAction` fails → `maybeRequestAiMove` loops.

**Fix:** `PAWN_ROWS - row` (9) for pawns, `WALL_ROWS - row` (8) for walls — same flip in `benchmark/lib/gorisanson_bridge.mjs`.

**Lesson:** Two “standard” coordinate systems on one board — always test one known pawn move (e2 from start) through the full bridge.

---

## 8. Remote engine red `!` after second move

**Symptom:** Ishtar/Ka show error state after human's second ply; WebSocket `log Error` or close.

**Cause:** We sent `makemove` only after **human** plies. Scraped app sends every `takeAction` to all engines — including the AI's own `bestmove`. Server was one ply behind → illegal position on next human move.

**Fix:** `syncRemoteEnginesAfterMove` after human moves **and** after remote AI `onBestMove`.

**Lesson:** Cloud engines are state machines; mirror the scraped sync contract, don't assume `bestmove` updates server memory.

---

## 9. Endgame sideways moves with 0 walls left

**Symptom:** After wall stock hits 0, MCTS sometimes steps sideways or backward despite a clear race to the goal — throws wins or delays losses.

**Cause:** Gorisanson MCTS still picks among pawn children by visit count; rollouts use heuristics that are not pure shortest-path when branching is low.

**Fix (v2):** No hard “walls = 0 → BFS skip.” `gorisanson_search_core.mjs` estimates branching each move from pawn moves + `ourWalls × openSlots × totalWallsLeft` (no wall enumeration). When `b^d` fits ~800k nodes at depth ≥ 8, switch to iterative minimax (BFS distance eval: `opp_dist - our_dist`). Otherwise MCTS.

**Lesson:** Branching is mostly wall inventory and board space left — count that, don’t enumerate. Minimax when the proxy says the tree is small enough.

---

## 10. `test_replay_legality` — external replay used pre-fix rules

**Symptom:** `test_replay::test_replay_legality` failed. Move 24 `g1v` was deemed illegal.

**Cause:** The hardcoded replay came from a game played before the horizontal boundary fix (commit `9e4cbf5`, `js_col == 8`). Under the corrected rules, `g1v` at move 24 blocks a goal path and is correctly rejected.

**Fix:** Marked old test `#[ignore]`. Added `g1v_correctly_rejected_after_replay_prefix` which asserts the wall is **not** in the legal move list.

**Lesson:** External replays can silently embed pre-fix illegal moves. Always isolate the specific move before marking a test wrong.

---

## 11. `expand_frontier_no_row_wrap_east_west` — wrong assertion

**Symptom:** Test asserted flood from `(0,0)` didn't reach `(1,0)` — but of course it does (south step is legal).

**Cause:** Test intent was to verify that an east shift from `(0,8)` (board east edge) doesn't bleed into `(1,0)`. Assertion was testing the wrong thing.

**Fix:** Refactored test to explicitly place a pawn at `(0,8)`, shift east, verify the resulting bit is in the padding column and `flood_sq_from_bit` returns `None` for it.

**Lesson:** "No row wrap" tests must isolate the specific edge square, not test reachability from a corner.

---

## 12. `flood_bit_index` — `const fn` disallows `u32::from(u8)`

**Symptom:** Compile error: `cannot call conditionally-const associated function <u32 as From<u8>>::from`.

**Cause:** `From::from` is not stabilised as `const fn` in Rust stable. `u32::from(row)` inside a `const fn` is rejected.

**Fix:** `row as u32` and `col as u32` casts — explicit type casts are allowed in `const fn`.

**Lesson:** In `const fn` contexts, use `as` casts, not trait-based conversions.

---

## 13. Known-path wall skip — eager cache caused regression

**Symptom:** After adding `WallPathCache`, perft 4 jumped from ~3.0s to ~4.27s.

**Cause:** `WallPathCache::new` was called for every wall candidate because the topology check returned `true` for many floaters. Building both shortest paths (two BFS passes) per position wiped out the BFS-skip savings.

**Fix:** Made `WallPathCache` lazy via `Option<WallPathCache>` with `get_or_insert_with`. Now it is built at most once per position and only when `can_wall_block_topology` has already returned `true` for at least one wall. All subsequent topology-true walls reuse the same cache.

**Lesson:** "Build once per position, not per candidate" only works if initialisation is deferred until the first candidate actually needs it.

---

## 14. Phantom gap mouths on board edges

**Symptom:** Walls near funnel/gap positions incorrectly pruned or kept; `gap_mouth_keeps_t_junction` test expected non-empty gap zone at a T-junction with no actual sealed territory.

**Cause:** `corridor_mouth_mask` and `gap_play_zone_mask` used neighbor bounds `0..=9` instead of `0..=8`. Row/col 9 is off the 9×9 board — every bottom/right edge square looked adjacent to "sealed" void outside the grid.

**Fix:** Neighbor iteration `0..=8` only. Test updated: no sealed pocket → `gap_zone == 0`.

**Lesson:** Off-by-one on board edges creates phantom topology. Gap logic is for **real** sealed components, not map borders.

---

## 15. Root plays passive wall at equal score (funnel `f4v` vs `e4`)

**Symptom:** White behind in race; search picks tempo-wasting wall (`gain: -10`) tied at same score as `e4` (`gain: +1`).

**Cause:** Root tie-break preferred wall over pawn when behind at equal score; `ROOT_TIEBREAK_BAND` (15 cm) allowed moves with **lower** scores to replace best if resistance was higher.

**Fix:** Exact ties break on `move_immediate_gain` first, then resistance. Band removed. Belt-and-suspenders only promotes strictly higher scores.

**Lesson:** When 80+ root moves tie on eval, **move choice** semantics matter as much as search depth. Path gain is the Quoridor analog of "is this a capture?"

---

## 16. Timeout fake fail-high (negamax + iterative deepening)

**Symptom:** `bestmove` disagreed with best `root_moves` entry after 3s search; last-searched wall had optimistic LMR score.

**Cause:** On time stop, `negamax` returned `alpha` from child; parent negated it into a window artifact. Partial depth was committed as if complete.

**Fix:** `should_stop()` checked **before** consuming child score. Aborted ID iteration discarded — keep previous completed depth's move. `committed_root_moves` snapshot per finished depth.

**Lesson:** Sebastian's iterative deepening rule: timer hits zero mid-depth-6 → play depth-5 move. Never play a half-searched tree.

---

## 17. Mate extensions never fired (`mateExtensions == 0`)

**Symptom:** Log always showed `mateExtensions: 0` even with forcing lines.

**Cause:** `search_child` clamped unproven mate to static eval **before** the extension loop — mate distance erased, loop never extended.

**Fix:** Run extension loop first; clamp only after extensions exhausted.

**Lesson:** Order of horizon handling matters. Clamp is a safety rail, not a preprocessor.

---

## 18. Search 10k nps vs perft 18M nps (not a perft bug)

**Symptom:** "We used to get depth 4 in 3.3s" — perft 4 still ~3.4s / 247M nodes. Search stuck at depth 2–3 on funnel.

**Cause:** Search nodes paid full legal movegen + BFS inside eval (`pawn_mobility`), triple opponent-path rebuild, CAT built twice, per-wall BFS in ordering, forcing extension at `dist ≤ 2`, `eval_stm` every child for mate clamp, qdepth 10 with all walls in qsearch.

**Fix:** Pawn-only mobility; witness-path gate in ordering; shared path/CAT per node; qsearch noisy-only walls; lazy eval_stm; extension threshold tightened; qdepth 6 / width cap 8.

**Lesson:** Perft green ≠ search fast. Profile the **search node**, not the movegen oracle.

---

## 19. Quiescence treated all pruned walls as noisy

**Symptom:** Horizon lines explored passive walls deep in qsearch — eval saw temporary path delay, missed opponent bypass.

**Cause:** `collect_search_moves(..., tactical_only: true)` only filtered pawn moves; walls still used full `wall_should_search` list.

**Fix:** Qsearch wall is noisy iff it lengthens opponent shortest path (witness gate + BFS gain). Quiet → stand pat. Empty noisy set → return static eval.

**Lesson:** Chess quiescence = captures. Quoridor quiescence = **path-length shocks**, not "still has walls in hand."

---

## 20. CAT COLD threshold in `wall_should_search` exploded tree

**Symptom:** After multi-route CAT prune, depth 3 took 400k+ nodes; ~90 walls searchable per node.

**Cause:** `wall_edge_heat >= CAT_COLD_CM` (60) marks huge fringe; almost every central wall qualified.

**Fix:** Use `CAT_HOT_CM` (160) for prune gate. Witness path still admits exact shortest-path cuts.

**Lesson:** Multi-route awareness needs a **tight** heat floor — cold fringe is for LMR, not move list membership.

---

## 21. `bitboard_jump_lateral` test wrong geometry

**Symptom:** `bitboard_matches_scalar` failed; asserted lateral targets `(4,3)` and `(4,5)` with pawns at rows 3 and 5 (not face-to-face).

**Cause:** Test position didn't block straight jump correctly; expected squares unreachable in real rules.

**Fix:** Adjacent pawns `(4,4)/(5,4)`, wall behind black — laterals `(5,3)/(5,5)` only.

**Lesson:** Pawn jump tests need face-to-face + blocked forward jump, not distant pawns.

---

## 22. Incremental `DirMasks` on `Board` regressed perft ~12×

**Symptom:** After wiring `dir_masks` into `set_wall` + `Board`, d4 jumped **~3.4s → ~20–40s** (same node count).

**Cause:** Patching masks on every wall make/unmake in the perft tree dominates; wall edges ≫ movegen nodes. One `DirMasks::from_board` per node (scratch hash cache) is cheaper.

**Fix:** Reverted on-board incremental masks. Kept `BfsScratch` hash-keyed cache + `ShiftCanStep` pawns.

**Lesson:** Perft optimizations must account for **wall branching in the tree**, not just movegen at each node.

---

## 23. Eval-zone ID spin (ply37 d53 @ -1.69)

**Symptom:** Game A ply37: eval flat at **-1.69**, ID spun **d36→d57** burning full 10s budget.

**Cause:** `EvalZoneState` required `marginal_nodes < 20_000` per depth — deep cheap iterations never triggered stop.

**Fix:** Stop when `stable_iters ≥ 3 && depth ≥ 12` regardless of marginal node cost.

**Lesson:** "Cheap depth" is relative; stable eval means **stop**, not "keep going because nodes are small."

---

## 24. False perft regression from stale `target-bench` + E-core pinning

**Symptom:** d4 appeared ~40s or timed out at 10s while CLI on `target/release` showed ~3.4s.

**Cause:** (1) `CARGO_TARGET_DIR=target-bench` stale binary. (2) Timed test pinned worker to last logical core (E-core on hybrid CPUs).

**Fix:** Benchmark `target/release`; pin worker to core 2 in timed test.

**Lesson:** Always verify binary path and CPU core class before claiming a perf regression.

---

## Oracle stack (for cross-platform debugging)

1. **Primary:** scraped `web/src/lib/gameLogic.js` (netlify UI rules)
2. **Secondary:** [gorisanson/quoridor-ai](https://github.com/gorisanson/quoridor-ai) full legal moves (`benchmark/lib/gorisanson_moves.mjs`)
3. **Reference:** pavlosdais C wall geometry (`_vendor/pavlosdais-quoridor`) — no perft

All three agree at **perft depth 3** (2_062_264 nodes) after bug #3 fix.
