# Episode 11 — Search hardening, CAT v3 gaps, and honest quiescence

- **branch:** _(checkpoint when ready)_
- **commit:** _(update after tag)_
- **tag:** `checkpoint-11-search-hardening`

## Hook

"We matched perft 4 in three seconds — so why does the AI still play `f4v` when `e4` wins the sprint? Episode 11: the bugs weren't in movegen. They were in **how we stop searching**, **how we pick a move at equal scores**, and **what we call a capture in Quoridor**."

---

## What we fought (plot arc)

| #   | Symptom                                             | Root cause                                                                                                         | Fix                                                                     |
| --- | --------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------- |
| A   | Weird passive wall at root while behind             | Root tie-break preferred wall over pawn at **equal** score; `ROOT_TIEBREAK_BAND` let **worse** scores replace best | Tie on `immediate path gain` first, then resistance; band removed       |
| B   | `bestmove` ≠ highest root candidate after time stop | Timeout mid-root-loop: child returns `alpha` → negates to fake fail-high; partial depth committed                  | Stop **before** consuming child score; discard aborted ID iteration     |
| C   | `mateExtensions` always 0                           | `clamp_unproven_mate` ran **before** mate-extension loop, erasing claims                                           | Reorder: extend first, clamp after                                      |
| D   | Gap-mouth walls pruned in funnel positions          | `corridor_mouth_mask` used `0..=9` bounds → every board edge = phantom sealed mouth                                | Neighbor loop uses `0..=8` only                                         |
| E   | Depth 4 regression (was ~3.3s, stalled at d2–d3)    | Search ~10k nps while perft ~18M nps — eval + ordering burned BFS per node                                         | See performance section below                                           |
| F   | Qsearch explored quiet walls 10 plies deep          | `tactical_only` only filtered pawns; walls used full prune list                                                    | Noisy wall = lengthens opp shortest path; quiet stands pat; width cap 8 |
| G   | CAT v2 tunnel vision in prune                       | `wall_should_search` used one witness path + expensive BFS triplet per wall                                        | Witness path + **CAT hot corridor edge** (multi-route, no extra BFS)    |

---

## Sebastian Lague mapping (chess → Quoridor)

Use this table when narrating the episode — same engine skeleton, different "noisy move":

| Chess concept       | Quoridor equivalent in Titanium                                                    |
| ------------------- | ---------------------------------------------------------------------------------- |
| Perft / divide      | `perft 4` = 247_569_030 — **not** the bottleneck anymore                           |
| MVV-LVA ordering    | Path delta: pawn steps that shorten our BFS; walls on opp witness path             |
| Iterative deepening | Yes — aspiration windows; **must** keep last _completed_ depth on timeout          |
| Killer / history    | _(not yet)_ — CAT corridor heat is our positional memory                           |
| LMR                 | `ln(d)×ln(m)/2.25`, cap `depth/2`; extra reduction for cold CAT / off-path walls   |
| Quiescence captures | Walls that **increase** opponent `shortest_distance`; pawn steps that shorten ours |
| Zobrist + TT        | Yes — search TT separate from perft TT                                             |
| Eval pitfalls       | Path length base + wall inventory + mobility; CAT **never** in static eval         |

**Quiescence theory note (on-camera):** "Search until out of walls" is wrong — good players hoard walls. We extend only while a move **changes path lengths** (the Quoridor capture), not while wall stock > 0.

---

## Performance pass (search ≠ perft)

Perft 4: **3.4s / 247M nodes** (~18M nps) — unchanged, correct.

Search was ~10k nps because every node paid:

1. **`pawn_mobility` in eval** — ran full legal movegen (BFS every wall slot) to count ≤5 pawn moves → fixed: `generate_pawn_moves_for` only
2. **`move_order_score` per wall** — 3–4 make/unmake + BFS → witness-path gate: off-path walls score with zero BFS
3. **Triple opponent path** — built in `collect_search_moves`, `order_moves`, and LMR → once per node, shared
4. **CAT built twice** — collect + order → once per node (`depth ≥ 2` only; qsearch skips CAT)
5. **Forcing extension at `dist ≤ 2`** — fired in ordinary races; depth never decreased → tightened to `≤ 1` or ≤1 pawn move
6. **`eval_stm` on every child** for mate clamp → lazy: only when score is unproven mate
7. **Qdepth 10 → 6**, qsearch move cap **8**

After fixes: funnel position reaches **depth 3 in ~3s** (was stuck at depth 2). Depth 4+ still needs per-node BFS caching (next episode).

---

## CAT v3 multi-route (why not one path)

CAT v2 problem: one reconstructed shortest path = tunnel vision. A wall can block an **equal-length reroute** without touching the witness path.

CAT v3 (`engine/src/cat/build.rs`):

- Per player: distance field + corridor delta `from + to - shortest`
- Heat on all squares with delta ≤ 3 (not just one backtracked path)
- `wall_should_search`: witness-path hit **OR** `wall_edge_heat ≥ CAT_HOT_CM` (160)
- Using `CAT_COLD_CM` (60) here admitted ~90 walls/node — tree exploded; HOT keeps list tight

**Gap positions (screenshot episode B-roll):** vertical chain with one mouth — walls touching the gap play zone must stay searchable; walls probing sealed interior away from the mouth must prune. Cross-gap `H` through `V|gap|V` always searchable.

---

## Negamax / αβ checklist (verified this session)

- [x] Negamax sign flip at child (`-negamax(..., -β, -α)`)
- [x] TT mate score ply adjustment (`score_to_tt` / `score_from_tt`)
- [x] Null-move pruning (R=2/3)
- [x] Futility at depth 1–2 (non-tactical only)
- [x] Aspiration windows + fail-soft re-search
- [x] Repetition detection on search path
- [x] Root: no LMR (`ply == 0`)
- [x] Mate horizon clamp unless proven by depth or PV
- [x] `committed_root_moves` snapshot — diagnostics match played move after ID

---

## Pawn movegen (bitboard experiment)

`engine/src/movegen/pawn_bits.rs` — bitmask pawn gen via `DirMasks` shifts; tested against scalar `generate_pawn_moves_slice` (perft depth 3 walk). **Not wired into hot path yet** — perft uses scalar; bitboard is bench/compare only.

Fixed test: face-to-face adjacent pawns + wall behind opponent → lateral jumps only (old test had pawns 2 apart, wrong geometry).

---

## Web: CAT vision UI

See [CAT-VIEW-UI.md](CAT-VIEW-UI.md).

- Removed γ sharpness slider — show **raw engine cm** on every square
- Colors anchored to engine thresholds (60 / 160 / 240), not per-position normalize
- Toggle: Analysis / Play / Replay side panel → **CAT** checkbox

---

## Demo commands (recording)

```bash
cd engine
cargo test --release                    # 73 passed
cargo run --release -- perft 4          # 247569030 ~3.4s

# Funnel position — should play e4 not f4v
cargo run --release -- genmove --engine minimax --time 3 --log \
  e2 e8 e3 e7 e4 e6 d1h e6h d4 c6h d5 a6h e5 e5v

# CAT JSON for web overlay
cargo run --release -- cat e2 e8 e3
```

Web: enable **CAT** in Analysis, read numbers on squares (≥60 warm tint, ≥160 hot).

---

## Still open (tease next episode)

1. Per-node BFS distance cache (parent invalidation on wall moves)
2. Killer / history heuristics for quiet pawn steps
3. Wire `pawn_bits` into hot movegen if bench wins on real positions
4. Depth 4 at 3.3s **search** (not just perft) on midgame positions

---

## Files touched (this arc)

| Area        | Files                                                                                     |
| ----------- | ----------------------------------------------------------------------------------------- |
| Search      | `engine/legacy/search/alphabeta.rs`                                                       |
| CAT prune   | `engine/src/cat/prune.rs`                                                                 |
| CAT viz API | `engine/src/cat/viz.rs`                                                                   |
| Movegen     | `engine/src/movegen/legal.rs`, `pawn_bits.rs`                                             |
| Web CAT     | `web/src/lib/catHeatmap.js`, `boardView.js`, `controlsView.js`, `catHint.js`, `board.css` |
