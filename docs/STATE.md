# Titanium Engine — Session State Handoff

**Purpose:** Carry context into a new chat without re-discovery.  
**Last updated:** perft closure + eval-spin fix (Jun 2026).

---

## Where we are

| Layer | Status |
| ----- | ------ |
| **Perft** | **Done for now.** d4 = **247_569_030** in **~3.2–3.4s** release (`ShiftCanStep` pawns, `BfsScratch` mask cache). Timed regression: `cargo test --release perft_depth4 -- --ignored --nocapture`. |
| **Movegen** | Production pawn path: `PawnGenMode::ShiftCanStep`. Bitboard/scalar modes remain for benches/tests. |
| **Search** | Pure **ID negamax** + aspiration + adaptive LMR + qsearch + TT + CAT v3 prune. MCTS moved to `search/deprecated/` (routes to negamax). |
| **Mate zone** | Stops losing mates at `mate_dist + 4` instead of spinning to d200+. |
| **Eval zone** | Stops flat eval spin at `stable_iters ≥ 3 && depth ≥ 12` — **no** `marginal_nodes` gate (fixes ply37 d53 / won-position d28+ waste). |
| **CAT overlay** | Web shows raw cm on squares; `titanium cat` API. |
| **Benchmarks** | [BENCHMARK_gorisanson_vs_titanium.md](../benchmark/BENCHMARK_gorisanson_vs_titanium.md) — Games A (loss) + B (win pre-fix). `capture_baseline.mjs` not finished. |

---

## Perft — closed (do not grind further)

| Depth | Nodes | Time (idle CPU) |
| ----- | ----- | --------------- |
| 3 | 2_062_264 | ~0.07–0.08s |
| 4 | 247_569_030 | **~3.2–3.4s** |

**Rejected:** incremental `DirMasks` on `Board` — regressed perft ~12× because wall make/unmake in the tree dominates; one `from_board` per movegen node (scratch hash cache) is faster.

**Pitfall:** `CARGO_TARGET_DIR=target-bench` in shell → stale binaries; use `engine/target/release/titanium.exe`.

---

## Known weaknesses (next work = **eval**)

1. **Opening depth** — still **d4–d6** in 10s (target was much higher). Adaptive LMR helps midgame but opening starved.
2. **Static eval quality** — path-distance eval (+ CAT) misses positional structure; Game A loss vs Gorisanson from opening passivity.
3. **BFS in eval** — `shortest_distance` / mobility still hot; per-node distance cache not wired.
4. **Won-position spin** — eval-zone stop added; needs replay validation on Game B ply37–41.
5. **baseline_depths.json** — `capture_baseline.mjs` hung on opening; never committed.

---

## Architecture snapshot

```
engine/src/
├── core/board.rs          Board, Move, zobrist, make/unmake
├── util/grid.rs           can_step, flood layout, wall bits
├── movegen/
│   ├── legal.rs           legal moves, WallPathCache, ShiftCanStep default
│   └── pawn_bits.rs       bitmask pawn variants (bench/tests)
├── path/                  BfsScratch + DirMasks hash cache (not on Board)
├── cat/                   CorridorAttention, prune, viz
├── search/
│   ├── alphabeta.rs       ID negamax, qsearch, LMR
│   ├── lmr_profile.rs     stage_t, mate/eval zone controllers
│   └── deprecated/mcts.rs inactive
└── util/perft.rs          perft_fast, timed d4 test
```

---

## Regression commands

```bash
cd engine
cargo test --release                              # lib tests
cargo test --release perft_depth4 -- --ignored --nocapture   # ~3.5s timed d1..d4
cargo run --release -- perft 4                    # 247569030 ~3.4s

cargo test --release eval_zone_stops
cargo test --release lost_mate_tq1               # if present
```

---

## Next chat priorities (eval)

See closing handoff in chat or section below in README — focus:

1. **Eval function** — dual BFS distance, wall value, mobility; tune weights vs Gorisanson positions.
2. **Per-node BFS cache** — parent distances; invalidate on wall moves only (big search nps win).
3. **Opening depth** — depth-first LMR defaults vs tactical widen; replay Game A opening.
4. **Validate eval-zone stop** — replay `tq1` ply37 (loss) and Game B ply37–41 (win spin).
5. **capture_baseline.mjs** — shorter `BASELINE_TIME_SEC=3` smoke; commit `baseline_depths.json`.

---

## Video / docs index

| Doc | Content |
| --- | ------- |
| [video/00-SERIES-OVERVIEW.md](video/00-SERIES-OVERVIEW.md) | Series arc — **updated** perft + MCTS deprecation |
| [video/PERFT-BENCHMARKS.md](video/PERFT-BENCHMARKS.md) | Timings + smart test |
| [video/PERFT-OPTIMIZATIONS.md](video/PERFT-OPTIMIZATIONS.md) | Layer 4 final + rejected Layer 5 |
| [video/BUG-DIARY.md](video/BUG-DIARY.md) | Entries #22–#24 added |
| [video/11-search-hardening.md](video/11-search-hardening.md) | Negamax/CAT session |
| [BENCHMARK_gorisanson_vs_titanium.md](../benchmark/BENCHMARK_gorisanson_vs_titanium.md) | Regression games |
