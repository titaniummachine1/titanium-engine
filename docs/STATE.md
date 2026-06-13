# Titanium Engine — Session State Handoff

**Purpose:** Carry context into a new chat without re-discovery.  
**Last updated:** Jun 2026 — O1 pawn LUT is production (cold-start built; binary
2.8MB→1.0MB), perft d5/d6 gates added, gen13 ACE port landed (`acev13/`).

---

## Where we are

| Layer | Status |
| ----- | ------ |
| **Movegen** | **Closed.** Shift walls + **`O1Lookup` pawns (production default — perft-proven fastest at d4/d5)**; shift/scalar retained as bench/test alts (`docs/MOVEGEN.md`). |
| **Perft** | Gates exact. Bench d3 ~**210–240M nps** (Zobrist §A on `main`). |
| **Search** | Pure **ID negamax** + aspiration + adaptive LMR + qsearch + TT + CAT v3 prune. |
| **ACE** | v11 port (`ace/`, pathfix gen11_ghi) **and v13 port (`acev13/`, ACEV13.html)** — gen13 inlines `certify_win.js`, so the certificate solver runs (eval floor + last-wall refutation gate). CLI: `--engine ace-v13[-ti]`. |
| **Tables** | Pawn O1 tables built at **cold start** (`movegen::o1::runtime`, ~0.1s); `embed-tables` feature bakes them in for prewarmed builds. Binary 2.8MB→1.0MB. |
| **Eval** | Path-distance + CAT; opening depth still shallow. |

---

## Movegen — do not grind further

Single-thread only. No GPU. No movegen multithreading.

| Depth | Nodes | Notes |
| ----- | ----- | ----- |
| 3 | 2_062_264 | CI gate |
| 4 | 247_569_030 | Stress oracle |
| 5 | 28_837_934_502 | sub-12s, public record (`PERFT5_STARTPOS`) |
| 6 | 3_257_436_276_501 | full enumeration (`PERFT6_STARTPOS`) |

**Next perf wins are not movegen:** L3 in wall-heavy search, eval cache — see `docs/MOVEGEN-HANDOFF.md`. §A (Zobrist/Undo) merged on `main`.

---

## Architecture snapshot

```
engine/src/
├── core/board.rs          Board, Move, zobrist, make/unmake
├── movegen/
│   ├── legal.rs           legal moves, lazy WallTrialCtx, O1Lookup default
│   ├── pawn_bits.rs       pawn variants (bench/tests)
│   └── o1/
│       ├── lookup.rs      wall_masks(), shift L2/TOPO; pawn O1 LUT hot path
│       ├── gen/           table-compute (single source of truth, shared w/ emitter)
│       ├── runtime.rs     cold-start PawnTables + OnceLock tables() + prewarm()
│       └── embedded.rs    baked consts (only under `--features embed-tables`)
├── acev13/                gen13 ACE port (game/net/race/search/certify/session)
├── path/parallel.rs       u128 flood + bit theft (L3)
├── search/alphabeta.rs    ID negamax, LMR, CAT prune
└── util/perft.rs          perft_fast, bulk d1, timed d4 test
```

---

## Regression commands

```bash
cd engine
cargo test --release
cargo test --release perft_depth4_matches_oracle -- --ignored --nocapture
cargo run --release --bin titanium -- perft 4
cargo run --release --bin titanium -- bench 3 10
cargo bench --bench perft_pawn_modes
```

---

## Next priorities (Fable / search)

1. **Make/unmake + Zobrist** — profile and slim `Undo` (see MOVEGEN-HANDOFF.md §A)
2. **Eval function** — dual BFS distance, wall value, mobility
3. **Per-node BFS cache** in search eval
4. **Completeness program** — invariant hash collision hunt (batch oracle)
5. **Opening depth** — LMR / replay validation

---

## Video / docs index

| Doc | Content |
| --- | ------- |
| [MOVEGEN.md](MOVEGEN.md) | Production movegen (current) |
| [MOVEGEN-HANDOFF.md](MOVEGEN-HANDOFF.md) | Fable handoff |
| [video/PERFT-BENCHMARKS.md](video/PERFT-BENCHMARKS.md) | Perft gates |
| [video/PERFT-OPTIMIZATIONS.md](video/PERFT-OPTIMIZATIONS.md) | Historical discovery log |
| [video/11-search-hardening.md](video/11-search-hardening.md) | Negamax/CAT session |
