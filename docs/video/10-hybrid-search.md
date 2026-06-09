# Episode 10 — Hybrid search (MCTS ↔ minimax)

> **Status (Jun 2026):** Historical episode for the **JS Gorisanson lab** hybrid. The **Rust Titanium engine** deprecated MCTS — all `genmove` routes to pure negamax + adaptive LMR. Keep this doc for the web-worker arc; do not describe Rust as MCTS-hybrid in new recordings.

- **branch:** _(checkpoint when ready)_
- **commit:** _(update after tag)_
- **tag:** `checkpoint-10-hybrid-search`

## Hook

"We beat Ka in the browser — but the engine was burning 10 seconds in a won endgame. Episode 10: **one search core** that picks MCTS or minimax from game state, races politely when ahead, and reads the terminal like a human."

---

## What shipped

| Piece                                | Path                                                      |
| ------------------------------------ | --------------------------------------------------------- |
| Shared hybrid search                 | `benchmark/lib/gorisanson_search_core.mjs`                |
| Branching proxy (walls × open slots) | same — no wall enumeration                                |
| Win-path fast exit                   | `tryEfficientWinMove` when `our_dist < opp_dist`          |
| Minimax + LMR                        | Gorisanson move order + late-move reduction               |
| Web worker                           | `web/src/workers/gorisansonWorker.js`                     |
| Ka terminal bench                    | `benchmark/titanium_lab_vs_ka.mjs`                        |
| Unicode terminal board               | `benchmark/lib/terminal_board.mjs`                        |
| Readable ply log                     | `benchmark/lib/terminal_reporter.mjs`                     |
| Live UI footer + naive eval          | `web/src/ui/gameFooter.js`, `gameLogic.naiveDistanceEval` |

---

## Search modes (per move)

```
trivial / opening     → instant
win-path (race)       → BFS shortest step when ahead in distance
MCTS                  → high branching (wall stock × open board)
minimax + LMR         → low branching; ordered moves from gorisanson heuristics
```

**Branching proxy:** `pawn_moves + ourWalls × (openSlots/128) × (totalWallsLeft/20) × 12`  
If `b^d ≤ ~800k` at depth ≥ 8 → minimax; else MCTS.

**Not:** hard "0 walls → skip search". Enemy walls still matter until the proxy says the tree is small.

---

## Terminal benchmark + web replay

```bash
node benchmark/titanium_lab_vs_ka.mjs --games 2 --time 10 --ka short -v
node benchmark/titanium_lab_vs_ka.mjs -v --board   # ASCII grid each ply
```

Each game prints a **`tq1 …`** replay line — copy into web **Replay** tab to scrub the board visually.

Format: `tq1#{"winner":"Ka","plies":46} e2 e8 e3 …` (shared: `web/src/lib/replayCode.js`).

Example ply line (after layout fix):

```
── ply  35 ── White / Titanium ── minimax d7 (12ms) ──
    ▶ g5
 9 · · · · · · · · ·
 8 · · · · · · · · ·
 ...
    walls left W:0 B:3   on board h:12 v:11
```

**Prior result (pre win-path fix):** Game 1 Ka def. Titanium in 46 plies — Ti used full 10s MCTS midgame, minimax only last ~6 plies. Re-run after episode 10 changes.

---

## LMR + move order (from gorisanson MCTS)

Stolen from `_vendor/quoridor-mcts/src/js/ai.js`:

1. Pawn moves on **shortest path** first (`chooseShortestPathNextPawnPositionsThoroughly`)
2. Walls that **disturb opponent path** (`getArrOfValidNoBlockNextWallsDisturbPathOf`)
3. Late moves searched at reduced depth; **re-search** if score beats alpha

Constants: `LMR_MIN_DEPTH=3`, `LMR_AFTER_MOVE=4`.

---

## Web UI (same episode)

- Eval bar: **left of board** only; naive `Black_dist − White_dist`
- Footer: turn, move list, live mode (`MCTS 45k sims` / `win-path` / `minimax d9`)
- No 10s think when forced win path

---

## Terminal Quoridor references

| Project                                                       | Notes                                                 |
| ------------------------------------------------------------- | ----------------------------------------------------- |
| [Zoridor](https://github.com/ringtailsoftware/zoridor)        | Zig, full terminal board, MVM `-1 machine -2 machine` |
| [quoridor.js](https://github.com/OyvindSabo/quoridor.js)      | `getUnicodeRepresentation()` box-drawing              |
| [pavlosdais/Quoridor](https://github.com/pavlosdais/Quoridor) | C engine, αβ + TT, `showboard` command                |

We use a **compact pawn grid** for benchmarks; full wall mesh later if needed.

---

## Video beats

1. Show old terminal log (wall of `time 49865 sims`) — unreadable
2. `-v --board` after reporter
3. Browser: Titanium vs Ka, footer live mode, fast endgame plies
4. Whiteboard: branching proxy formula → mode switch
5. Optional: explain LMR on one minimax ply in DEBUG

---

## Next

- Episode 09 pondering (prep exists)
- Rust port of hybrid core (replace JS worker for Titanium brand)
- Full Unicode wall board in terminal (zoridor-style)
