# Episode 05 — First perft bug: lateral wall check

- **branch:** `checkpoint/05-perft-bugfix` *(after commit)*
- **commit:** *(fill after commit)*
- **tag:** `checkpoint-05-perft-bugfix`

## Hook

"Perft depth 1 matched. Depth 2 didn't. Two nodes off — in Quoridor that's a smoking gun."

## Symptoms

```bash
node benchmark/compare_perft.mjs 2
# JS 16677  vs  Rust 16679
node benchmark/perft_diff.mjs 2
# d8v  JS 127  Rust 128
# e8v  JS 127  Rust 128
```

## Debug workflow (chess-style divide)

1. **Total perft** — `compare_perft.mjs` JS oracle vs Rust
2. **Divide diff** — `perft_diff.mjs` lists which root move subtrees disagree
3. **Drill down** — `perft_diff.mjs 1 d8v` on child position
4. **Move diff** — compare legal move lists; found Rust extra pawn `d9`, JS extra wall `d7h` (symptom)

## Root cause

Bad port of scraped `pawnCanMove` for **lateral** steps in `grid.rs`.

JS checks **two vertical wall anchors** per sideways step:

| Direction | wallAnchor | sideAnchor (step Down) |
|-----------|------------|-------------------------|
| **Right** | `from` | one row below `from` |
| **Left** | `to` | one row below `to` |

Our old code used wrong columns/rows → after wall `d8v`, Black at `e9` could illegally step left to `d9`.

## Fix

```rust
// Right — wallAnchor = from, sideAnchor = step(from, Down)
(0, 1) => !has_vertical(board, js_from, col) && !has_vertical(board, row, col),
// Left — wallAnchor = to, sideAnchor = step(to, Down)
(0, -1) => !has_vertical(board, js_to, nc) && !has_vertical(board, nr, nc),
```

Regression test: `vertical_d8v_blocks_black_left_from_e9` in `grid.rs`.

Oracle test: `perft_depth2_matches_js_oracle` → **16_677** nodes.

## C++ competition reference (pavlosdais/Quoridor)

Cloned to `_vendor/pavlosdais-quoridor` — **no perft**, but same wall idea:

```c
// Moving right from (i,j): block if vertical at (i,j) OR (i+1,j)
char wallOnTheRight(i, j) {
    return wall_matrix[i][j]=='r' || wall_matrix[i+1][j]=='r';
}
// Moving left: wallOnTheLeft(i,j) = wallOnTheRight(i, j-1)
```

Different data structure (`char** wall_matrix` vs our `u64` bitboards), same geometry.

Their move gen is **explicit** (separate white/black, jump cases unrolled). We use **generic** `can_step` + BFS — faster to port from JS, but easier to get anchor rows wrong.

## Demo after fix

```bash
cd engine && cargo test
node benchmark/compare_perft.mjs 2
node benchmark/perft_diff.mjs 2
```

## Lesson for the series

- Quoridor has no standard perft table → **JS UI rules are our Stockfish**
- Depth 2 is enough to catch pawn-wall interaction bugs
- Always **divide** before staring at 131 root moves

## Next episode

Alpha-beta + Zobrist TT (Phase 1 search).
