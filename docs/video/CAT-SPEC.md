# Consensus Attention Table (CAT) — Specification

The **Consensus Attention Table (CAT)** is a per-square heuristic heat map used exclusively
for **move ordering** and **Late Move Reductions (LMR)** in the αβ search.
It is **not** used for static position evaluation.

All values are in **centi-units** (integer arithmetic; 100 cm = 100%).
No floating-point operations. The entire table is built in one BFS pass per player.

---

## Formula

```
attention_weight_cm(dist) = 100 - clamp(dist × 3, 0, 30)
```

| dist | weight |
|-----:|-------:|
| 0    | 100 cm |
| 1    | 97 cm  |
| 3    | 91 cm  |
| 5    | 85 cm  |
| 10   | 70 cm  |
| 15+  | 70 cm (floor) |

---

## Algorithm (per player)

### Step 1 — Forward BFS (level by level)

Run a level-BFS from the player's pawn on the current wall graph.
For every square reached at distance `dist`:

```
cat[square] += attention_weight_cm(dist)
```

Record `dist[sq]` and `parent[sq]` for every reached square.
Stop as soon as any square in the goal row is first touched (early stop — we only need
the shortest path length and one goal square, not the full component).

### Step 2 — Back-propagation (shortest path)

Walk `parent[]` from the first goal square reached back to the pawn.
For every square on that path (including both endpoints):

```
cat[square] += attention_weight_cm(dist[square])
```

This doubles the weight for squares on the optimal route.

### Step 3 — Merge both players

Call steps 1–2 for Player One, then Player Two, accumulating into the **same** `cat` array.
The result merges:
- P1's offensive route (squares on P1's shortest path to goal).
- P2's defensive relevance (squares on P2's path = natural blocking targets for P1).

---

## Resulting score ranges

| Square type                    | Max score (one player) | Max combined (both) |
|--------------------------------|----------------------:|--------------------:|
| On-path, adjacent (dist ≤ 1)   | **200 cm**            | **400 cm**          |
| On-path, far (dist ≥ 10)       | 140 cm                | 280 cm              |
| Off-path, adjacent             | 100 cm                | 200 cm              |
| Off-path, far                  | 70 cm                 | 140 cm              |

---

## Usage in search (`search.rs`)

| Threshold         | Effect                                                    |
|-------------------|-----------------------------------------------------------|
| `CAT_HOT_CM = 180` | Move is tactical — skip LMR, search at full depth       |
| `CAT_COLD_CM = 80` | Move is cold — apply +1 extra ply of LMR reduction      |
| Otherwise         | CAT score used as a tie-breaker in `move_order_score`    |

CAT is built once per search node via `BfsScratch::build_consensus_attention`.
Pawn moves never change the wall graph, so the CAT is valid for all sibling nodes
at the same depth — but currently rebuilt each time (see `docs/STATE.md` §Next optimizations, item 4).

---

## Why it works (intuition)

Quoridor has no "captures." A move's importance is determined by whether it sits on
a player's optimal route. CAT quantifies exactly this:

- A wall placed on a **hot square** (200 cm) likely blocks or extends a player's path — tactical.
- A wall on a **cold square** (≤80 cm) is far from both paths — cosmetic, safe to reduce.
- Pawn steps to a **hot square** shorten the path or start a race — search at full depth.

The dual accumulation (forward + backprop) ensures that squares *on* the path score higher
than squares merely *near* the path, providing a natural priority gradient.

---

## Implementation pointers

| Symbol | File | Notes |
|--------|------|-------|
| `ConsensusAttention` | `engine/src/path.rs` | `type [u16; 81]` |
| `attention_weight_cm` | `engine/src/path.rs` | Formula function |
| `add_player_attention` | `engine/src/path.rs` | Steps 1–2 above |
| `BfsScratch::build_consensus_attention` | `engine/src/path.rs` | Calls both players, returns table |
| `CAT_HOT_CM`, `CAT_COLD_CM` | `engine/src/search.rs` | Search thresholds |
| `cat_score_for_move` | `engine/src/search.rs` | Maps `Move` → CAT score |
