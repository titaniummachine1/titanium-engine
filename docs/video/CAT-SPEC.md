# Corridor Attention Table (CAT) v3 — Specification

The **Corridor Attention Table (CAT)** is per-square / per-wall-edge heat used for **move ordering**, **LMR**, and **search move pruning**. It is **not** used in static evaluation.

All values are **centi-squares** (integer). `100 cm` ≈ one shortest-path step in eval units.

---

## v3 vs v2 (why v3 exists)

|               | CAT v2                                                      | CAT v3                                      |
| ------------- | ----------------------------------------------------------- | ------------------------------------------- |
| Route model   | One backtracked shortest path per player                    | **All squares** with corridor delta ≤ 3     |
| Tunnel vision | Wall on equal-length reroute missed if off witness path     | `wall_edge_heat` sees multi-route corridors |
| Build         | `engine/src/cat/build.rs` — `add_player_corridor_attention` | same module, corridor delta formula         |

Witness shortest path is still used for **fast** gates (ordering, qsearch noisy check). CAT heat handles **near-equal** routes.

---

## Corridor heat formula

Per player, per square `sq`:

```
delta(sq) = dist_from_pawn(sq) + dist_to_goal(sq) - shortest_to_goal
heat(sq)  = corridor_heat(delta) * pawn_path_weight(dist_from) / 100
```

`corridor_heat(delta)` — nonzero only if `delta ≤ MAX_RELEVANT_CORRIDOR_DELTA` (3):

```
corridor_heat(δ) = round( CAT_CORRIDOR_CM / (1 + δ × log2(δ+2)) ), min 1
```

`CAT_CORRIDOR_CM = 200`. Bottleneck bonus (+40 cm) when delta ≤ 2 and ≤1 forward continuation.

Both players accumulate into search `CorridorAttention`; **web display** uses per-player **max** (not sum) to avoid full-board flood.

---

## Thresholds (`engine/src/cat/constants.rs`)

| Constant              | Value | Effect                                                   |
| --------------------- | ----: | -------------------------------------------------------- |
| `CAT_CORRIDOR_CM`     |   200 | Per-player corridor ceiling                              |
| `CAT_HOT_CM`          |   160 | Tactical — skip LMR; **wall prune gate** for multi-route |
| `CAT_COLD_CM`         |    60 | Cold fringe — +1 LMR; UI warm tint from here             |
| `BOTTLENECK_BONUS_CM` |    40 | Narrow corridor squares                                  |

**UI display max** = `CAT_CORRIDOR_CM + BOTTLENECK_BONUS_CM` = **240 cm**.

---

## Wall edge heat

For wall at `(row, col, orientation)`:

```
edge_heat = max( heat(square_a), heat(square_b) ) per wall segment
wall_edge_heat = max(edge_top, edge_bottom) + min(edge_top, edge_bottom) / 4
```

Plus optional shape bonus (`wall_shape_attention_bonus`) for cross-gap geometry when local heat ≥ HOT — ordering only, does not un-prune dead walls.

---

## Search integration

### `collect_search_moves` (prune list)

Walls kept if `wall_should_search`:

1. Not in dead zone (all touched squares unreachable)
2. Cross-gap or blocks cross-gap → always search
3. Touches gap play zone (mouth + ring)
4. Probes sealed interior away from gap → **skip**
5. Intersects opponent **witness** shortest path → search
6. Else `wall_edge_heat ≥ CAT_HOT_CM` → search (multi-route)

Pawns: all legal (main search); qsearch pawns with `our_path_gain > 0` only.

### LMR / futility

- `move_corridor_attention(mv)` = square or edge heat + shape bonus
- `≥ CAT_HOT_CM` → tactical (no reduction)
- `< CAT_COLD_CM` → +1 reduction ply
- Walls off witness path → non-tactical without BFS

### Quiescence (noisy moves)

Chess captures → Quoridor **path shocks**:

- Pawn: shortens our shortest path
- Wall: increases opponent shortest path (witness gate + BFS confirm)
- Quiet → stand pat; max 8 noisy moves; `MAX_QDEPTH = 6`

---

## Gap / funnel positions (B-roll reference)

Vertical wall chain with single mouth at top:

- Walls through `V|gap|V` → cross-gap → **always search**
- Flank block beside gap → **always search**
- Wall touching only sealed void away from gap mouth → **prune**

Bug fixed: neighbor loops in `gap_play_zone_mask` must use `0..=8`, not `0..=9` (phantom edge mouths).

---

## Implementation pointers

| Symbol                                       | File                                        |
| -------------------------------------------- | ------------------------------------------- |
| `CorridorAttention`                          | `engine/src/cat/attention.rs`               |
| `build_corridor_attention`                   | `engine/src/cat/build.rs`                   |
| `collect_search_moves`, `wall_should_search` | `engine/src/cat/prune.rs`                   |
| `cat_snapshot_json`                          | `engine/src/cat/viz.rs`                     |
| Search consumer                              | `engine/legacy/search/alphabeta.rs`         |
| Web overlay                                  | `web/src/lib/catHeatmap.js`, `boardView.js` |

---

## Intuition (narration)

Quoridor has no captures. A move matters if it sits on **any** near-optimal route for either player. CAT v3 paints those corridors; the witness path is the express lane check; together they stop us pruning the wall that closes the last gap — or keeping ninety walls that touch cold fringe.
