# Titanium Engine — Training Handoff Notes

**Last updated:** 2026-08-05  
**Author:** Cascade (previous sessions)  
**Purpose:** Handoff for GLMM 5.2 to implement proper policy flywheel training

---

## What Was Done

### 1. Policy Network Architecture (DONE, working)

**File:** `src/titanium/policy/mod.rs`  
**Architecture:** Conv policy head — 14 planes × 9×9 → Conv2d(14→64, 3×3) → 6× ResBlock(64, 1×1) → Conv2d(64→8, 1×1) → FC(648→209)  
**Parameters:** ~194K  
**Output:** 209 spatial logits (81 pawn cells + 64 horizontal wall slots + 64 vertical wall slots)  
**Blob format:** Header (magic, planes, channels, blocks, head_channels, actions) + layer weights/biases  
**Blob size:** 776,860 bytes  
**Tests:** 8/8 pass (`cargo test --release -p titanium --lib policy`)

### 2. Pretraining (DONE, plateaued)

**Script:** `training/tools/policy_head/train_spatial_policy.py`  
**Data:** 96 arena games (Sugiyama/Titanium/Sparky bots) + self-play games  
**Encoding:** 14-plane canonical (side-to-move relative):
- Planes 0-1: own/opp pawn position
- Planes 2-3: horizontal/vertical walls
- Plane 4: own walls remaining (normalized)
- Plane 5: opp walls remaining
- Plane 6: constant 1.0
- Planes 7-8: BFS distance to goal (own/opp)
- Planes 11-12: legal wall slots (H/V)
- Plane 13: distance difference (opp_dist - own_dist)

**Best pretraining result:** val_loss=1.275, top1=71.5% (iteration 2 of flywheel)

### 3. Policy Flywheel (DONE, converged/plateaued)

**Script:** `training/tools/policy_head/policy_flywheel.py`  
**Iterations:** 17/20 (converged after patience exhausted)  
**Self-play:** 64 games per iteration, 1.0s/move, 8 threads  
**Data window:** 500 recent self-play games (sliding window) + 96 arena games  
**Training:** 200 epochs max, early stopping patience=15  

**Flywheel history (key iterations):**

| Iter | val_loss | top1_acc | Notes |
|------|----------|----------|-------|
| 1    | 2.195    | 53.9%    | Fresh init |
| 2    | 1.275    | 71.5%    | **Best** — arena data dominates |
| 3    | 1.395    | 69.9%    | Self-play diluting |
| 5    | 1.705    | 61.2%    | Regression — stale self-play |
| 9    | 1.365    | 68.1%    | Partial recovery |
| 15   | 1.396    | 69.2%    | |
| 17   | 1.520    | 65.5%    | Converged — no improvement |

**Output:** `training/runs/conv_policy_flywheel/policy_best.bin` (iter 2)  
**Final:** `training/runs/conv_policy_flywheel/policy_final.bin` (iter 17)

### 4. NNUE Distance-Field Extension (DONE, not integrated)

**File:** `src/titanium/eval/nnue.rs`  
**Added fields:** `dist_field_active: bool`, `dist_me_canon`, `dist_opp_canon`, `dist_diff_canon` (each `Box<[[f64; 81]; 2]>`)  
**Pretrained weights:** `src/weights/net_weights_dist.bin` (345,488 bytes)  
**Status:** Compiles, loads, backward compatible (zero-padded → inactive). SPRT gate was inconclusive (32/100 games, score 0.531).

### 5. RL Dist-Field Flywheel (DONE, saturated)

**Script:** `training/tools/policy_head/rl_nnue_dist_flywheel.py`  
**Result:** Saturated at val_acc 63.5%, gate 52-48 (inconclusive). 243 params too few for meaningful gains.

---

## What Did NOT Work

### Problem 1: Self-play data quality plateaued

The flywheel generated 64 games per iteration at 1.0s/move, but **the self-play data became stale** — the same engine playing itself produces increasingly homogeneous games. By iteration 5, val_loss regressed from 1.275 to 1.705. The sliding window of 500 games was too small to diversify, and the engine's move selection didn't change enough between iterations.

**Root cause:** The policy network was NOT wired into the engine's move ordering. Self-play games used the same search with the same NNUE weights — the policy head was trained offline but never influenced the search. Each iteration's self-play was essentially identical to the previous one.

### Problem 2: Only final move recorded, not move ordering

**This is the critical gap the user identified.** The `match --dump-games` command outputs only:
```
GAME e2 e8 e3 e7 e4 e6 d3h c6h
RESULT W
```

It records the **move played** (algebraic notation), not:
- The move ordering index (where in the sorted candidate list the chosen move ranked)
- The search's evaluation of each candidate
- The PV (principal variation) at each ply
- The node count spent on each candidate

**What's needed:** The engine should log per-ply:
1. All legal moves at that position
2. The move ordering index of the move actually played (0 = first candidate = best)
3. The search score for each candidate (or at least the top N)
4. Time/nodes spent

This lets the policy network learn from **which moves the search prioritized**, not just the single best move. A good policy network should predict the search's move ordering, making the search faster (more pruning) and better (wider effective search at same depth).

### Problem 3: Policy head not wired into search

The policy head was trained but never integrated into `search_impl.rs` move ordering. The engine's move ordering still uses CAT (Corridor Attention) heatmap, which:
- Costs 47% of search time
- The net learned to ignore CAT eval inputs (mean |0.13| cp)
- Is scaffolding — useful for training, removable at inference

**What's needed:** Replace CAT with policy head logits for move ordering. The policy head runs on ~7% of nodes (can afford BFS for full 14-plane encoding).

### Problem 4: Dist-field weights too small (243 params)

The RL dist-field flywheel trained only 243 weights (3 planes × 81 cells). This is far too few for meaningful improvement. The NNUE itself (74.5K params) is the binding constraint, not the dist-field add-on.

---

## What Needs To Be Done (For GLMM 5.2)

### Step 1: Add move ordering logging to the engine

**File to modify:** `src/main.rs` (the `match` command) or `src/titanium/search/search_impl.rs`

Add a `--dump-move-indices` flag to the `match` command that outputs per-ply:
```
PLY 0 pos=e2|e8|... candidates=123 played_idx=0 played=e2
  cand[0] e2  score=+0.51  nodes=12008
  cand[1] e3  score=+0.20  nodes=5000
  cand[2] d4  score=-0.10  nodes=3000
  ...
PLY 1 pos=e2|e8|e3|... candidates=121 played_idx=0 played=e8
  ...
```

Or a more compact format:
```
MIDX 0 0 e2 123 0 0.51 12008 1 0.20 5000 2 -0.10 3000
MIDX 1 0 e8 121 0 -0.90 640896 1 -0.54 213417
```

**Key implementation detail:** The search already sorts moves by CAT/policy ordering. The index of the chosen move in this sorted list is the training target. If the policy head correctly predicts the search's ordering, the index should be 0 (or very low) for most moves.

### Step 2: Wire policy head into move ordering

**File:** `src/titanium/search/search_impl.rs`  
**Current move ordering:** CAT heatmap (expensive, 47% of search time)  
**Target:** Policy head logits (cheap inference, runs on ~7% of nodes)

The policy head produces 209 spatial logits. For move ordering:
1. Run policy head on the current position (only at internal nodes, not leaves)
2. Map each legal move to its action index (0-208)
3. Sort by policy logit (descending)
4. Use this ordering for PVS alpha-beta search

**Loading the policy blob:** The engine already has `src/titanium/policy/mod.rs` with blob loading. Need to:
- Load `policy_best.bin` at startup
- Call `PolicyHead::forward(&self, planes: &[f32; 1134]) -> [f32; 209]` at each node
- Map move → action index using the same encoding as `move_to_action_index()` in the Python script

### Step 3: Proper flywheel with move index logging

**New script:** `training/tools/policy_head/policy_flywheel_v2.py`

Each iteration:
1. Run self-play with `--dump-move-indices` (1s/move, 8 threads, 64+ games)
2. Parse the move index log — extract (position, all_candidates, played_idx, scores)
3. Train policy head with cross-entropy on the **full move ordering** (not just the played move)
   - Target: softmax over search scores (temperature-scaled)
   - Or: cross-entropy on the played move's index (lower = better ordering)
4. Export updated policy blob
5. Reload engine with new policy weights for next iteration's self-play
6. Repeat

**Critical difference from v1:** The policy network actually influences the search (via move ordering), so each iteration's self-play is genuinely different. The move index logging provides much richer training signal than just the final move.

### Step 4: SPRT gate after each flywheel iteration

Run 100-game SPRT match between current policy weights and previous best. Only accept new weights if they pass the gate (score > 0.55 with tight confidence).

---

## File Map

### Engine (Rust)
- `src/titanium/policy/mod.rs` — Policy head module (ConvPolicyHead, blob I/O, 14-plane encoding)
- `src/titanium/eval/nnue.rs` — NNUE weights + dist-field extension
- `src/titanium/search/search_impl.rs` — Search + eval (move ordering here)
- `src/main.rs` — CLI entry point (`match` command, `--dump-games`)
- `src/movegen/` — Move generation (legal.rs, wall_masks.rs, pawn_bits.rs)
- `src/weights/net_weights.bin` — Current NNUE weights (74.5K params)
- `src/weights/net_weights_dist.bin` — NNUE + dist-field weights (345K)

### Training (Python)
- `training/tools/policy_head/policy_flywheel.py` — V1 flywheel (plateaued, only logs final move)
- `training/tools/policy_head/train_spatial_policy.py` — Pretraining script
- `training/tools/policy_head/train_conv_policy.py` — Conv policy trainer
- `training/tools/policy_head/rl_nnue_dist_flywheel.py` — RL dist-field flywheel (saturated)
- `training/tools/policy_head/frame.py` — Game replay utilities
- `training/titanium_training/store/` — Move codec, state replay, position extraction

### Training Data
- `training/runs/arena_games2.db` — 96 arena games (Sugiyama/Titanium/Sparky)
- `training/runs/conv_policy_flywheel/` — Flywheel outputs (17 iterations)
  - `policy_best.bin` — Best weights (iter 2, val_loss=1.275, top1=71.5%)
  - `policy_final.bin` — Final weights (iter 17, val_loss=1.520, top1=65.5%)
  - `flywheel_history.json` — Full iteration history
- `training/runs/rl_dist_flywheel/rl_games.db` — Self-play game database

### Website
- `site/web/src/lib/titaniumEngines.js` — Single source of truth for engine versions
- `site/web/src/workers/titaniumWasmWorker.js` — WASM worker (uses registry for constructor selection)
- `site/web/src/lib/titaniumWasmClient.js` — WASM client (uses LATEST_TITANIUM.mode)

---

## Build Commands

```powershell
# Native build
$env:RUSTFLAGS = '-C target-cpu=native'
cargo build --release -p titanium

# WASM build (from site/web/)
cd site/web
npm run build:wasm

# Tests
$env:RUSTFLAGS = '-C target-cpu=native'
cargo test --release -p titanium --lib

# Policy tests only
cargo test --release -p titanium --lib policy

# Website build
cd site/web
npm run build          # local
npx vite build --mode ghpages  # GitHub Pages

# Training flywheel (v1, plateaued)
python training/tools/policy_head/policy_flywheel.py --iterations 20 --games-per-iter 64 --threads 8
```

---

## Key Findings

1. **Policy head architecture is sound** — 194K params, conv with residual blocks, 209 spatial actions. Pretraining reached 71.5% top-1 accuracy.

2. **Flywheel plateaued because policy was never wired into search** — self-play games were identical between iterations since the policy didn't affect move selection.

3. **Move ordering is the training signal, not just the final move** — the engine needs to log which index the chosen move was in the sorted candidate list. This is the key insight for v2.

4. **CAT costs 47% of search time** — replacing it with policy head logits would be a major speedup even if policy quality is equal.

5. **NNUE is 74.5K params** — conservative but proven. 200K-param engines beat 15M-param engines. Capacity is not the binding constraint at this scale.

6. **Distance fields are nearly free** — d0_layers/d1_layers already maintained incrementally by search. But 243 extra params is too few to matter.

7. **Arena data (96 games) was the best training signal** — iteration 2 (arena-dominated) had the best val_loss. Self-play diluted quality.

---

## Engine Versioning (Website)

- **v19** (latest): witness-optimized movegen, `new_v19()` constructor, live NNUE weights
- **v18** (active): v17 search + latest NNUE, `new_v18()` constructor
- **v17** (deprecated): frozen snapshot, `new_v17()` constructor, fixed NNUE
- **v16** (retired): removed from website UI, kept in `PlayerType` enum for saved game compat

All engine versions managed via `site/web/src/lib/titaniumEngines.js` — single source of truth. Adding a new version = one entry in `TITANIUM_ENGINES` array.

---

## Repo Layout

```
C:/gitProjects/Quoridor best AI/
├── engine/                    # Live engine repo (always latest, on main)
│   ├── src/
│   │   ├── titanium/
│   │   │   ├── policy/        # Policy head module
│   │   │   ├── eval/          # NNUE
│   │   │   ├── search/        # Search implementation
│   │   │   └── position/      # Game state
│   │   ├── movegen/           # Move generation
│   │   ├── weights/           # NNUE weight files
│   │   └── wasm.rs            # WASM bindings
│   ├── Cargo.toml
│   └── TRAINING_HANDOFF.md    # This file
├── site/web/                  # Website + WASM build
│   ├── src/
│   │   ├── lib/titaniumEngines.js  # Engine version registry
│   │   ├── workers/titaniumWasmWorker.js
│   │   └── ...
│   ├── .github/workflows/deploy-pages.yml  # CI: builds WASM + deploys
│   └── package.json
├── training/                  # Training scripts + data
│   ├── tools/policy_head/     # Policy training scripts
│   ├── runs/                  # Training outputs
│   └── titanium_training/     # Training library (move codec, state replay)
└── artifacts/                 # Release artifacts + worktrees
```
