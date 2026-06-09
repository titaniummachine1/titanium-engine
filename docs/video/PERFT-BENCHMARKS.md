# Perft benchmarks — depth 3 is the correctness standard

**Default gate:** perft **3** from startpos = **2_062_264** nodes (all oracles must match).

## Node table (startpos)

| Depth |           Nodes | Use                                             |
| ----- | --------------: | ----------------------------------------------- |
| 0     |               1 | sanity                                          |
| 1     |             131 | fast smoke                                      |
| 2     |          16,677 | quick divide drills                             |
| **3** |   **2,062,264** | **CI / competition cross-check**                |
| **4** | **247,569,030** | stress test — locked oracle (`PERFT4_STARTPOS`) |
| 5     |  28,837,934,502 | Ishtar/Canta reference only (~18s)              |

## Commands (default = depth 3)

```bash
cd engine
cargo test --release                    # includes perft_depth3_matches_js_oracle
cargo test --release perft_depth4 -- --ignored --nocapture   # timed d1..d4, 10s cap on d4
cargo run --release -- perft            # same as perft 3
cargo run --release -- perft-id 3       # iterative deepening 0..3
cargo run --release -- divide           # divide at depth 3

node benchmark/perft_triple.mjs         # scraped JS + gorisanson + Rust
node benchmark/compare_perft.mjs          # JS vs Rust timing
node benchmark/perft_diff.mjs           # divide diff when something breaks
```

## Titanium perft stack (fundamentals)

| Layer           | What                                                                  |
| --------------- | --------------------------------------------------------------------- |
| Tree walk       | `make_move` / `unmake_move` (no clone per node)                       |
| Hash            | Zobrist incremental                                                   |
| TT              | Clustered buckets (4 slots), `(hash, depth) → nodes`                  |
| Move gen        | `generate_legal_moves_slice` → stack `[Move; 140]`                    |
| Wall legality   | Collision → topology → **known-path skip** → in-place flood trial     |
| Flood fill      | `DirMasks` (N/S/E/W u128) + bitwise shifts on centered 11-wide layout |
| Component reuse | Ishtar trick: if P2 pawn ∈ P1 flood component, skip P2 flood          |
| Build           | `lto = fat`, `codegen-units = 1`                                      |

Full discovery log: `PERFT-OPTIMIZATIONS.md`.

## Release timings (this machine, startpos)

| Depth | Time       | nps (approx) | Notes                                             |
| ----- | ---------- | ------------ | ------------------------------------------------- |
| 3     | **~0.06s** | ~34M         | After Layer 4 flood fill                          |
| 4     | **~3.4s**  | ~73M         | Locked oracle — search slowness is separate issue |
| 5     | ~18s       | —            | Ishtar reference only                             |

Run `cargo run --release -- bench 3 20` for a stable nps average.

## Competition comparison (3s wall-clock race)

| Engine                                                              | Max depth in 3s | Nodes at best      |
| ------------------------------------------------------------------- | --------------- | ------------------ |
| **Titanium Rust**                                                   | **3**           | 2,062,264 (~0.10s) |
| scraped UI JS                                                       | 2               | 16,677             |
| [gorisanson/quoridor-ai](https://github.com/gorisanson/quoridor-ai) | 2               | 16,677             |
| pavlosdais C                                                        | —               | no perft           |

Rust reaches the **standard depth-3 gate** in under a second. JS oracles need ~15–30s for depth 3 — run `perft_triple.mjs` when validating, not every save.

## The surprise: Rust ≠ infinite depth

We moved to Rust expecting to “just go deeper.” Reality:

|                     | Chess habit | Quoridor reality               |
| ------------------- | ----------- | ------------------------------ |
| Root branching      | ~20         | **~131**                       |
| Serious perft depth | 6+          | **3** (2M nodes)               |
| Depth 4             | routine     | **~250M nodes** even optimized |

**Rust + make/unmove + flood fill made depth 3 trivial and depth 4 ~3.4s.** Depth 5+ is still exponential — search needs pruning, not more perft tricks.

**Build pitfall:** stale `target-bench/` or `CARGO_TARGET_DIR` env → false ~40s d4 readings. Always benchmark `engine/target/release/titanium.exe` after `cargo build --release`.

For **search** (not correctness perft), the same rule applies: raw move gen × naive tree walk dies. That’s why gorisanson [prunes walls in MCTS](https://github.com/gorisanson/quoridor-ai) and pavlosdais uses αβ + TT — not because JS/C is slow, but because **dumb full trees are impossible**.

Our plan: perft 3 proves legality; **αβ + TT + tactical walls** makes search tractable.

## Video line

"We don't use chess depth 6. Quoridor correctness is **perft 3, two million nodes**. And even in Rust, perft 4 taught us: **speed without smarts is still exponential.**"
