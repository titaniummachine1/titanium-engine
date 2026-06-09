# Titanium Engine — video series overview

Save this file. Record in any order; episodes are checkpointed in git.

## Arc

1. Scraped a Quoridor site — AI is on a server, not in the bundle
2. Built **Titanium Engine** in Rust — rules first, speed second
3. Perft caught our first real bug — fixed with divide debugging
4. **Plot twist:** Rust crushes perft 3; perft 4 needed make/unmake + TT + flood fill — now **~3.4s** (not minutes)
5. Threading prep — `Engine` layout + root-parallel perft bench (`thread-bench`)
6. ~~Hybrid MCTS~~ → **pure αβ + adaptive LMR** (MCTS deprecated, routes to negamax)
7. **Search hardening** — negamax stop/tie-break/qsearch/CAT gap bugs; raw CAT overlay on web
8. **Perft closure** — timed d4 regression test; incremental board masks tried and rejected
9. **Next:** eval quality, BFS cache in search, opening depth vs Gorisanson

## Episode list

| #   | File                                                     | Topic                              |
| --- | -------------------------------------------------------- | ---------------------------------- |
| 01  | [01-path-bfs.md](01-path-bfs.md)                         | Bitboards + BFS                    |
| 02  | [02-legal-moves.md](02-legal-moves.md)                   | 131 moves, JS parity               |
| 03  | [03-perft.md](03-perft.md)                               | Divide harness                     |
| 04  | [04-bench.md](04-bench.md)                               | Criterion / NPS                    |
| 05  | [05-first-perft-bug.md](05-first-perft-bug.md)           | d8v lateral wall bug               |
| 06  | [06-threading-prep.md](06-threading-prep.md)             | Titanium vs Titanium               |
| 07  | [07-ai-opponents.md](07-ai-opponents.md)                 | Gorisanson local boss              |
| 08  | [08-greedy-ui-lab.md](08-greedy-ui-lab.md)               | Testing lab UI + greedy `genmove`  |
| 10  | [10-hybrid-search.md](10-hybrid-search.md)               | ~~MCTS↔minimax~~ (historical; MCTS deprecated) |
| 11  | [11-search-hardening.md](11-search-hardening.md)         | Weird moves, gaps, qsearch, CAT UI |
| —   | [00-HOW-THE-ENGINE-WORKS.md](00-HOW-THE-ENGINE-WORKS.md) | Narration script                   |
| —   | [CAT-VIEW-UI.md](CAT-VIEW-UI.md)                         | Debug overlay spec                 |
| —   | [BUG-DIARY.md](BUG-DIARY.md)                             | All plot twists                    |
| —   | [PERFT-BENCHMARKS.md](PERFT-BENCHMARKS.md)               | Depth timings                      |

## Git checkpoints

See [README.md](README.md) for branch names and commit hashes.

**Future:** every tagged checkpoint → build → round-robin → **Elo ladder** ([TOURNAMENT-ROADMAP.md](TOURNAMENT-ROADMAP.md)). Playable opponents start at episode **07 (αβ)**; earlier tags are perft/speed benchmarks only.

## Competitors (reference only)

| Project                                                             | Role in our repo                       |
| ------------------------------------------------------------------- | -------------------------------------- |
| [gorisanson/quoridor-ai](https://github.com/gorisanson/quoridor-ai) | MCTS + heuristics; perft cross-check   |
| [pavlosdais/Quoridor](https://github.com/pavlosdais/Quoridor)       | C αβ competition engine; wall geometry |
| quoridor-ai.netlify.app                                             | Rules oracle (scraped)                 |

## Do NOT run in videos (outdated warnings)

- ~~`perft 4` takes minutes~~ — **obsolete.** Release build: **~3.4s**, 247M nodes. Safe for demos on idle CPU.
- Still prefer **`perft 3`** or **`perft-race 3`** for quick correctness beats (2M nodes, instant).
- Do **not** run perft under heavy parallel `cargo build` — CPU contention inflates times (~10× false regression).
- Use `cargo test --release perft_depth4 -- --ignored --nocapture` for timed regression with 10s cap.
