# Titanium Quoridor

**Titanium Engine** — Quoridor AI (**iterative-deepening αβ + CAT corridor pruning**) with a reverse-engineered play UI. Legacy MCTS path is deprecated (routes to negamax).

Repo: [github.com/titaniummachine1/titanium-quoridor](https://github.com/titaniummachine1/titanium-quoridor)

## Layout

| Path         | Purpose                                                                               |
| ------------ | ------------------------------------------------------------------------------------- |
| `engine/`    | **Titanium** — Rust search core (in development)                                      |
| `web/`       | Playable UI (scraped from [quoridor-ai.netlify.app](https://quoridor-ai.netlify.app)) |
| `scraped/`   | Deobfuscated extracts + raw bundle archive                                            |
| `extracted/` | Protocol docs + WebSocket client                                                      |
| `benchmark/` | Rust Titanium vs Gorisanson / self / Ka / Ishtar                                      |

## Quick start — web UI

```bash
cd web
npm install
npm run dev
```

Play Human vs **Ishtar** or **Ka** (remote WebSocket engines).

## Quick start — Titanium engine (Rust)

```bash
cd engine
cargo build --release
cargo test
cargo run --release -- perft          # depth 3 (default) → 2_062_264 nodes
cargo run --release -- divide         # divide at depth 3
cargo run --release -- bench 2 20
cargo bench
```

Benchmarks (build `engine` release binary first):

```bash
cd engine && cargo build --release && cd ..
node benchmark/titanium_vs_gorisanson.mjs --games 1
node benchmark/titanium_vs_titanium.mjs 1
node benchmark/titanium_vs_ka.mjs --games 1
node benchmark/titanium_vs_ishtar.mjs --games 1
```

## Engine roadmap

1. **Phase 1** — Board, eval (dual BFS), iterative deepening αβ, Zobrist TT, aspiration windows ✅
2. **Phase 2 (current)** — Adaptive LMR, mate/eval zone ID stops, CAT v3 prune, opening depth
3. ~~Guided MCTS hybrid~~ — **deprecated** (`search/deprecated/mcts.rs`); all `genmove` routes to negamax
4. **Bench** — vs Gorisanson MCTS JS, vs Ishtar@Short (external exam)

## Documentation

| Doc                                                                    | Purpose                                                       |
| ---------------------------------------------------------------------- | ------------------------------------------------------------- |
| [docs/STATE.md](docs/STATE.md)                                         | **Session handoff** — current status, what broke, what's next |
| [docs/video/README.md](docs/video/README.md)                           | Video episode index + git checkpoints                         |
| [docs/video/11-search-hardening.md](docs/video/11-search-hardening.md) | Latest arc: negamax bugs, CAT gaps, qsearch, CAT UI           |
| [docs/video/BUG-DIARY.md](docs/video/BUG-DIARY.md)                     | Chronological plot twists for recording                       |

Analysis mode: toggle **CAT** on the board to see raw corridor heat (cm) from the engine.

## References (ideas only — not ports)

- [gorisanson/quoridor-ai](https://github.com/gorisanson/quoridor-ai) — MCTS heuristics
- [pavlosdais/Quoridor](https://github.com/pavlosdais/Quoridor) — αβ + TT
- [quoridor-ai.netlify.app](https://quoridor-ai.netlify.app) — UI + wire protocol scrape

## License

Engine: TBD. Web scrape artifacts and third-party references retain their original licenses.
