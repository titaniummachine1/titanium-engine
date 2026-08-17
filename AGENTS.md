# Titanium Engine — Agent Rules

## CRITICAL: This repo is strength-only

This repo holds the engine and exactly what builds, tests and benchmarks it.
It is the Stockfish rule: sources, build, tests, benches, engine docs. Nothing
that describes a *session*, a *training run*, or a *result* lives here.

| Belongs here | Belongs in the workspace above |
|---|---|
| `src/`, `benches/`, `examples/`, `tests/` | session notes, handoffs, briefs → `../archive/engine-handoffs/<yyyy-mm>/` |
| `Cargo.*`, `build.rs`, `.cargo/` | training pipeline and its run output → `../training/` |
| `docs/` — how the engine works | bench dumps, match results, Elo → `../results/`, `LEDGER.md` |
| `scripts/` — bench/measure the engine | built binaries and gate `.exe` → not versioned at all |

Enforced, not requested: `tests/repo_hygiene.rs` reads `git ls-files` and
**fails `cargo test`** if any of the right column is tracked here, naming the
file and where it should go. `.gitignore` stops the accidental `git add`; the
test stops the deliberate one. If something genuinely is part of the engine,
add it to `EXEMPT` in that file with a comment saying why.

## CRITICAL: No legacy versions in the live engine

The engine repo is **always and only** the latest strongest build. See `.devin/rules/no-legacy-versions.md` for the full policy.

**Short version:**
- Never add old version constructors to the live engine
- Never `git checkout <old-commit>` in the live engine repo
- To test old vs new: use a git worktree at the tagged commit
- Every version that ships to the website gets a git tag + release artifact (exe + wasm)
- The live engine has exactly ONE new constructor per version promotion. `new_v17` and `new_v18` are frozen references — do not delete, do not add more old ones.

## Build commands

```powershell
# Native build (always use native CPU features)
$env:RUSTFLAGS = '-C target-cpu=native'
cargo build --release -p titanium

# WASM build (from site/web/)
cd site/web
npm run build:wasm

# Tests
$env:RUSTFLAGS = '-C target-cpu=native'
cargo test --release -p titanium --lib
```

## Repo layout

- `C:/gitProjects/Quoridor best AI/engine` — live engine repo (always latest, on `main`)
- `C:/gitProjects/Quoridor best AI/site/web` — website + WASM build scripts
- `C:/gitProjects/Quoridor best AI/artifacts/worktrees/` — isolated worktrees for old version builds
- `C:/gitProjects/Quoridor best AI/artifacts/releases/` — release artifacts per version
