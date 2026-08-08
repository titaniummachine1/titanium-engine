# Titanium Engine — Agent Rules

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

## Canonical 100-game gate

Use `tools/binary_match/launch_broke_side_ab_100.ps1` as the canonical 100-game
A/B launcher; do not invent a new SPRT protocol. Its local match settings are:

- 100 games, 60 seconds per side, seed `20260717`
- 8 plies maximum opening prefix, 12-ply book cap
- Claustrophobia human openings plus the configured extended DAG book
- deterministic mirrored opening pairs: each selected opening is played once per color order
- 8 local workers when running the local gate
- one engine thread per client (`--engine-threads 1`)
- `titanium-v17` session routing
- native `go rem` time management for both engines
- pondering disabled: the routed `titanium-v17` session does not expose the unrouted experimental `go infinite` protocol

The canonical harness is `tools/binary_match/parallel_engine_match.py`; its
`preassign_openings()` function is the source of truth for deterministic pairing.
