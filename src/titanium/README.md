# titanium

Production Titanium engine (v15+ play path).

## Purpose

Play Quoridor: search, eval, endgame proofs, time management, UCI.

## Owns

Layered modules under this directory (see façades). Stable crate paths like `titanium::race` are **compatibility re-exports** — ownership is in the façade folders.

## Uses

Core / movegen / pathfinding / cat (Layer 0).

## Public API

`run_titanium_session_stdio`, `TitaniumSearch`, `GameState`, race/certify surfaces, move-id helpers — see `mod.rs` re-exports.

## Laws

See [`docs/architecture.md`](../../../docs/architecture.md): Rules #1–#3, layers, ExactDP boundary, training outside engine.
