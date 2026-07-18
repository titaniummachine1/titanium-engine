# search

## Purpose

αβ search, iterative deepening, TT, LMR/EME, and wiring of race/certify cuts.

## Owns

Search decisions (prune, aspirate, extend, reduce). The large `search_impl` file is intentionally unsplit in v1.0.

LMR helpers (`v16_lmr`, `cat_index_lmr`) and TT cache-tier sizing (`tt_sizing`) live in this folder next to the play engine.

## Uses

Eval (scores), Endgame race/certify (proofs/bounds), Position, Timeman, Opening, Core.

## Public API

`TitaniumSearch`, `ThinkResult`, think helpers — via `titanium::search` and re-exports.

## Legacy

Historical αβ / CLI / perft TT lives in `engine/legacy/search/` (not under `src/`, not this `search/`). Do not put new play-search code there.

## Must not

Must not call ExactDP or `validation::`. Must not put race theorems here — Endgame owns them.
