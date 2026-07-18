# eval

## Purpose

Static evaluation: NNUE / HalfPW and eval-facing features.

## Owns

NNUE (`nnue`, also aliased as `titanium::net`), eval distance helpers, field planes / viz.

## Uses

Position / Core.

## Public API

`nnue` / `net`, `dist`, `field_planes`, `fields_viz`.

## Must not

Eval must not set aspiration windows or prune — it returns scores; Search decides (Rule #3).
Jump-aware race distance lives in `endgame/race`, not here (Rule #2).
