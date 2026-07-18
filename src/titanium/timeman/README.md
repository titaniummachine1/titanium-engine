# timeman

## Purpose

Move-time allocation and clock heuristics.

## Owns

`time_alloc` (TM budgets, length bounds used as information).

## Uses

Endgame distance facts when budgeting; does not prune the tree.

## Public API

`allocate_move_budget*` and related — via `titanium::time_alloc`.
