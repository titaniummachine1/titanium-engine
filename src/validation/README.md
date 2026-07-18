# validation

## Purpose

External fixtures and checkers that never participate in production search.

## Owns

Canta perft fixture replays (`canta`).

## Uses

Core board / movegen (to replay fixture games).

## Public API

`canta::board_after_canta_game` and related helpers.

## Must not

Search must not import this module — validation cannot affect Elo.
