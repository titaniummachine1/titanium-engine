# endgame

## Purpose

Mathematically exact endgame logic: race proofs, win certificates, and the ExactDP reference table.

## Owns

- Race (jump-aware / race theorems) — production
- Certify (`cert_win`) — production
- ExactDP (`exact_dp`) — validation reference
- Cert bridge (race ↔ certify checks)

## Uses

Position (board state) and Core via Position.

## Public API

`race` bounds / `RaceBound` / jump-aware distances; `certify` / `cert_win`; re-exported at `titanium::` for stable callers.

## Must not

Search must not call ExactDP — it is an exponential reference implementation intended for validation/tests/benchmarks only; calling it from search would steal clock and is not a production proof path.

Validation/tests/benches may use ExactDP.
