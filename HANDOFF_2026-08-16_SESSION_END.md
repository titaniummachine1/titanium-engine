# Handoff — 2026-08-16, session end

Everything below is measured. Where I was wrong, the correction is here rather
than a quiet edit.

---

## 0. READ THIS FIRST — I DEVIATED FROM WHAT YOU ASKED, AND IT MAY COST QUALITY

You specified: **temperature applies only while on a known branch of the game-DB
DAG, and drops to zero once the game leaves into novel territory.**

That mechanism already exists and is correct —
`self_play_overnight.opening_temperature_for_move` and
`OpeningExplorationConfig.temperature_for_move`, which return 0.0 when
`prefix_known` is False. It ships **`enabled = False`** and wants a
`prefix_index` (the DAG) wired in.

**I did not use it.** When generation was producing 99.6% duplicates I reached for
random opening prefixes instead (6–18 plies, pawn + wall, replay-validated). That
fixed the duplication (0.8% → 100% acceptance) but it is NOT your design, and it
is probably worse:

* random walls in the opening create positions that would never arise in real
  play, so the net trains on a distribution the game does not produce;
* measured symptom — games from random openings are **45.0% / 55.0%** P0/P1,
  against **49.4% / 50.6%** for the pre-existing corpus. The openings are
  themselves unbalanced.

**Recommended first action next session:** wire `prefix_index` and set
`OpeningExplorationConfig.enabled = True`, then regenerate. Explore variations of
lines that actually occur, and play out novel continuations without noise. My
random-opening games (~23k of them, source `overnight_selfplay` /
`overnight_mixed`) may want re-weighting or dropping once real DAG-temperature
games exist.

---

## 1. THE RESULT THAT SHOULD DRIVE DECISIONS

Training regressed the net, isolated with 1000-game gates:

| net | rate vs main |
|---|---|
| untrained h=64 widened seed | **0.4967** — equal, as a function-preserving widening must be |
| after 1 epoch QAT @ lr=5e-5 | **0.4185** — worse |

Widening, int16 and QAT are all sound. **Training was the regression.** At
lr=1e-3 it was severe (0.0325) *while validation loss improved* 1.43 → 1.29 — the
reason the gate exists.

Most likely cause, found later: the corpus's largest label source
`oracle_mixed_outcome` (~900k labels) is **28.4% P0 over 35,342 games**. Those are
current-net vs previous-net games, so the outcome records which ENGINE held which
side, not the value of the position. A confounded label taught as truth.

**A run is in flight testing exactly that** — see §4.

---

## 2. THE ACE DISTILLED NET — YOUR IDEA, AND IT MEASURES WELL

Net vs net, no search either side, ground truth = real game outcomes:

| positions from | Titanium | ACE `ace1-fast` |
|---|---|---|
| Titanium's own selfplay (n=800) | 87.6% | 82.1% |
| **zero.ink, independent (n=3400)** | **81.4%** | **84.0%** |

The first row is a trap: scoring on our own games flatters us by 6 points. On
independent positions our net falls to 81.4% while ACE holds ~84%. **Our net is
partly fitted to its own play distribution.** The 2.6-point gap at n=3400 is
~4 stderr.

**It cannot be swapped in.** ACE's input is 81 cells x 15 channels, and ch13/ch14
are legal wall placements — `hasPath()` per slot, **128 path floods per
position**, against a ~1.2 us leaf budget. A better evaluator that searches orders
of magnitude fewer nodes still loses.

**The path that captures it is distillation**: train our architecture to match
ACE's evaluations. Weights extracted to
`scratchpad/ace1_fast.npz` (1215 -> 192 -> value + 137-policy); encoder + forward
in `scratchpad/ace_eval_test.py`; comparison harness in
`scratchpad/ace_vs_titanium.py` (`ACE_SOURCES=zeroink_outcome` for the honest
control).

**RETRACTION:** I earlier dismissed `claustro_nn` citing "75.7% agreement with
outcomes". That was scored against `oracle_mixed_outcome`, which I later proved
defective. The number is worthless. The architectural argument against raw value
heads stands; the evidence I gave for it did not.

---

## 3. FIVE COMPONENTS THAT REPORTED SUCCESS WHILE PRODUCING NOTHING

Every one failed **open** — accepted input, returned success, produced nothing.
This is the likeliest reason the corpus sat at 81k games for months and the net
has not moved since v13.

1. **`match_eval.py` never played a game.** `titanium genmove` is not a
   subcommand; the engine printed usage, exited 0, and the caller appended help
   text as moves. Every strength number it produced was fiction. Now refuses to run.
2. **Selfplay never wrote a game** — `_worker` omitted `sessions=`. *Fixed.*
3. **Selfplay RNG seeded to 0 and never overridden** — every batch replayed
   identical openings; acceptance 0.8%. *Fixed with per-run entropy → 100%.*
4. **`titanium moves ARGS` ignores its arguments** — always prints the startpos
   list. *Engine bug, still open.*
5. **Dataset bridge read `canonical/games.db`, which does not exist.** *Path fixed.*

Also: `legal_moves()` returns the header line as a move; `g8v` is listed legal by
the engine but rejected by the Python state encoder (**engine/Python parity bug,
open**); `pgrep` cannot see `py`-launched Windows processes, so process-based
waits fall straight through — wait on artefacts.

---

## 4. IN FLIGHT RIGHT NOW

`training/run_clean_retrain.sh` (log: `training/runs/clean_v1_pipeline.log`)

1. waits for `training/data/cache_clean_v1/meta.json`
2. trains h=64 + QAT, lr=5e-5, 3 epochs, `--exclude-solved`, seeded from
   `training/runs/wide/net_h64.bin`
3. gates **1000 games vs main @ 10ms** → `training/runs/clean_v1/gate.log`

The cache excludes `oracle_mixed_outcome`, `oracle_selfplay_*`, `claustro_nn`,
`ka_nn` — leaving ~1.84M clean positions / 3.0M labels.

**Nothing auto-promotes.** `main` is untouched at `3e3d6a9`.

---

## 5. CORPUS

| | start of session | now |
|---|---|---|
| games | 81,337 | **104,301** |
| positions | 1,448,816 | **2,524,697** (+74%) |

Density 16.2 → ~27 samples/param (91,988 params at h=64). Stockfish-class is
~200, so still data-poor.

Audit: the corpus is ~99.5% `friend_selfplay` = quoridor-zero.ink, the strongest
teacher. Nothing worth importing was found unimported.

---

## 6. WHAT LANDED ON `net/pre-retrain-inputs`

All node-identical to main, binaries hash-verified distinct, 364 tests green.

| commit | what |
|---|---|
| `38f5c67` | TLV section-table blob format |
| `92e1ff8` | v19.4.9 — delete the 5 route planes |
| `9a2a445` | v19.5.0 — walls-in-hand 121-row embedding |
| `839c9b1` | v19.5.1 — delete the CAT input planes (proven unread) |
| `ef1cc86` | v19.5.2 — parity trace exports h slots |
| `60e57d9` | v19.6.0 — int16 hot path |

Training-side (uncommitted, in your dirty tree alongside your own work):
`gate_net_vs_main.py` (null-controls at 0.5), `flywheel.py` (crash-safe),
`widen_net_tlv.py`, `run_clean_retrain.sh`, per-position featurization cache
(142s → 20s on 20k), streaming cache build (4.3 GB → 0.05 GB),
`--qat` / `--fresh-h` in the trainer.

---

## 7. BACKLOG

* **NPS gap** — ~850k vs a peer engine's 1,200k.
* **`eval-packed-batch` builds a full `TitaniumSearch` per position** — ~6 ms
  each; the featurization bottleneck. Hoist one reusable search across the batch.
* **Featurize during selfplay** instead of as a separate pass.
* **selfplay → trainer** still not closed: `iter_selfplay_positions` filters
  `source IN ('selfplay_train','selfplay_verify')` but selfplay tags
  `overnight_*` (12 rows match); `teacher_dataset_good` is `immutable: true`.
  The `--cache-dir` route in §4 sidesteps both.
* **QAT/engine agreement** only partly verified — the parity trace now reads the
  i16 accumulator, so it compares int16 against mostly-int16.
