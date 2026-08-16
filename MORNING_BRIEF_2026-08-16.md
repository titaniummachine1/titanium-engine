# Morning brief — 2026-08-16

Everything here is measured. Where a claim was wrong, the correction is recorded
rather than replaced.

---

## 1. THE ONE RESULT THAT MATTERS

Training on the current corpus makes the net **worse**, and it is not the
architecture's fault. Isolated with a 1000-game gate:

| net | rate vs main |
|---|---|
| untrained h=64 widened seed | **0.4967** — equal, exactly as a function-preserving widening must be |
| after 1 epoch QAT @ lr=5e-5 | **0.4185** — clearly worse |

So widening, int16 and QAT are all sound. **The regression is training itself.**
The shipped net is v13-derived and already fitted to this exact zero.ink corpus;
further passes only drift it. A smaller learning rate shrinks the damage, it does
not make it positive.

At lr=1e-3 the damage was severe: the net opened with a **wall move** (the tell
`search_impl.rs` documents) and scored **0.0325**. Meanwhile val loss *improved*
1.43 -> 1.29. A metric that improves while strength collapses is why the gate
exists.

**Conclusion: the corpus is exhausted for this net. More Elo has to come from new
positions, not more passes.**

---

## 2. WHY THE NIGHT WENT TO GENERATION

Data audit (all 17 game sources, all label sources):

- corpus is **~99.5% `friend_selfplay` = quoridor-zero.ink**, the strongest teacher
- everything unimported anywhere totals **+9% new positions** (1,538,424 distinct
  in store vs 1,411,901 in the dataset)
- `claustro_nn` is the largest label source (1,144,460) and adds exactly **1** new
  position; it is a raw AZ value head (~one rollout), weaker than Titanium's own
  search, and disagrees with real outcomes 24% of the time. **Not imported.**

Data density is ~**16 samples/parameter** (91,988 params at h=64) against a
Stockfish-class ~200. The shortage is new positions, and only generation fixes it.

---

## 3. FOUR MORE SILENT FAILURES FOUND

Each accepted input and quietly did something else, so everything downstream
reported success on nothing.

1. **`_worker` never passed `sessions=`** — every selfplay game used the dead
   `genmove` path. *Fixed.*
2. **`titanium moves ARGS` ignores its arguments** — always prints the startpos
   list. An opening generator querying it per ply builds illegal sequences like
   `[e2, e8, e2]`. *Worked around; engine bug still open.*
3. **`legal_moves()` returns the header line** `"131 legal moves at startpos"` as
   if it were a move. *Worked around.*
4. **`g8v` is listed legal by the engine but rejected by the Python state
   encoder** — a genuine engine/Python parity disagreement. *Open.*

Openings are now computed from board geometry, no engine dependency.
Before: every game `plies=27`, **99.6% rejected** as duplicates.
After: 20/20 accepted, 20 distinct lines; against the live DB, **44% acceptance**
(rejects are collisions with the 62,839 stored line hashes).

---

## 4. RUNNING NOW

Selfplay generation, 8 threads, 10 ms/move, batches of 1000.

**Sustained rate, measured over a 2-minute window (not the opening burst):
31 games/min, 1,430 positions/min** — about 1,860 games/hr. An earlier note in
this file said ~3,700 games/hr; that was the first-minutes rate and was roughly
double the truth.

Projected to 08:00: **+8,127 games, +369,069 positions**, taking the corpus from
1.49M to ~1.86M positions (+25%) and density from 16.2 to ~20.3 samples/param.

### Acceptance: a deterministic-seed bug, NOT collision decay

I spent hours reading falling acceptance as "the line-hash store is filling up".
That was wrong. `run_batch_streaming(seed: int = 0)` and `main()` never passes a
seed, so every batch built `random.Random(0 + game_idx)` — **game i drew the
identical opening in every batch.** Batch 1 stored ~1000 unique games; every
batch after it regenerated the same 1000 openings and was deduped away.

  - pawn-only openings: 44% -> 7.1%
  - pawn + random walls, replay-validated: 19.1%
  - **after seeding the RNG with per-run entropy: 100%** (37 accepted, 0 rejected)

The diversity work was real but secondary; the seed was the whole story. Isolation
that settled it: 12/12 games accepted into a VIRGIN database while the live DB was
rejecting 99.2% — same code, same engine, so the rejects had to be duplicates
rather than illegal play.

Lesson for the next person: acceptance collapsing toward zero in a generator is a
determinism smell first, a diversity problem second.

### Result as of ~07:30

| | start of night | now | delta |
|---|---|---|---|
| games | 81,337 | **91,246** | +9,909 |
| positions | 1,448,816 | **1,837,076** | +388,260 (**+26.8%**) |

Data density 16.2 -> **20.0 samples/param** (91,988 params at h=64).

Post-fix throughput is ~2,540 positions/min and steady across four intervals.
Pre-fix it was ~470/min and falling. **The hour after the seed fix produced more
usable data than the five hours before it**, which is the clearest possible
statement of how much that bug cost.

### Data-quality landmine found while spot-checking

Outcome balance by source (decisive games only):

| source | P0 | P1 | n |
|---|---|---|---|
| tonight, random openings | 45.0% | 55.0% | 4,895 |
| pre-existing corpus | 49.4% | 50.6% | 79,498 |
| **`oracle_selfplay`** | **99.7%** | **0.3%** | **14,069** |
| `pool_selfplay` | 48.5% | 51.5% | 12,608 |

**`oracle_selfplay` is 99.7% P0 wins across 14,069 games** — 17% of the game
store. Quoridor's first-player edge is a few percent, not 99.7%, so this is a
systematic defect in that source, not a property of the game. It is NOT in
`teacher_dataset_good` (which is 99.5% friend_selfplay), but **any corpus rebuild
from `games_turso.db` would pull it in and teach the net a huge first-player
prior.** Filter or fix it before rebuilding.

Tonight's own 45/55 skew is milder and is probably the random wall openings
occasionally trapping the first player; worth watching, not alarming.

Nothing is being trained or promoted. `main` is untouched at `3e3d6a9`.

---

## 5. FIRST THINGS TO DO

1. **Close selfplay -> trainer.** Two decisions, both yours:
   - `iter_selfplay_positions` filters `source IN ('selfplay_train','selfplay_verify')`
     but selfplay tags `overnight_selfplay` / `overnight_mixed` — only 12 rows match.
   - its output `teacher_dataset_experimental_extended` is deliberately NOT active;
     `teacher_dataset_good` is `immutable: true`.
   Until both are settled, generated games bank but never train.
2. **Then retrain on the enlarged corpus** — this is the first time there will be
   genuinely new data to learn from.
3. **A truly independent int16-vs-f64 check.** What I ran is weaker than it
   looks: quantizing the hot path also made the parity trace read back the i16
   accumulator, so it compares int16 against *mostly*-int16. Results were
   shipped/widened nets 0.00 cp (12/12 exact), QAT net 0.17 cp mean / 1 cp max —
   which validates the OUTPUT layer and the QA/QB scaling, not total quantization
   cost. Total cost, measured properly by building both ways: 5/6 fixed-depth
   positions identical, `low-wall` move changed e6 -> e3v. To do this right, keep
   an f64 accumulator path available purely as a reference.
   Also counterintuitive: the QAT net deviates MORE than non-QAT ones, because its
   weights sit near grid boundaries where the two paths round differently.
4. Backlog in `HANDOFF_PRE_RETRAIN_INPUTS.md` §8: NPS gap (~850k vs a peer's
   1,200k), `eval-packed-batch` allocating a full search object per position,
   featurizing during selfplay instead of as a separate pass, QAT/engine
   agreement unverified.

---

## 6. WHAT LANDED ON `net/pre-retrain-inputs`

All node-identical to `main`, binaries hash-verified distinct, 364 tests green.

| commit | what |
|---|---|
| `38f5c67` | TLV section-table blob format |
| `92e1ff8` | v19.4.9 — delete the 5 route planes |
| `9a2a445` | v19.5.0 — walls-in-hand 121-row embedding |
| `839c9b1` | v19.5.1 — delete the CAT input planes (proven unread) |
| `ef1cc86` | v19.5.2 — parity trace exports h slots, not MAX_NET_H |
| `60e57d9` | v19.6.0 — int16 hot path |

Plus, in `training/`: a gate that null-controls at 0.5 (`gate_net_vs_main.py`),
a crash-safe flywheel (`flywheel.py`), a TLV-aware widener (`widen_net_tlv.py`),
per-position featurization caching (142s -> 20s on 20k positions), and `--qat`
and `--fresh-h` in the trainer.

**`match_eval.py` now refuses to run.** It never played a game: `titanium genmove`
is not a subcommand, so the engine printed usage and the caller appended help text
as moves. Every strength number it ever produced was fiction.
