# DAG tight-edge LMR classifier — GATE REJECT

**Verdict: REJECT. Do not merge. Default stays off.**

## Gate

Promotion harness `tools/binary_match/parallel_engine_match.py`, 100 games,
60s/side sudden death, 8 workers, book openings, out-of-process arms from one
binary.

    A = titanium-v18-dag-lmr    B = titanium-v18
    A 41W 0D 59L / 100    score 0.410    wilson_lb 0.319
    ~ -63 Elo  [-136, +4]

## What the pre-gate numbers said, and why they were wrong

    fixed depth 12   119,762 -> 49,207 nodes   (2.43x fewer), same move, same score
    fixed 1500ms     +1 depth in 2 of 3 positions, same move everywhere

Every one of those favoured the candidate. It still lost 41-59. This is the
cleanest instance yet of the standing rule that node efficiency and depth are
not strength: a 2.4x narrower tree reached the same answers on smoke positions
and played worse in games. The narrowing was over-pruning.

## Two confounds — this does NOT clear the classifier itself

The reference project gates this exact classifier at E-011 **+55.2** [34.2,76.7]
(n=1028). Two documented differences between their base and ours:

1. **Reduction strength.** Theirs is `LMR r=1` — reduce one ply, verify
   re-search. Ours routes through `plan_v16_wall_lmr`'s hard override:
   `final_reduction = max_safe`, `child_depth_used = 1`. Cut to depth 1 is a far
   more aggressive mechanism than r=1, and it was applied to a classifier that
   fires on many more walls than the 10%-CAT-attention test it replaced. That
   combination is very likely the over-pruning.

2. **Killers.** Their E-012 retest of killers on the LMR base measured
   **-27.9** [-46.6,-9.3] and states the mechanism: killers "disrupt the ordering
   LMR's reduce-set depends on." Their +55.2 was measured with killers OFF.
   Titanium has killers live at 4 sites in the move loop.

So the honest claim is narrow: **this wiring loses ~63 Elo.** Whether the
classifier pays on a base with r=1 reduction and/or killers off is untested.

## Kept

The primitive and its soundness oracle stay — they cost nothing when the flag is
off and they are the same object `slack.rs` and CAT v8 need:

- `pawn_shortest_edges` — wall-slot touch masks from a descent over the
  goal-distance field `refresh_dist` already maintains; no flood.
- `dag_untouched_wall_cannot_change_that_players_distance_oracle` — 600 random
  games, every legal wall probed both sides, zero violations.
- `titanium-v18-dag-lmr` session arm, for re-gating either follow-up.

## Follow-ups, in the order their expected value ranks

1. **wall inventory cost.** Titanium has no `wall_cp` equivalent; its only wall
   term is the learned `ws[2] * wd`. Reference E-008 measured a hand-set wall
   reserve at **+180 Elo** [132,235] with a sharp threshold at ~1 tempo
   (sweep: cp 90 -> 68.5%, cp 120 -> 92.0%, cp 200 -> 100% vs greedy). Largest
   single item in their ledger; untested here.
2. DAG classifier with `r=1` reduction instead of the depth-1 hard cut.
3. DAG classifier with killers disabled (tests E-012's stated interaction).

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
