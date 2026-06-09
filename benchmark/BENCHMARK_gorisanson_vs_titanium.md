# Benchmark: Gorisanson vs Titanium αβ + CAT

Use these games as regression benchmarks when changing LMR, mate-zone, or opening search.

---

## Game A — Titanium loses (Black)

**Replay ID:** `tq1#{"winner":"white","plies":71}`  
**Date captured:** 2026-06-09  
**White:** Gorisanson (JS MCTS) · 10s → 500ms from ply 50  
**Black:** Titanium αβ + CAT · 10s · ≤2.0B nodes  
**Result:** White wins · final `White=a9 Black=i4` · path W=0 B=5

---

## Game B — Titanium wins (White) · pre-LMR-fix baseline

**Replay ID:** `tq1#{"winner":"white","plies":55}`  
**Date captured:** 2026-06-09 (before depth-first LMR + eval-spin changes)  
**White:** Titanium αβ + CAT · 10s · ≤2.0B nodes  
**Black:** Gorisanson (JS, original) · **500ms** throughout  
**Result:** White wins · blowout · final `White=g9 Black=b8` · path W=0 B=19 · margin **+19**

```
e2 e8 e3 e7 e4 e6 d1h d6h f1h e5 e6 f6h d6 c5v d5 d4 c1v c4 d4 c3 b2h b5h b3v e1v c3v h8h a4h g5v h2h d7v g4h g7h e4 f2v f4 c4 g4 c5 h4 b5 i4 a5 i5 a6 i6 a7 i7 a8 i8 b8 h8 b7 g8 b8 g9
```

### Game B executive summary

| Area                        | Observed                                                            | Verdict                                       |
| --------------------------- | ------------------------------------------------------------------- | --------------------------------------------- |
| Opening ID depth (ply 1–20) | **d4–d5** @ ~300–500k nodes / 10s                                   | Same bottleneck as Game A                     |
| Winning conversion          | **c3v** ply25 → **+9.57**; **a4h** ply27 → **+11**                  | Excellent — engine finds crushing plan        |
| Won-position eval spin      | **d21–d28** identical PV @ +30.51 (ply37); **d17–d31** spin (ply39) | Bad — wastes 10s confirming already-won lines |
| Mate zone (winning endgame) | **d29→8.1s**, **d20→2.6s**, **d12→586ms**, **d5→64ms**              | **Excellent** — early stop once mate found    |
| Play quality                | i-file march + wall box; Black path **B=19** at finish              | Dominant when eval structure is clear         |

### Game B phase notes

**Opening (ply 1–20):** Still **d4–d5** only. Ply1 five roots tie at **+0.97** (g-10 walls). Eval wobbles after **d5** (ply15: **-3.67**) — tactical volatility not yet triggering any stop.

**Turning point (ply 21–27):**

| Ply | Move    | Depth | Score      | Note                                    |
| --- | ------- | ----- | ---------- | --------------------------------------- |
| 21  | b2h     | d5    | -1.57      | Recovering from -3.9                    |
| 25  | **c3v** | d4    | **+9.57**  | Massive eval jump — box Black on h-file |
| 27  | **a4h** | d4    | **+11.01** | a-file pressure; path W10/B19           |

**Middlegame crush (ply 29–36):** Eval **+13 → +18**. Ply33 **e4** reaches **d17** in 10s (477k nodes) — deep because eval still shifting (+15.69→+16.99). Ply35 **f4** hits **d23** @ **+30.49** with long forced PV.

**Won-position spin (ply 37–41):**

| Ply | Move | Depth   | Score  | Spin pattern                                    |
| --- | ---- | ------- | ------ | ----------------------------------------------- |
| 37  | g4   | **d28** | +30.51 | PV frozen d21–d28                               |
| 39  | h4   | **d31** | +30.37 | PV frozen d17–d28; d29–31 tweak wall order only |
| 41  | i4   | **d31** | +31.89 | Same class                                      |

→ Symmetric bug to Game A ply38 **d53 @ -3.63**: stable eval + cheap marginal depths. **Fixed in engine:** eval-zone stop at `stable_iters ≥ 3 && depth ≥ 12` (no marginal_nodes gate) — **re-validate on replay**.

**Mate sequence (ply 43–55):** Mate scores appear; search stops early:

| Ply | Move | Depth | Mate | Time                           |
| --- | ---- | ----- | ---- | ------------------------------ |
| 43  | i5   | d29   | +M15 | 8.1s (still spinning pre-mate) |
| 45  | i6   | d20   | +M11 | 2.6s ✓                         |
| 47  | i7   | d16   | +M9  | 1.4s ✓                         |
| 49  | i8   | d12   | +M7  | 586ms ✓                        |
| 51  | h8   | d9    | +M5  | 324ms ✓                        |
| 53  | g8   | d5    | +M3  | 64ms ✓                         |
| 55  | g9   | —     | win  | 36ms                           |

### Game B strategic observations (Titanium as White)

1. **c3v + a4h** is the winning idea — walls box Black while White marches i-file.
2. **d6** (ply13) and **d5** (ply15) were locally dubious (-0.3 → -3.7) but recovered once structure clarified.
3. **h2h / g4h** (ply29–31) accelerate path lead without wasting last wall too early.
4. When eval > +20, engine keeps deepening to **d28–d31** instead of playing fast — time better spent on opening prep in other games.

### Game A vs Game B comparison

| Metric          | Game A (Ti Black, lost) | Game B (Ti White, won) |
| --------------- | ----------------------- | ---------------------- |
| Opponent budget | 10s → 500ms             | 500ms always           |
| Opening depth   | d4–d6                   | d4–d5                  |
| Eval-spin waste | Losing: d53 @ -3.63     | Winning: d28–d31 @ +30 |
| Mate zone       | d9–d14 (good)           | d5–d20 (excellent)     |
| Outcome         | Lost a-file race        | Blowout +19 path       |

**Takeaway:** Search infrastructure (mate zone) works in both directions. Opening depth and eval-spin are **color-agnostic bugs**. Game B proves Titanium can crush 500ms Gorisanson when eval structure is found — Game A shows it can still lose the same race when Black.

---

## Game A executive summary

| Area                        | Observed                                   | Target / verdict                        |
| --------------------------- | ------------------------------------------ | --------------------------------------- |
| Opening ID depth (ply 2–20) | **d4–d6** @ ~300–520k nodes / 10s          | Plan target d40–d80 — **not met**       |
| Mate-zone (losing endgame)  | **d9–d14** @ -M2…-M8                       | Good — no d200+ spin                    |
| Midgame eval-flat spin      | **d53 @ -3.63** (ply 38)                   | Bad — same PV, wasted 10s               |
| Horizon mate garbage        | **-M48 @ d80** (ply 40)                    | Bad — should clamp / ignore             |
| Play quality                | Lost a-file race despite +4.5 eval midgame | Strategic errors + shallow opening prep |

---

## Think-chain observations by phase

### Opening (ply 2–20)

- Titanium consistently **d4–d6**, full **10.00s**, **~288k–520k nodes**.
- Root candidates often **tie on score** (e.g. ply2: five walls at **-0.97**, all **g-10**).
- Book hints order moves; search does **not** skip — only 4 ID rounds fit in budget.
- **~75k nodes / depth / ~2.5s per depth** → tree still too heavy per iteration even with aggressive LMR.

**Benchmark expectation after LMR fix:** opening `searchDepth ≥ 30` in 10s (CI floor); stretch goal d40+.

### Early middlegame (ply 6–14)

- Black eval **+4.47 → +4.87** (ply 6–8) — engine knew it was ahead.
- Moves: reactive walls (`g2h`, `h7h`, `f7h`) with small path impact.
- Depth ramps slowly: **d5–d6** by ply 16.

**Mistake class:** failure to **convert eval lead** into race/path pressure; walls don't shorten Black path or lengthen White's.

### Critical middlegame (ply 24–36)

| Ply | Move | Depth | Score     | Note                                 |
| --- | ---- | ----- | --------- | ------------------------------------ |
| 24  | e6   | d9    | +3.15     | Long PV pawn march                   |
| 28  | e5   | d13   | **-2.57** | Eval cliff — same pawn push repeated |
| 32  | e5   | d13   | **-5.07** | Confirms losing plan                 |
| 34  | e6   | d16   | -6.03     | Deep search, still wrong             |
| 36  | e5   | d19   | -4.83     |                                      |

**Mistake class:** **e5/e6 pawn churn** in a lost race structure; deep search finds forced lines but **root move choice** doesn't respect White's a-file threat.

### Depth-spin bugs (non-mate)

**Ply 38 `e6`:** **d53** @ **-3.63**, PV identical from d33→d53, ~884k nodes / 10s.  
→ Eval stable, marginal nodes tiny per depth after d33 — **mate-spin analogue needed for eval**.

**Ply 40 `e7`:** **d80** @ **-M48** — horizon mate (dist 48 > trusted 64? actually 48 is trusted — but -M48 is likely PV artifact). Wasted depth before time ended at 8.36s.

### Endgame (ply 64–70) — mate zone working

| Ply | Move | Depth | Mate | Stop OK?    |
| --- | ---- | ----- | ---- | ----------- |
| 64  | h6   | d14   | -M8  | ✓ (~dist+6) |
| 66  | h5   | d10   | -M6  | ✓           |
| 68  | h4   | d9    | -M6  | ✓           |
| 70  | i4   | d6    | -M2  | ✓           |

Compare to pre-fix tq1: **d214 @ -M8** — major win for mate controller.

---

## Strategic mistakes (Black / Titanium)

1. **Ignored White's a-file sprint** — White plays `a2h` (ply 11) and never gets blocked on the a-file.
2. **Edge walls without path gain** — `g2h`, `h7h`, `f7h` (g-10 or g+2 only).
3. **Pawn pushes e5/e6** when eval turns negative — engine searches deep but picks **local pawn activity** over race walls.
4. **Failed to convert +4.5** at ply 6–8 — no move shortened Black path to goal by ≥2.
5. **Lost to 500ms Gorisanson** from ply 50 — position already strategically lost.

---

## Engine / search hypotheses

### Why opening stays d4 despite aggression 2.0

1. **Aggressive LMR → more fail-high re-searches** → node budget burns on corrections, not new depths.
2. **Root never LMR'd** — every ID ply pays full root width (~20–60 moves post-CAT).
3. **`depth_balance_floor: 70`** pushes aggression up when marginal_nodes < 500 — may **increase** re-search cost without completing depth 5+.
4. **High `cat_heat_lmr_slope` (0.03)** on wide opening move lists — cold walls get slashed but re-search restores them.

### User direction (2026-06-09)

> Default **deep preparation**; **widen / shallow ID only** when tactically difficult (eval swinging, complex CAT) or when **forced mate** found (refine band).

Inverted from current plan: start **gentle LMR (baseline)**, push depth only when eval stable + time left; widen on eval volatility.

---

## Positions to pin in `capture_baseline.mjs`

| Key                  | Moves prefix                 | What to measure                   |
| -------------------- | ---------------------------- | --------------------------------- |
| `opening_ply0`       | (empty)                      | depth @ 10s — currently ~d4       |
| `opening_ply4`       | e2 e8 e3 e7                  | early book                        |
| `wall_heavy`         | …14 walls                    | middlegame t                      |
| `tq1_ply38_spin`     | Game A replay through ply 37 | should stop < d40 @ stable -3.63  |
| `tq1_ply37_won_spin` | Game B replay through ply 36 | should stop < d20 @ stable +30.51 |
| `tq1_lost_ply69`     | Game A …f5                   | mate stop ≤ dist+6                |
| `tq1_won_ply43_mate` | Game B …a6                   | mate stop ≤ dist+6, time < 3s     |

---

## Settings changelog in this game (White only)

- Ply 35–37: Gorisanson visits ramped 66k → **2.0B**
- Ply 50: Gorisanson time **10s → 2s → 500ms**

Titanium settings unchanged throughout.

---

## Next engineering tasks

1. **LMR default → depth-first baseline** (aggression ~1.0); push aggression only when depth stalls _and_ eval stable.
2. **Widen LMR** on eval swing / high `stage_t` / aspiration storm.
3. **Eval-spin guard** — stable score 3+ iters + low marginal_nodes → stop ID (ply 38 case).
4. **Horizon mate clamp** — `|mate_dist| > 64` or score delta per depth huge → don't pump depth.
5. Re-run `capture_baseline.mjs` after changes; compare to this doc.
