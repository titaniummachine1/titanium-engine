# Pre-retrain input cleanup — handoff

Branch `net/pre-retrain-inputs`, based on `main` at `3e3d6a9`. Three commits, all
node-identical against `main`, none pushed yet.

Everything below is measured. Where an earlier document was wrong, the correction
is recorded rather than quietly replaced.

---

## 1. THE FINDING THAT REFRAMES THE RETRAIN

**The shipped net is ACE v13.** Not "descended from" — numerically.

| comparison | result |
|---|---|
| live vs **v17** | **99.06% bit-identical**; the only 405 differing weights are the 5 route planes, which live zeroes |
| live vs **v13 frozen** | pearson r = **1.0000**, relative L2 drift **0.003** |
| v17 vs **v13 frozen** | r = 1.0000, drift 0.003 |

`w1c`, `po`, `px`, `b1`, `w2` are **bit-identical between v17 and the live net**.
The lineage is v13 → a 0.3% nudge → v17 → route planes switched off → today.
Architecture never moved: `h = 32`, single hidden layer, HalfPW, all four blobs.

Every Elo in v18 and v19 came from search. The net side is essentially untouched
since v13, which is the argument for retraining from scratch rather than
fine-tuning again — and it means the headroom here is real.

Reproduce: read the u64 `h` header, slice by the layout in `nnue.rs`, diff per
plane. Do not infer "the net improved" from a version bump.

---

## 2. WHAT LANDED

| commit | what | evidence |
|---|---|---|
| `38f5c67` | Section-table (TLV) blob format, cherry-picked from the experiment branch | node-identical on 5 positions, 359 tests |
| `92e1ff8` | v19.4.9 — delete the 5 route planes and the eval that read them | node-identical on 6 positions, 359 tests |
| `9a2a445` | v19.5.0 — walls-in-hand 121-row embedding | node-identical on 6 positions, ~0.8% NPS, 364 tests |
| `839c9b1` | v19.5.1 — delete the CAT input planes | node-identical on 6 positions, 364 tests |

Node identity throughout is fixed depth 8, single thread, on `startpos`,
`wall-maze`, `dense-maze`, `c3h-midgame`, `endgame-c5`, `low-wall`, comparing
nodes/move/score — **with the two binaries verified distinct by hash first**.

### Why TLV had to come first

`main` could not ship a new weight array at all. The loader was positional with an
eight-variant size-sniffing cascade, and adding walls-in-hand would have made it
ten. The section table makes missing sections zero-fill and unknown sections
ignore, so a trainer can add a tail without a Rust change and an old blob keeps
loading. The route deletion and the walls-in-hand input are both cheap on top of
it and were both painful before it.

### The route-plane correction

The previous handoff recorded these planes as never-learned scaffolding. **That
was wrong.** They were learned in v13, carried through v17 at magnitude ~0.05, and
then deliberately zeroed. "Trained then switched off" is a different fact from
"never worked" — if the route signal is ever wanted back it must be re-derived,
not un-zeroed.

One subtlety worth keeping: the legacy reader still **skips** the 405 f64 rather
than dropping them. They sit at a fixed offset between `px` and the CAT tail, so
not stepping over them would decode every later section from the wrong place.
Skipping is not the same as deleting in a positional format.

### The walls-in-hand input

121 rows indexed `(walls_me * 11 + walls_opp)`, side-to-move canonical, summed
into the hidden layer alongside `po`/`px`. Joint rather than two scalars because
the value of holding a wall is entirely conditional on how many the opponent
holds — eight-to-two and two-to-eight are not two points on one axis. At `h=32`
that is 3,872 weights, 9% of the net.

Inert until trained: shipped blobs carry no `WH__` section, so it loads zero-filled
and `x + 0.0` is exact.

---

## 3. CAT: DECIDED, ON A MEASUREMENT

**The CAT input planes are provably unread.** The live blob has 325 nonzero CAT
weights; zeroing all of them changes nothing across five positions at depth 8 —
not one node, move, or score. The CAT eval block was already removed from
`evaluate()`; only diagnostics still touch the planes.

They are the *same class of dead weight the route planes were*. **Deleted in
`839c9b1`.** The legacy reader skips the tail, and unlike the route planes the
skip LENGTH varies by variant — 1 plane (v5), 3 (witness), 5 (normalized) — so the
count is derived from the same length flags that detect the variant. Getting that
wrong would have silently misaligned the dist tail behind it.

**The net now carries no input the engine does not read.**

**Do not confuse the two CATs.** These are the five CAT *net-input* planes.
CAT-the-heatmap used for LMR move ordering is well calibrated and must stay.
Diagnostic consumers live at `search_impl.rs:3457`, `:3540`, `:3714`.

---

## 4. QUANTIZATION, MEASURED

f64 AVX2 vs i16 AVX2, both hand-vectorized, so this compares optimized against
optimized rather than against whatever LLVM happened to emit.

| width | infer | prep | weighted (prep 8.1%, infer 3.6%) |
|---|---|---|---|
| h=32 | 4.28x | 2.13x | **2.26x** |
| h=64 | 5.02x | 2.65x | **2.81x** |
| h=128 | 4.91x | 2.96x | **3.10x** |
| h=256 | 7.47x | 5.39x | **5.59x** |

Against today's shipped baseline (f64, `h=32`):

- **i16 at h=64** — 2x width and NNUE gets **35% cheaper**
- **i16 at h=128** — 4x width for **+16% NNUE time** (= +1.9% of total search) at
  an *identical* 288 KB `w1c` footprint

### The trap, recorded because I fell in it

My first benchmark measured int16 as **slower** (0.69x). That was my design, not a
result: I used an **i32 accumulator**, and i16→i32 sign-extension costs more than
the extra lanes buy, while f64 elementwise already vectorizes 4-wide.

**The accumulator must stay i16** with saturating adds (16 lanes), widening only
at the output dot product via `madd_epi16`. Use i32 and the entire win evaporates.

Two things already in our favour: the activation is `clamp(0.0, 1.0)`, a clipped
ReLU, so the fixed-point range is known by construction rather than calibrated;
and widening needs no Rust change at all — `h` comes from the blob header,
`MAX_NET_H = 256`, and `training/tools/net2net_widen.py` exists.

Quantization must be quantization-aware from the **first** training step. Post-hoc
quantization of an f64-trained net is where this normally fails.

---

## 5. WHAT IS STILL OPEN, IN ORDER

1. **Update the trainer** — `training/titanium_training/`. This is now blocking.
   - `models/eval_forward.py` and `training/trainer.py` still compute `route_out`;
     the Rust parity JSON no longer emits that key, so
     `tests/test_trainer_scalar_parity.py` will **fail loudly**. That failure is
     the intended signal, not collateral.
   - `models/halfpw.py` has its *own* length-based variant sniffing
     (`route_only_f64s`, `cat_v5_f64s`, …) that does not know `DIST` or `WH__`
     exist. It should read the section table instead of re-deriving lengths.
   - The trainer must emit `WH__` and stop emitting route and CAT planes.
   - **Retraining without this recreates exactly the bug we just deleted**: a net
     that spends capacity on inputs the engine never reads.
2. **Decide the teacher before spending the retrain** (§7).
3. **Quantize and widen** (§4) — QAT from step one, i16 accumulator, `h=128`.
4. **Leave the distance input alone.** The plain wall-only flood is 0.034 plies
   from truth on average; the adversarial version is the better *bound* and the
   worse *feature*. Do not swap them.

## 6. HYGIENE DONE

- `perf/adversarial-bitboard-flood` had 745 lines of uncommitted diff. Proved
  semantically null — **identical sorted identifier multisets across all 28
  files** — and discarded. Patch banked in the session scratchpad if ever wanted.
- `.playwright-mcp/` and `claustrophobia_app.js` are gitignored. The JS file is
  8,710 lines and is **not** a duplicate of any sibling copy (the nearest,
  `.local/claustrophobia-dev/app.js`, is 7,482), so it was not deleted.


---

## 7. THE TEACHER QUESTION (opened 2026-08-15)

`~/Downloads/ace_full_v3.html` (14 MB) embeds two complete nets:

- **`ace1-weights`** — AZ-style trunk, 18 layers, 128 filters, self-attention,
  **1,427,945 params**, `epoch15000.ckpt`, input NHWC `[N,9,9,15]`, 137 actions.
- **`ace1-fast-weights`** — a distillation with the *same shape as ours*: single
  hidden layer, **H=192**, ~260k params, value **+ 137-way policy**, ReLU. `ka-br`
  runs it as a real AB leaf with NNUE-style sparse accumulation.

**Neither is a drop-in eval.** From `ka-encoder`: the 15 channels are mostly dense
(five constant broadcast planes at 81 nonzeros each, ~288 passability entries) —
~825 active of 1215, roughly 158k MACs/eval against Titanium's ~1.2 µs budget at
850k NPS. Worse, **ch13/ch14 encode legal wall placements**, and
`wallPlacableUngated` runs `hasPath(0) && hasPath(1)` per slot: **128 path floods
per position encoded**. `ka-br` calls itself a "leaf-cost regime" port. ACE and
Titanium sit at opposite ends of the nodes-vs-eval-quality tradeoff.

**What it is worth:** a candidate teacher, and independent evidence that a
192-wide single-hidden-layer net works as an AB leaf for this game.

**One design point it settles in our favour.** ACE encodes walls-in-hand as two
scalar constant planes (ch2/ch3). Fine for an 18-layer trunk where depth builds
the interaction — but `ace1-fast` is single-hidden-layer and inherited it anyway,
so it can only represent the two hands *linearly*. It cannot express "my 8th wall
is worth little when you hold 9." Our 121-row joint embedding can.

**Open, and worth settling BEFORE the retrain:** is this stronger than zero.ink
(~300 Elo above Titanium, the current label source)? The file contains no strength
claim — I checked. It is a match, not an argument. Retraining against the
second-best teacher and discovering it afterwards is the expensive mistake.
