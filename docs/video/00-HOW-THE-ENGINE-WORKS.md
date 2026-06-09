# How Titanium Engine works (human explanation — video narration)

Use this as voiceover script. No machine code — just the story.

---

## The game we're simulating

Two pawns on a 9×9 board. Each player has 10 walls. Goal: reach the opposite side.

Every turn you either:

- move your pawn one step (with jump rules when blocked), or
- place a wall in one of the 64 gutter slots on the board.

That's it. But **walls explode the move count** — about 131 legal choices from the opening.

---

## What's inside the engine (layers)

Think of four layers, bottom to top:

### 1. The board (`board.rs`)

Remembers:

- where both pawns are
- two bitboards (64 bits each) for horizontal and vertical walls
- whose turn, walls remaining

Walls are **bits in a integer** — flip a bit on/off, test a bit in one CPU instruction. That's our “chess bitboard” idea, adapted for Quoridor.

### 2. Can I walk there? (`grid.rs`)

Before BFS or move gen: **can this pawn step in this direction?**

Checks the two wall segments that could block that step (walls are 2 cells wide). This was the source of our first perft bug — sideways steps need different wall anchors than up/down.

### 3. Can I still win? (`path.rs`)

When someone tries to place a wall, we ask: **can both players still reach their goal?**

BFS on the 9×9 grid — same idea as the scraped JavaScript, but with a tiny visited bitset instead of a Set of strings.

### 4. What can I play? (`moves.rs`)

- Generate pawn moves (including jumps)
- Try every wall slot; keep it if rules say legal
- “Floating” walls that don't touch anything are still **legal** in the UI rules (even if dumb strategically)

### 5. Is move gen correct? (`perft.rs`)

**Perft** = count every legal sequence of moves to depth N.

Chess engines compare against known tables. Quoridor has none — we compare against the **scraped website rules** and [gorisanson's](https://github.com/gorisanson/quoridor-ai) full legal move generator.

---

## Data flow for one search node (future αβ)

```
position → legal moves → for each move:
              clone board → apply → evaluate (BFS distances) → recurse
```

Right now we only have the first half (moves + perft). Search comes next.

---

## Why Rust?

Not magic — **volume**. Perft depth 3 is 2 million nodes. Each node:

- list ~100+ moves
- many walls each trigger BFS

JavaScript does the same math; it just can't iterate fast enough. We measured:

- **Rust depth 3 in ~0.2s**
- **JS depth 2 in ~0.13s**, depth 3 is painful on JS

### The surprise we hit

We thought Rust = “perft 10.” **Wrong.**

Depth 4 is ~**100×** more nodes than depth 3. Even in Rust, **naive clone-every-node perft doesn't scale past 3** without smarts:

- **Make/unmake** instead of clone
- **Don't search every wall** in play (only in correctness perft)
- **Transposition table** so positions aren't recomputed

Rust gives runway. **Smarts spend it.** That's what [gorisanson](https://github.com/gorisanson/quoridor-ai) and competition αβ engines do — not optional polish.

---

## What changed since this script was written (Jun 2026)

- **αβ search ships** — ID negamax + CAT prune + adaptive LMR (MCTS deprecated in Rust engine)
- **Make/unmake + TT + flood fill** — perft d4 **~3.4s** (247M nodes), not minutes
- **Perft still full legality** — wall pruning is for search only

*The narration below is still valid for early episodes; update live demos to `perft 3` or timed `perft 4` (~3s).*

---

## One-line pitch for the video

“We stole the rules from the website, bitboarded the walls, BFS for paths, perft to prove it's right — then Rust to survive the 2-million-node opening.”
