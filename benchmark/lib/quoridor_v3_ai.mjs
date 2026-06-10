/**
 * Quoridor v3 αβ engine (vendored JS) for headless benchmarks.
 */

import fs from 'node:fs';
import vm from 'node:vm';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { algebraicMovesToV3, v3MoveToAlgebraic } from '../../web/src/lib/quoridorV3Codec.js';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../..');
const ENGINE_PATH = path.join(ROOT, 'web', 'src', 'vendor', 'quoridor-v3', 'engine.js');

let Quoridor;
let Search;

function loadEngine() {
  if (Quoridor && Search) {
    return;
  }
  const code = fs.readFileSync(ENGINE_PATH, 'utf8');
  const sandbox = {
    module: { exports: {} },
    Uint8Array,
    Int16Array,
    Date,
    Infinity,
  };
  vm.runInNewContext(`${code}\n;`, sandbox);
  ({ Quoridor, Search } = sandbox.module.exports);
}

/**
 * @param {string[]} algebraicHistory
 * @param {{ timeMs?: number, maxDepth?: number }} [opts]
 */
/**
 * Seeded random opening — pawn-biased legal moves, for match variety.
 * Same seed → same opening, so color-swapped pairs share positions.
 */
export function randomOpeningMoves(plies = 4, seed = 1) {
  loadEngine();
  let s = (seed >>> 0) || 1;
  const rnd = () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
  const game = new Quoridor();
  const out = [];
  for (let i = 0; i < plies; i++) {
    const legal = game.legalMoves();
    const pawns = legal.filter((m) => m < 100);
    const walls = legal.filter((m) => m >= 100);
    const pool = walls.length === 0 || rnd() < 0.7 ? pawns : walls;
    const pick = pool[Math.floor(rnd() * pool.length)];
    out.push(v3MoveToAlgebraic(pick));
    game.makeMove(pick);
  }
  return out;
}

export function chooseQuoridorV3Move(algebraicHistory = [], opts = {}) {
  loadEngine();
  const timeMs = opts.timeMs ?? 500;
  const maxDepth = opts.maxDepth ?? 24;
  const game = new Quoridor();
  if (algebraicHistory.length > 0) {
    game.loadState({ moves: algebraicMovesToV3(algebraicHistory) });
  }
  const search = new Search(game);
  const started = performance.now();
  const result = search.think(timeMs, maxDepth, false);
  const elapsedMs = performance.now() - started;
  const algebraic = v3MoveToAlgebraic(result.move);
  return {
    move: algebraic,
    meta: {
      stoppedBy: 'alphabeta',
      searchDepth: result.depth,
      nodes: result.nodes,
      rootScore: result.score,
      elapsedMs: result.ms ?? elapsedMs,
    },
  };
}
