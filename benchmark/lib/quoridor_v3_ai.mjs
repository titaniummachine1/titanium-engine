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
