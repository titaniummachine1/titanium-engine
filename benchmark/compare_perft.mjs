/**
 * Perft cross-check: Rust titanium vs scraped JS QuoridorBoard.
 * Run: node benchmark/compare_perft.mjs [depth]
 */

import { createRequire } from 'node:module';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { performance } from 'node:perf_hooks';

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { QuoridorBoard } = require(path.join(root, 'web/src/lib/gameLogic.js'));

const DEFAULT_DEPTH = 3;
const ORACLE_NODES = { 1: 131n, 2: 16677n, 3: 2062264n };
const depth = Number(process.argv[2] ?? DEFAULT_DEPTH);

function actionLabel(action) {
  if (action.wallType) {
    const { column, row } = action.coordinate;
    return `${column}${row}${action.wallType}`;
  }
  const { column, row } = action.coordinate;
  return `${column}${row}`;
}

function cloneBoard(board) {
  const copy = new QuoridorBoard();
  copy.playerToMove({ playerNum: board.playerToMove() });
  copy.moveNumber(board.moveNumber());
  for (let p = 1; p <= board.numPlayers(); p++) {
    copy.playerPosition({
      playerNum: p,
      coordinate: { ...board.playerPosition({ playerNum: p }) },
    });
    copy.wallsRemaining({
      playerNum: p,
      numWalls: board.wallsRemaining({ playerNum: p }),
    });
  }
  copy.setWalls(board.getWalls());
  return copy;
}

function jsPerft(board, d) {
  if (d === 0) return 1n;
  let nodes = 0n;
  for (const action of board.validActions()) {
    const next = cloneBoard(board);
    next.takeAction(action);
    nodes += jsPerft(next, d - 1);
  }
  return nodes;
}

function rustPerft(d) {
  const out = execSync(`cargo run --quiet --release -- perft ${d}`, {
    cwd: path.join(root, 'engine'),
    encoding: 'utf8',
  });
  const match = out.match(/perft \d+ (\d+)/);
  return match ? BigInt(match[1]) : null;
}

const board = new QuoridorBoard();
const d1 = board.validActions().length;

const jsStart = performance.now();
const jsNodes = jsPerft(new QuoridorBoard(), depth);
const jsMs = performance.now() - jsStart;

const rustStart = performance.now();
const rustNodes = rustPerft(depth);
const rustMs = performance.now() - rustStart;

console.log(`depth ${depth} (startpos) — default correctness depth is 3`);
console.log(`depth-1 move count: ${d1}`);
if (ORACLE_NODES[depth]) {
  console.log(`oracle nodes: ${ORACLE_NODES[depth]}`);
}
console.log(`JS   nodes: ${jsNodes}  time: ${(jsMs / 1000).toFixed(2)}s`);
console.log(`Rust nodes: ${rustNodes}  time: ${(rustMs / 1000).toFixed(2)}s (includes process spawn)`);

if (jsNodes !== rustNodes) {
  console.error('MISMATCH');
  process.exit(1);
}

console.log('OK — perft counts match');
if (jsMs > 0) {
  console.log(`speed ratio (JS wall time / Rust CLI): ${(jsMs / rustMs).toFixed(1)}x`);
}
