/**
 * Perft divide diff — find first move where JS and Rust subtree counts diverge.
 * Like Stockfish perftree / chess "divide" debugging.
 *
 * Usage:
 *   node benchmark/perft_diff.mjs [depth] [move1 move2 ...]
 *   node benchmark/perft_diff.mjs 3
 *   node benchmark/perft_diff.mjs 1 d8v
 */

import { createRequire } from 'node:module';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { QuoridorBoard } = require(path.join(root, 'web/src/lib/gameLogic.js'));

const depth = Number(process.argv[2] ?? 3);
const prefix = process.argv.slice(3);

function label(action) {
  if (action.wallType) {
    return `${action.coordinate.column}${action.coordinate.row}${action.wallType}`;
  }
  return `${action.coordinate.column}${action.coordinate.row}`;
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

function jsDivide(board, d) {
  const lines = new Map();
  for (const action of board.validActions()) {
    const mv = label(action);
    const next = cloneBoard(board);
    next.takeAction(action);
    lines.set(mv, jsPerft(next, d - 1));
  }
  return lines;
}

function rustDivide(d, moves) {
  const args = ['run', '--quiet', '--release', '--', 'divide', String(d)];
  if (moves.length) args.push(...moves);
  const out = execFileSync('cargo', args, {
    cwd: path.join(root, 'engine'),
    encoding: 'utf8',
  });
  const map = new Map();
  for (const line of out.trim().split('\n')) {
    const m = line.match(/^(\S+)\s+(\d+)$/);
    if (m) map.set(m[1], BigInt(m[2]));
  }
  const total = [...map.values()].reduce((a, b) => a + b, 0n);
  return { map, total };
}

let board = new QuoridorBoard();
for (const mv of prefix) {
  board.takeAction(mv);
}

const jsLines = jsDivide(board, depth);
const { map: rustLines, total: rustTotal } = rustDivide(depth, prefix);
const jsTotal = [...jsLines.values()].reduce((a, b) => a + b, 0n);

const pathLabel = prefix.length ? prefix.join(' ') + ' ' : '';
console.log(`position: ${pathLabel || 'startpos'}  depth: ${depth}`);
console.log(`JS total: ${jsTotal}   Rust total: ${rustTotal}`);

if (jsTotal === rustTotal) {
  console.log('OK — totals match');
  process.exit(0);
}

const allMoves = new Set([...jsLines.keys(), ...rustLines.keys()]);
const diffs = [];
for (const mv of [...allMoves].sort()) {
  const js = jsLines.get(mv) ?? null;
  const rust = rustLines.get(mv) ?? null;
  if (js !== rust) {
    diffs.push({ mv, js: js?.toString() ?? '-', rust: rust?.toString() ?? '-' });
  }
}

console.log(`\n${diffs.length} divide line(s) differ:\n`);
for (const d of diffs) {
  console.log(`  ${d.mv.padEnd(6)}  JS ${d.js.padStart(6)}   Rust ${d.rust.padStart(6)}`);
}

console.log('\nTip: drill down — node benchmark/perft_diff.mjs <depth-1> <prefix> <move>');
process.exit(1);
