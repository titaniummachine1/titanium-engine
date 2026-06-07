import { createRequire } from 'node:module';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { QuoridorBoard } = require(path.join(root, 'web/src/lib/gameLogic.js'));

const after = process.argv[2] ?? 'd8v';

function label(a) {
  return a.wallType
    ? `${a.coordinate.column}${a.coordinate.row}${a.wallType}`
    : `${a.coordinate.column}${a.coordinate.row}`;
}

const board = new QuoridorBoard();
board.takeAction(after);
const js = new Set(board.validActions().map(label));

const rustOut = execSync(`cargo test debug_d8v -- --nocapture`, {
  cwd: path.join(root, 'engine'),
  encoding: 'utf8',
});
const rust = new Set(
  rustOut
    .split('\n')
    .filter((l) => /^[a-h][1-9]/.test(l))
    .map((l) => l.trim()),
);

const onlyJs = [...js].filter((m) => !rust.has(m)).sort();
const onlyRust = [...rust].filter((m) => !js.has(m)).sort();

console.log(`after ${after}: JS ${js.size} Rust ${rust.size}`);
console.log('only JS:', onlyJs);
console.log('only Rust:', onlyRust);
