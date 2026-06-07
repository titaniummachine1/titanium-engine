import { createRequire } from 'node:module';
import { execSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const require = createRequire(import.meta.url);
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const { QuoridorBoard } = require(path.join(root, 'web/src/lib/gameLogic.js'));

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
    copy.playerPosition({ playerNum: p, coordinate: { ...board.playerPosition({ playerNum: p }) } });
    copy.wallsRemaining({ playerNum: p, numWalls: board.wallsRemaining({ playerNum: p }) });
  }
  copy.setWalls(board.getWalls());
  return copy;
}

function rustMovesAfter(moves) {
  // not implemented - use divide on custom position later
  return new Set();
}

const wall = 'd8v';
const board = new QuoridorBoard();
board.takeAction(wall);

const jsLabels = new Set(board.validActions().map(label));
console.log('JS moves after', wall, ':', jsLabels.size);

// get rust moves by shelling out - need position FEN or apply move
// For now diff by comparing known labels from titanium moves after applying via future API
// Quick hack: run rust with stdin not available - compare manually

const onlyJs = [];
const onlyRust = [];

// Export JS list for manual rust check
console.log([...jsLabels].sort().join('\n'));
