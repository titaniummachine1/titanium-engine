/**
 * Path distance from a Gorisanson game state (BFS on pawn graph).
 */

import { gorisansonMoveToAction } from './gorisanson_bridge.mjs';
import { toAlgebraic } from '../../web/src/lib/gameLogic.js';

function pawn(game, playerIndex) {
  return playerIndex === 0 ? game.pawn0 : game.pawn1;
}

function hWallBlocksBelow(hWalls, row, col) {
  if (col > 0 && hWalls[row]?.[col - 1]) return true;
  if (col < 8 && hWalls[row]?.[col]) return true;
  return false;
}

function vWallBlocksRight(vWalls, row, col) {
  if (row > 0 && vWalls[row - 1]?.[col]) return true;
  if (row < 8 && vWalls[row]?.[col]) return true;
  return false;
}

function canStep(game, r, c, dr, dc) {
  const hWalls = game.board.walls.horizontal;
  const vWalls = game.board.walls.vertical;
  if (dr === 1) {
    if (r > 7) return false;
    return !hWallBlocksBelow(hWalls, r + 1, c);
  }
  if (dr === -1) {
    if (r < 1) return false;
    return !hWallBlocksBelow(hWalls, r, c);
  }
  if (dc === 1) {
    if (c > 7) return false;
    return !vWallBlocksRight(vWalls, r, c);
  }
  if (dc === -1) {
    if (c < 1) return false;
    return !vWallBlocksRight(vWalls, r, c - 1);
  }
  return false;
}

function bfsDistance(game, playerIndex) {
  const startPawn = pawn(game, playerIndex);
  const goalRow = startPawn.goalRow;
  const startRow = startPawn.position.row;
  const startCol = startPawn.position.col;

  const visited = Array.from({ length: 9 }, () => new Array(9).fill(false));
  const queue = [{ row: startRow, col: startCol, dist: 0 }];
  visited[startRow][startCol] = true;

  while (queue.length > 0) {
    const { row, col, dist } = queue.shift();
    if (row === goalRow) return dist;

    for (const [dr, dc] of [
      [1, 0],
      [-1, 0],
      [0, 1],
      [0, -1],
    ]) {
      if (!canStep(game, row, col, dr, dc)) continue;
      const nr = row + dr;
      const nc = col + dc;
      if (visited[nr][nc]) continue;
      visited[nr][nc] = true;
      queue.push({ row: nr, col: nc, dist: dist + 1 });
    }
  }
  return 255;
}

export function evalPosition(game) {
  const whiteDist = bfsDistance(game, 0);
  const blackDist = bfsDistance(game, 1);
  return {
    whiteDist,
    blackDist,
    margin: blackDist - whiteDist,
  };
}

export function pawnSquare(game, playerIndex) {
  const p = pawn(game, playerIndex);
  return toAlgebraic(
    gorisansonMoveToAction([[p.position.row, p.position.col], null, null]),
  );
}

export function wallsUsed(game, playerIndex) {
  return 10 - pawn(game, playerIndex).numberOfLeftWalls;
}

export function wallsLeft(game, playerIndex) {
  return pawn(game, playerIndex).numberOfLeftWalls;
}
