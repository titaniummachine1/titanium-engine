/**
 * Validate engine moves against Gorisanson rules (authoritative for benchmarks).
 */

import { allLegalMoves, cloneGorisansonGame, moveLabel } from './gorisanson_moves.mjs';

function partEqual(a, b) {
  if (a === null && b === null) return true;
  if (!a || !b) return false;
  return a[0] === b[0] && a[1] === b[1];
}

export function movesEqual(a, b) {
  return partEqual(a[0], b[0]) && partEqual(a[1], b[1]) && partEqual(a[2], b[2]);
}

export function isLegalMove(game, move) {
  return allLegalMoves(game).some((m) => movesEqual(m, move));
}

function describeMove(move) {
  try {
    return moveLabel(move);
  } catch {
    return JSON.stringify(move);
  }
}

/**
 * @returns {{ ok: true } | { ok: false, reason: string, legalCount: number }}
 */
export function validateMove(game, move) {
  const legal = allLegalMoves(game);
  const label = describeMove(move);

  if (legal.some((m) => movesEqual(m, move))) {
    return { ok: true };
  }

  const [pawn, horiz, vert] = move;

  if (pawn) {
    const [r, c] = pawn;
    const positions = game.validNextPositions;
    if (!positions?.[r]?.[c]) {
      return {
        ok: false,
        reason: `${label}: pawn step to (${r},${c}) not in validNextPositions`,
        legalCount: legal.length,
      };
    }
    return {
      ok: false,
      reason: `${label}: pawn move rejected (${legal.length} legal moves)`,
      legalCount: legal.length,
    };
  }

  if (horiz) {
    const [r, c] = horiz;
    if (game.pawnOfTurn.numberOfLeftWalls <= 0) {
      return { ok: false, reason: `${label}: no walls left`, legalCount: legal.length };
    }
    const walls = game.validNextWalls;
    if (!walls?.horizontal?.[r]?.[c]) {
      return {
        ok: false,
        reason: `${label}: horizontal wall (${r},${c}) overlaps or out of bounds`,
        legalCount: legal.length,
      };
    }
    if (!game.testIfExistPathsToGoalLinesAfterPlaceHorizontalWall(r, c)) {
      return {
        ok: false,
        reason: `${label}: horizontal wall (${r},${c}) blocks all paths for one player`,
        legalCount: legal.length,
      };
    }
    return {
      ok: false,
      reason: `${label}: horizontal wall illegal (${legal.length} legal)`,
      legalCount: legal.length,
    };
  }

  if (vert) {
    const [r, c] = vert;
    if (game.pawnOfTurn.numberOfLeftWalls <= 0) {
      return { ok: false, reason: `${label}: no walls left`, legalCount: legal.length };
    }
    const walls = game.validNextWalls;
    if (!walls?.vertical?.[r]?.[c]) {
      return {
        ok: false,
        reason: `${label}: vertical wall (${r},${c}) overlaps or out of bounds`,
        legalCount: legal.length,
      };
    }
    if (!game.testIfExistPathsToGoalLinesAfterPlaceVerticalWall(r, c)) {
      return {
        ok: false,
        reason: `${label}: vertical wall (${r},${c}) blocks all paths for one player`,
        legalCount: legal.length,
      };
    }
    return {
      ok: false,
      reason: `${label}: vertical wall illegal (${legal.length} legal)`,
      legalCount: legal.length,
    };
  }

  return { ok: false, reason: `${label}: empty move tuple`, legalCount: legal.length };
}

/** First legal move as emergency fallback (shortest-path pawn if possible). */
export function fallbackLegalMove(game) {
  const legal = allLegalMoves(game);
  if (!legal.length) {
    return null;
  }
  const pawnOnly = legal.find((m) => m[0] !== null);
  return pawnOnly ?? legal[0];
}

export function tryApplyMove(game, move) {
  const trial = cloneGorisansonGame(game);
  trial.doMove(move, true);
  return trial;
}
