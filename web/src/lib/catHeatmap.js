/** CAT v3 heat → subtle board overlays (never solid black bars on walls). */

/** Unreachable sealed square — only case that gets a dark skip overlay. */
export function isSquareSkipped(reachable) {
  return reachable === false;
}

/** @returns {{ fill: string, opacity: number } | null} */
export function catSquareOverlay(heat, reachable, maxCm = 400) {
  if (isSquareSkipped(reachable)) {
    return null;
  }
  if (!Number.isFinite(heat) || heat <= 0) {
    return null;
  }
  const scale = Number.isFinite(maxCm) && maxCm > 0 ? maxCm : 400;
  const t = Math.min(1, heat / scale);
  if (t < 0.15) {
    const u = t / 0.15;
    const g = Math.round(110 + 50 * u);
    return {
      fill: `rgba(${g}, ${g}, ${Math.round(100 + 20 * u)}, ${0.22 + 0.12 * u})`,
      opacity: 1,
    };
  }
  const u = (t - 0.15) / 0.85;
  if (u < 0.35) {
    const v = u / 0.35;
    return {
      fill: `rgba(255, ${Math.round(200 + 30 * v)}, ${Math.round(40 * (1 - v))}, ${0.28 + 0.12 * v})`,
      opacity: 1,
    };
  }
  if (u < 0.7) {
    const v = (u - 0.35) / 0.35;
    return {
      fill: `rgba(255, ${Math.round(150 - 70 * v)}, 0, ${0.32 + 0.1 * v})`,
      opacity: 1,
    };
  }
  const v = (u - 0.7) / 0.3;
  return {
    fill: `rgba(${Math.round(230 - 40 * v)}, ${Math.round(50 - 30 * v)}, ${Math.round(30 + 10 * v)}, ${0.38 + 0.12 * v})`,
    opacity: 1,
  };
}

/** Outline color for searchable wall hints (not filled bars). */
export function catWallOutlineColor(heat, maxCm = 400) {
  if (!Number.isFinite(heat) || heat <= 0) {
    return 'rgba(120, 115, 105, 0.55)';
  }
  const overlay = catSquareOverlay(heat, true, maxCm);
  if (!overlay) {
    return 'rgba(120, 115, 105, 0.55)';
  }
  return overlay.fill.replace(/,\s*[\d.]+\)$/, ', 0.85)');
}

/**
 * @param {string[]} algebraicMoves
 */
export async function fetchCatSnapshot(algebraicMoves) {
  const res = await fetch('/api/titanium/cat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ moves: algebraicMoves }),
  });
  const data = await res.json();
  if (!res.ok || data.error) {
    throw new Error(data.error ?? `CAT request failed (${res.status})`);
  }
  return data;
}

/**
 * @param {Array<{alg: string, heat: number, search?: boolean, skip?: boolean, pruned?: boolean}>} walls
 * @returns {Map<string, {heat: number, search: boolean, skip: boolean}>}
 */
export function indexCatWalls(walls) {
  const map = new Map();
  for (const entry of walls ?? []) {
    if (!entry?.alg) {
      continue;
    }
    const skip = entry.skip ?? entry.pruned ?? false;
    const search = entry.search ?? !skip;
    map.set(entry.alg, {
      heat: entry.heat ?? 0,
      search,
      skip,
    });
  }
  return map;
}

/** Engine row/col 0..8 → flat index in squares[81]. */
export function catSquareIndex(engineRow, engineCol) {
  return engineRow * 9 + engineCol;
}
