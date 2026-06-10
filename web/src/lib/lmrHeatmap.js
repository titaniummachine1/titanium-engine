/** LMR vision — root move depth / reduction overlays from engine JSON. */

/**
 * @param {string[]} algebraicMoves
 * @param {number} [timeSec]
 */
export async function fetchLmrSnapshot(algebraicMoves, timeSec = 10) {
  const res = await fetch('/api/titanium/lmr', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ moves: algebraicMoves, timeSec }),
  });
  const data = await res.json();
  if (!res.ok || data.error) {
    throw new Error(data.error ?? `LMR request failed (${res.status})`);
  }
  return data;
}

function normalizeLmrEntry(entry) {
  const reduction = Number(entry.reduction ?? 0);
  const childFull = Number(entry.childDepthFull ?? entry.child_depth_full ?? 0);
  const childUsed = Number(entry.childDepthUsed ?? entry.child_depth_used ?? childFull);
  return {
    move: entry.move ?? entry.mv,
    kind: entry.kind ?? (entry.is_pawn || entry.isPawn ? 'pawn' : 'wall'),
    order: entry.order ?? 0,
    catCm: entry.catCm ?? entry.cat_cm ?? 0,
    tactical: Boolean(entry.tactical),
    hot: Boolean(entry.hot),
    pruned: Boolean(entry.pruned),
    reduction,
    childDepthFull: childFull,
    childDepthUsed: childUsed,
    reSearched: Boolean(entry.reSearched ?? entry.re_searched),
    inFullWindow: Boolean(entry.inFullWindow ?? entry.in_full_window),
    score: entry.score ?? null,
    nodes: Number(entry.nodes ?? 0),
    sharePct: 0,
    searched: entry.searched !== false,
    unsearched: Boolean(entry.unsearched),
  };
}

function attachNodeShares(moves) {
  const total = moves.reduce((sum, m) => sum + (m.nodes > 0 ? m.nodes : 0), 0);
  if (total <= 0) {
    return moves;
  }
  return moves.map((m) => ({
    ...m,
    sharePct: m.nodes > 0 ? Math.round((m.nodes / total) * 100) : 0,
  }));
}

/**
 * Fill gaps in search rootMoves with the static pre-search plan (same legal list).
 * Search behaviour unchanged — viz only.
 *
 * @param {object[]} planMoves
 * @param {object[]} searchMoves
 */
export function mergeLmrPlanWithSearch(planMoves, searchMoves) {
  if (!planMoves?.length) {
    return searchMoves ?? [];
  }
  if (!searchMoves?.length) {
    return planMoves.map((m) => ({ ...m, unsearched: true, searched: false, nodes: 0 }));
  }
  const planByKey = indexLmrMoves(planMoves);
  const searchByKey = indexLmrMoves(searchMoves);
  const keys = new Set([...planByKey.keys(), ...searchByKey.keys()]);
  const merged = [];
  for (const key of keys) {
    const plan = planByKey.get(key);
    const search = searchByKey.get(key);
    if (search) {
      merged.push({
        ...plan,
        ...search,
        catCm: search.catCm ?? plan?.catCm ?? 0,
        searched: true,
        unsearched: false,
      });
    } else if (plan) {
      merged.push({
        ...plan,
        searched: false,
        unsearched: true,
        nodes: 0,
        sharePct: 0,
      });
    }
  }
  merged.sort((a, b) => a.order - b.order);
  return merged;
}

/**
 * @param {Array<Record<string, unknown>>} moves
 * @returns {Map<string, object>}
 */
export function indexLmrMoves(moves) {
  const map = new Map();
  for (const entry of moves ?? []) {
    const alg = entry.move ?? entry.mv;
    if (!alg) {
      continue;
    }
    map.set(String(alg), entry);
  }
  return map;
}

function maxReduction(moves, searchDepth) {
  let max = 1;
  for (const m of moves ?? []) {
    const full = m.childDepthFull || Math.max(0, searchDepth - 1);
    max = Math.max(max, m.reduction, full);
  }
  return max;
}

/** 0 = full depth (green), 1 = deepest cut (red). */
function reductionRatio(entry, viz) {
  const searchDepth = viz?.searchDepth ?? viz?.idDepth ?? 1;
  const full = entry.childDepthFull || Math.max(0, searchDepth - 1);
  if (full <= 0) {
    return 0;
  }
  return Math.min(1, Math.max(0, entry.reduction / full));
}

function lerp(a, b, t) {
  return Math.round(a + (b - a) * t);
}

function heatFill(ratio, alpha = 0.72) {
  const t = Math.min(1, Math.max(0, ratio));
  let r;
  let g;
  let b;
  if (t < 0.5) {
    const u = t / 0.5;
    r = lerp(72, 230, u);
    g = lerp(200, 180, u);
    b = lerp(120, 60, u);
  } else {
    const u = (t - 0.5) / 0.5;
    r = lerp(230, 230, u);
    g = lerp(180, 90, u);
    b = lerp(60, 70, u);
  }
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

/**
 * @param {object} payload
 * @param {object[]} [payload.planMoves] — pre-search plan to pad search gaps
 */
export function buildLmrViz(payload) {
  const shallow = payload.source === 'shallow';
  const profile = payload.lmrProfile ?? {};
  const depthLog = payload.depthLog ?? [];
  const deepFromLog = depthLog.length
    ? depthLog.reduce((best, e) => ((e.depth ?? 0) > (best?.depth ?? 0) ? e : best))
    : null;
  const searchDepth =
    payload.searchDepth ??
    profile.idDepth ??
    deepFromLog?.depth ??
    payload.idDepth ??
    1;

  let raw = payload?.moves ?? payload?.rootMoves ?? [];
  if (!shallow && payload.planMoves?.length) {
    const normalizedSearch = raw.map(normalizeLmrEntry);
    const normalizedPlan = payload.planMoves.map(normalizeLmrEntry);
    raw = mergeLmrPlanWithSearch(normalizedPlan, normalizedSearch);
  }
  if (!raw.length) {
    return null;
  }

  let moves = raw.map(normalizeLmrEntry);
  if (!shallow) {
    moves = attachNodeShares(moves);
  }
  const moveIndex = indexLmrMoves(moves);
  const maxR = maxReduction(moves, searchDepth);
  return {
    source: payload.source ?? 'search',
    shallow,
    idDepth: searchDepth,
    searchDepth,
    maxReduction: maxR,
    lmrProfile: profile,
    lmrReSearches: payload.lmrReSearches ?? null,
    totalNodes: moves.reduce((s, m) => s + m.nodes, 0),
    searchedCount: moves.filter((m) => m.searched).length,
    moveIndex,
    moves,
    label: shallow ? 'pre-search plan' : `search d${searchDepth}`,
  };
}

/**
 * @returns {{ fill: string, label: string }}
 */
export function lmrDepthStyle(entry, viz) {
  if (!entry) {
    return { fill: 'transparent', label: '' };
  }
  const ratio = reductionRatio(entry, viz);
  const alpha = entry.unsearched ? 0.38 : entry.pruned ? 0.42 : 0.72;
  const fill = heatFill(ratio, alpha);
  const used = entry.childDepthUsed;
  const label = entry.unsearched
    ? `not searched · plan −${entry.reduction} · d${used}`
    : entry.reduction > 0
      ? `d${used} (−${entry.reduction})`
      : `d${used} full`;
  return { fill, label };
}

export function lmrWallOutlineColor(entry, viz) {
  const style = lmrDepthStyle(entry, viz);
  return style.fill.replace(/,\s*[\d.]+\)$/, ', 0.92)');
}

export function lmrDisplayText(entry, viz) {
  if (!entry) {
    return '';
  }
  const used = entry.childDepthUsed;
  if (viz?.shallow) {
    if (entry.reduction > 0) {
      return `−${entry.reduction}`;
    }
    if (entry.catCm > 0) {
      return String(entry.catCm);
    }
    return `d${used}`;
  }
  if (entry.searched && entry.sharePct > 0) {
    return `${entry.sharePct}%`;
  }
  if (entry.reduction > 0) {
    return `−${entry.reduction}`;
  }
  if (entry.catCm > 0) {
    return String(entry.catCm);
  }
  return `d${used}`;
}

export function lmrSubLabel(entry, viz) {
  if (!entry) {
    return '';
  }
  const parts = [];
  if (viz?.shallow) {
    if (entry.reduction > 0 && entry.catCm > 0) {
      parts.push(String(entry.catCm));
    } else if (entry.reduction === 0) {
      parts.push(`d${entry.childDepthUsed}`);
    }
  } else {
    if (entry.reduction > 0 || entry.catCm > 0) {
      parts.push(`d${entry.childDepthUsed}`);
    }
    if (entry.catCm > 0 && entry.reduction > 0) {
      parts.push(String(entry.catCm));
    }
  }
  if (entry.reSearched) {
    parts.push('↺');
  }
  if (entry.unsearched) {
    parts.push('·');
  }
  return parts.join(' ');
}
