import { PlayerType } from '../lib/engineConfig.js';
import { formatEngineScore } from '../lib/playerRegistry.js';
import { playerColorName } from '../lib/playerColors.js';

const STOP_LABELS = {
  searching: 'searching',
  minimax: 'αβ',
  mcts: 'MCTS',
  time: 'time',
  visits: 'cap',
  opening: 'book',
  hybrid: 'hybrid',
  race: 'race',
  converged: 'done',
  trivial: 'instant',
};

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function deepestEntry(depthLog) {
  if (!depthLog?.length) {
    return null;
  }
  return depthLog.reduce((best, entry) => (entry.depth > (best?.depth ?? 0) ? entry : best));
}

function raceBar(whiteDist, blackDist) {
  if (!Number.isFinite(whiteDist) || !Number.isFinite(blackDist)) {
    return { pct: 50, label: '—', side: 'even' };
  }
  const margin = blackDist - whiteDist;
  const pct = Math.max(8, Math.min(92, 50 + margin * 4.5));
  if (margin > 0) {
    return { pct, label: `W +${margin}`, side: 'white' };
  }
  if (margin < 0) {
    return { pct, label: `B +${-margin}`, side: 'black' };
  }
  return { pct: 50, label: 'even', side: 'even' };
}

function seatFromPly(ply) {
  return ply % 2 === 1 ? 0 : 1;
}

function lastThinkForSeat(state, seatIndex) {
  const log = state.moveThinkLog;
  if (!log?.length) {
    return null;
  }
  for (let i = log.length - 1; i >= 0; i--) {
    if (seatFromPly(log[i].ply) === seatIndex) {
      return log[i];
    }
  }
  return null;
}

function payloadFromSnapshot(snap) {
  return {
    live: !!snap.live,
    engine: snap.engine,
    move: snap.move,
    ply: snap.ply,
    whiteDist: snap.whiteDist,
    blackDist: snap.blackDist,
    score: snap.score,
    depth: snap.depth,
    pv: snap.pv ?? '',
    nodes: snap.nodes,
    rootWinRate: snap.rootWinRate,
    stoppedBy: snap.stoppedBy,
    rootMoves: snap.rootMoves,
  };
}

function payloadFromLiveSearch(ls, ply) {
  const deep = deepestEntry(ls.depthLog);
  return {
    live: true,
    engine: ls.playerLabel ?? '?',
    move: null,
    ply,
    whiteDist: ls.whiteDist,
    blackDist: ls.blackDist,
    score: deep?.score ?? ls.rootScore,
    depth: deep?.depth ?? ls.searchDepth,
    pv: deep?.pv ?? '',
    nodes: ls.nodes ?? ls.simulations,
    rootWinRate: ls.rootWinRate,
    stoppedBy: ls.mode ?? 'searching',
    rootMoves: ls.rootMoves,
  };
}

function thinkPayloadForSeat(state, seatIndex) {
  const playerType = state.settings.players[seatIndex];
  if (playerType === PlayerType.Human) {
    return null;
  }

  const saved = state.lastThinkBySeat?.[seatIndex];
  const isLive =
    state.aiThinking &&
    state.thinkingPlayerType === playerType &&
    state.liveSearch;

  if (isLive) {
    const ls = state.liveSearch;
    const ply = (state.actions?.length ?? 0) + 1;
    const livePayload = payloadFromLiveSearch(ls, ply);
    const hasFresh =
      livePayload.depth ||
      livePayload.score != null ||
      livePayload.nodes > 0 ||
      livePayload.rootWinRate != null ||
      livePayload.pv;
    if (!hasFresh && saved) {
      return { ...payloadFromSnapshot(saved), live: true, move: null, ply, stoppedBy: 'searching' };
    }
    return livePayload;
  }

  if (saved) {
    return payloadFromSnapshot(saved);
  }

  const entry = lastThinkForSeat(state, seatIndex);
  if (!entry) {
    return null;
  }

  const deep = deepestEntry(entry.depthLog);
  return {
    live: false,
    engine: entry.engine,
    move: entry.move,
    ply: entry.ply,
    whiteDist: entry.whiteDist,
    blackDist: entry.blackDist,
    score: deep?.score ?? entry.rootScore,
    depth: deep?.depth ?? entry.searchDepth,
    pv: deep?.pv ?? '',
    nodes: entry.nodes,
    rootWinRate: entry.rootWinRate,
    stoppedBy: entry.stoppedBy,
    rootMoves: entry.rootMoves,
  };
}

function formatBudget(nodes, rootWinRate, stoppedBy) {
  const parts = [];
  if (rootWinRate != null) {
    parts.push(`${(rootWinRate * 100).toFixed(0)}% wr`);
  }
  if (nodes > 0) {
    const isMcts = stoppedBy === 'mcts' || stoppedBy === 'time' || stoppedBy === 'visits' || stoppedBy === 'opening';
    parts.push(`${Number(nodes).toLocaleString()}${isMcts ? ' sims' : ''}`);
  }
  const stop = STOP_LABELS[stoppedBy] ?? stoppedBy;
  if (stop) {
    parts.push(stop);
  }
  return parts.join(' · ');
}

function formatTopRoots(rootMoves) {
  if (!rootMoves?.length) {
    return '';
  }
  return [...rootMoves]
    .sort((a, b) => b.score - a.score)
    .slice(0, 3)
    .map((r) => `${r.move} ${formatEngineScore(r.score)}`)
    .join('  ');
}

function renderThinkCard(seatIndex, payload) {
  const colorClass = seatIndex === 0 ? 'think-card--white' : 'think-card--black';
  const liveClass = payload.live ? ' think-card--live' : '';
  const bar = raceBar(payload.whiteDist, payload.blackDist);
  const scoreText =
    payload.score != null && Number.isFinite(Number(payload.score))
      ? formatEngineScore(payload.score)
      : '';
  const depthText = payload.depth ? `d${payload.depth}` : '';
  const budget = formatBudget(payload.nodes, payload.rootWinRate, payload.stoppedBy);
  const roots = formatTopRoots(payload.rootMoves);
  const distText =
    payload.whiteDist != null ? `W${payload.whiteDist} · B${payload.blackDist}` : '';

  return `
    <article class="think-card ${colorClass}${liveClass}">
      <header class="think-card__head">
        <span class="think-card__seat">${playerColorName(seatIndex + 1)}</span>
        <span class="think-card__engine">${escapeHtml(payload.engine)}</span>
        ${payload.live ? '<span class="think-card__pulse">live</span>' : ''}
      </header>
      ${payload.move ? `<div class="think-card__move">${escapeHtml(payload.move)}<span class="think-card__ply">ply ${payload.ply}</span></div>` : ''}
      <div class="think-card__race" title="Path distance margin (B−W steps)">
        <div class="think-card__race-track">
          <div class="think-card__race-white" style="width:${bar.pct}%"></div>
        </div>
        <span class="think-card__race-label think-card__race-label--${bar.side}">${bar.label}</span>
      </div>
      <div class="think-card__metrics">
        ${distText ? `<span>${distText}</span>` : ''}
        ${scoreText ? `<span class="think-card__score">${scoreText}</span>` : ''}
        ${depthText ? `<span>${depthText}</span>` : ''}
      </div>
      ${budget ? `<div class="think-card__budget">${escapeHtml(budget)}</div>` : ''}
      ${payload.pv ? `<div class="think-card__pv">${escapeHtml(payload.pv)}</div>` : ''}
      ${roots ? `<div class="think-card__roots">${escapeHtml(roots)}</div>` : ''}
    </article>
  `;
}

export function renderEngineThinkCards(state) {
  const cards = [0, 1]
    .map((seat) => {
      const payload = thinkPayloadForSeat(state, seat);
      return payload ? renderThinkCard(seat, payload) : '';
    })
    .filter(Boolean);

  if (!cards.length) {
    return '';
  }

  return `<div class="engine-think-cards">${cards.join('')}</div>`;
}

export function updateEngineThinkCards(container, state) {
  const host = container.querySelector('.play-panel');
  if (!host) {
    return;
  }

  const html = renderEngineThinkCards(state);
  const existing = host.querySelector('.engine-think-cards');

  if (!html) {
    existing?.remove();
    return;
  }

  if (existing) {
    existing.outerHTML = html;
    return;
  }

  host.insertAdjacentHTML('beforeend', html);
}
