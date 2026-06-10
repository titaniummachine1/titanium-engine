import { PlayerType } from '../lib/engineConfig.js';
import { formatEngineScore } from '../lib/playerRegistry.js';
import { playerColorName } from '../lib/playerColors.js';

const MCTS_MODES = new Set(['mcts', 'opening', 'visits', 'bridge', 'bridge-visits', 'forced']);

function isTitaniumAb(payload) {
  const eng = String(payload.engine ?? '');
  return eng.includes('Titanium') || eng.includes('αβ');
}

function isNegamaxPayload(payload) {
  return (
    payload.stoppedBy === 'minimax' ||
    payload.mode === 'minimax' ||
    isTitaniumAb(payload)
  );
}

function usesRollouts(payload) {
  if (isNegamaxPayload(payload)) {
    return false;
  }
  if (MCTS_MODES.has(payload.stoppedBy) || payload.stoppedBy === 'time') {
    return true;
  }
  if (payload.stoppedBy === 'searching' || payload.stoppedBy === 'hybrid') {
    return (
      payload.rootWinRate != null ||
      ((payload.simulations ?? payload.nodes ?? 0) > 0 && !payload.depthLog?.length)
    );
  }
  return false;
}

function rolloutCount(payload) {
  return payload.simulations ?? payload.nodes ?? 0;
}

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

function setText(el, text) {
  if (!el) {
    return;
  }
  const next = text ?? '';
  if (el.textContent !== next) {
    el.textContent = next;
  }
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
    return { pct, label: `White +${margin}`, side: 'white' };
  }
  if (margin < 0) {
    return { pct, label: `Black +${-margin}`, side: 'black' };
  }
  return { pct: 50, label: 'Even race', side: 'even' };
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
    depthLog: snap.depthLog ?? [],
    pv: snap.pv ?? '',
    nodes: snap.nodes,
    simulations: snap.simulations ?? snap.nodes ?? 0,
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
    depthLog: ls.depthLog ?? [],
    pv: deep?.pv ?? '',
    nodes: ls.nodes ?? ls.simulations,
    simulations: ls.simulations ?? ls.nodes ?? 0,
    rootWinRate: ls.rootWinRate,
    stoppedBy: ls.mode ?? 'searching',
    rootMoves: ls.rootMoves,
  };
}

function idlePayload(seatIndex) {
  return {
    idle: true,
    live: false,
    engine: '—',
    move: null,
    ply: null,
    whiteDist: null,
    blackDist: null,
    score: null,
    depth: null,
    depthLog: [],
    pv: '',
    nodes: 0,
    simulations: 0,
    rootWinRate: null,
    stoppedBy: 'idle',
    rootMoves: null,
    seatIndex,
  };
}

function thinkPayloadForSeat(state, seatIndex) {
  const playerType = state.settings.players[seatIndex];
  if (playerType === PlayerType.Human) {
    return idlePayload(seatIndex);
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
      return { ...payloadFromSnapshot(saved), live: true, move: saved.move ?? null, ply, stoppedBy: 'searching' };
    }
    return { ...livePayload, move: saved?.move ?? null, ply: saved?.ply ?? ply };
  }

  if (saved) {
    return payloadFromSnapshot(saved);
  }

  const entry = lastThinkForSeat(state, seatIndex);
  if (!entry) {
    return idlePayload(seatIndex);
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
    depthLog: entry.depthLog ?? [],
    pv: deep?.pv ?? '',
    nodes: entry.nodes,
    simulations: entry.simulations ?? entry.nodes ?? 0,
    rootWinRate: entry.rootWinRate,
    stoppedBy: entry.stoppedBy,
    rootMoves: entry.rootMoves,
  };
}

function prefersWinRate(payload) {
  if (payload.rootWinRate == null || isNegamaxPayload(payload)) {
    return false;
  }
  if (MCTS_MODES.has(payload.stoppedBy)) {
    return true;
  }
  return payload.score == null && payload.depth == null;
}

function heroRight(payload) {
  if (usesRollouts(payload)) {
    const n = rolloutCount(payload);
    return { value: n > 0 ? n.toLocaleString() : '…', label: 'rollouts' };
  }
  return {
    value: payload.depth ? String(payload.depth) : '…',
    label: 'depth',
  };
}

function heroMetric(payload) {
  const right = heroRight(payload);

  if (prefersWinRate(payload)) {
    const pct = Number(payload.rootWinRate) * 100;
    if (!Number.isFinite(pct)) {
      return { left: '…', right: right.value, rightLabel: right.label, tone: 'even', mode: 'winrate' };
    }
    return {
      left: `${pct.toFixed(0)}%`,
      right: right.value,
      rightLabel: right.label,
      tone: pct >= 50 ? 'good' : 'bad',
      mode: 'winrate',
    };
  }

  if (payload.score != null && Number.isFinite(Number(payload.score))) {
    const n = Number(payload.score);
    return {
      left: formatEngineScore(n),
      right: right.value,
      rightLabel: right.label,
      tone: n > 50 ? 'good' : n < -50 ? 'bad' : 'even',
      mode: 'eval',
    };
  }

  if (payload.rootWinRate != null) {
    const pct = Number(payload.rootWinRate) * 100;
    return {
      left: `${pct.toFixed(0)}%`,
      right: right.value,
      rightLabel: right.label,
      tone: pct >= 50 ? 'good' : 'bad',
      mode: 'winrate',
    };
  }

  return {
    left: '…',
    right: right.value,
    rightLabel: right.label,
    tone: 'even',
    mode: payload.live ? 'searching' : 'idle',
  };
}

function formatNodes(nodes, stoppedBy, payload = {}) {
  if (!nodes || nodes <= 0) {
    return '';
  }
  const label = isNegamaxPayload({ stoppedBy, ...payload }) ? 'nodes' : 'sims';
  return `${Number(nodes).toLocaleString()} ${label}`;
}

function formatBudgetLine(payload) {
  const parts = [];
  const nodes = formatNodes(payload.nodes, payload.stoppedBy, payload);
  if (nodes) {
    parts.push(nodes);
  }
  const stop = STOP_LABELS[payload.stoppedBy] ?? payload.stoppedBy;
  if (stop && stop !== 'searching' && stop !== 'idle') {
    parts.push(stop);
  }
  return parts.join(' · ');
}

function pvHeadline(pv) {
  if (!pv) {
    return '';
  }
  const first = pv.trim().split(/\s+/)[0];
  return first || '';
}

function formatRootMoveScore(r, payload) {
  if (r.winRate != null && Number.isFinite(r.winRate)) {
    const visits = r.visits > 0 ? ` · ${Number(r.visits).toLocaleString()}v` : '';
    return `${(r.winRate * 100).toFixed(0)}%${visits}`;
  }
  if (usesRollouts(payload) && r.score != null && Number.isFinite(r.score)) {
    const visits = r.visits > 0 ? ` · ${Number(r.visits).toLocaleString()}v` : '';
    return `${r.score}%${visits}`;
  }
  return formatEngineScore(r.score);
}

function depthFeedSignature(depthLog) {
  if (!depthLog?.length) {
    return '';
  }
  const tail = [...depthLog].sort((a, b) => b.depth - a.depth).slice(0, 6);
  return tail.map((e) => `${e.depth}:${e.score}:${e.nodes}`).join('|');
}

function rootsSignature(rootMoves, payload) {
  if (!rootMoves?.length) {
    return '';
  }
  const sortKey = usesRollouts(payload)
    ? (r) => r.winRate ?? r.score ?? 0
    : (r) => r.score ?? 0;
  return [...rootMoves]
    .sort((a, b) => sortKey(b) - sortKey(a))
    .slice(0, 4)
    .map((r) => `${r.move}:${formatRootMoveScore(r, payload)}`)
    .join('|');
}

function thinkCardShellHtml(seatIndex) {
  const colorClass = seatIndex === 0 ? 'think-card--white' : 'think-card--black';
  return `
    <article class="think-card ${colorClass} think-card--idle" data-seat="${seatIndex}">
      <header class="think-card__head">
        <span class="think-card__seat">${playerColorName(seatIndex + 1)}</span>
        <span class="think-card__engine" data-field="engine">—</span>
        <span class="think-card__pulse" data-field="pulse" hidden>live</span>
      </header>

      <div class="think-card__played think-card__slot" data-field="played-slot">
        <span class="think-card__played-label">played</span>
        <span class="think-card__played-move" data-field="played-move">&nbsp;</span>
        <span class="think-card__ply" data-field="played-ply"></span>
      </div>

      <div class="think-card__hero think-card__hero--even" data-field="hero">
        <div class="think-card__hero-split">
          <div class="think-card__hero-side think-card__hero-side--eval">
            <div class="think-card__hero-value" data-field="hero-left">…</div>
            <div class="think-card__hero-label" data-field="hero-left-label">—</div>
          </div>
          <div class="think-card__hero-divider" aria-hidden="true"></div>
          <div class="think-card__hero-side think-card__hero-side--depth">
            <div class="think-card__hero-value think-card__hero-value--depth" data-field="hero-right">…</div>
            <div class="think-card__hero-label" data-field="hero-right-label">depth</div>
          </div>
        </div>
      </div>

      <div class="think-card__depth-feed think-card__slot" data-field="depth-feed" hidden>
        <div class="think-card__depth-title" data-field="depth-title">Depth</div>
        <ul class="think-card__depth-list" data-field="depth-list"></ul>
      </div>

      <div class="think-card__pv-lead think-card__slot" data-field="pv-lead" hidden>
        <span class="think-card__pv-lead-label">PV</span>
        <span class="think-card__pv-lead-move" data-field="pv-lead-move"></span>
      </div>

      <div class="think-card__race" title="Shortest-path steps to goal">
        <div class="think-card__race-track">
          <div class="think-card__race-white" data-field="race-bar" style="width:50%"></div>
        </div>
        <div class="think-card__race-meta">
          <span class="think-card__race-label think-card__race-label--even" data-field="race-label">—</span>
          <span class="think-card__race-dist" data-field="race-dist"></span>
        </div>
      </div>

      <div class="think-card__pv think-card__slot" data-field="pv-block" hidden>
        <span class="think-card__pv-label">line</span>
        <span class="think-card__pv-text" data-field="pv-text"></span>
      </div>

      <div class="think-card__roots-block think-card__slot" data-field="roots-block" hidden>
        <div class="think-card__roots-title" data-field="roots-title">Top lines</div>
        <ul class="think-card__roots-list" data-field="roots-list"></ul>
      </div>

      <div class="think-card__budget" data-field="budget"></div>
    </article>`;
}

function ensureThinkCardsHost(container) {
  const playPanel = container.querySelector('.play-panel');
  if (!playPanel) {
    return null;
  }
  let host = playPanel.querySelector('[data-think-cards-host]');
  if (!host) {
    playPanel.insertAdjacentHTML('beforeend', '<div class="engine-think-cards-host" data-think-cards-host></div>');
    host = playPanel.querySelector('[data-think-cards-host]');
  }
  let root = host.querySelector('.engine-think-cards');
  if (!root) {
    host.innerHTML = '<div class="engine-think-cards"></div>';
    root = host.querySelector('.engine-think-cards');
  }
  for (const seat of [0, 1]) {
    if (!root.querySelector(`[data-seat="${seat}"]`)) {
      root.insertAdjacentHTML('beforeend', thinkCardShellHtml(seat));
    }
  }
  return root;
}

function patchDepthFeed(card, payload) {
  const block = card.querySelector('[data-field="depth-feed"]');
  const list = card.querySelector('[data-field="depth-list"]');
  const title = card.querySelector('[data-field="depth-title"]');
  if (!block || !list) {
    return;
  }

  const show = !usesRollouts(payload) && payload.depthLog?.length > 0;
  block.hidden = !show;
  if (!show) {
    return;
  }

  setText(title, payload.live ? 'Depth (live)' : 'Depth');
  const sig = depthFeedSignature(payload.depthLog);
  if (card.dataset.depthSig === sig) {
    return;
  }
  card.dataset.depthSig = sig;

  const rows = [...payload.depthLog]
    .sort((a, b) => b.depth - a.depth)
    .slice(0, 6)
    .map((entry) => {
      const score = entry.score != null ? formatEngineScore(entry.score) : '—';
      const nodes = entry.nodes > 0 ? ` · ${Number(entry.nodes).toLocaleString()}n` : '';
      return `
        <li class="think-card__depth-row">
          <span class="think-card__depth-num">d${entry.depth}</span>
          <span class="think-card__depth-score">${escapeHtml(score)}</span>
          <span class="think-card__depth-nodes">${escapeHtml(nodes)}</span>
        </li>`;
    })
    .join('');
  list.innerHTML = rows;
}

function patchRoots(card, payload) {
  const block = card.querySelector('[data-field="roots-block"]');
  const list = card.querySelector('[data-field="roots-list"]');
  const title = card.querySelector('[data-field="roots-title"]');
  if (!block || !list) {
    return;
  }

  const show = payload.rootMoves?.length > 0;
  block.hidden = !show;
  if (!show) {
    return;
  }

  setText(title, usesRollouts(payload) ? 'Top MCTS moves' : 'Top lines');
  const sig = rootsSignature(payload.rootMoves, payload);
  if (card.dataset.rootsSig === sig) {
    return;
  }
  card.dataset.rootsSig = sig;

  const sortKey = usesRollouts(payload)
    ? (r) => r.winRate ?? r.score ?? 0
    : (r) => r.score ?? 0;
  const sorted = [...payload.rootMoves].sort((a, b) => sortKey(b) - sortKey(a)).slice(0, 4);
  list.innerHTML = sorted
    .map((r, i) => {
      const score = formatRootMoveScore(r, payload);
      const best = i === 0 ? ' think-card__root-row--best' : '';
      return `
        <li class="think-card__root-row${best}">
          <span class="think-card__root-move">${escapeHtml(r.move)}</span>
          <span class="think-card__root-score">${escapeHtml(score)}</span>
        </li>`;
    })
    .join('');
}

function patchThinkCard(card, seatIndex, payload) {
  const hero = heroMetric(payload);
  const bar = raceBar(payload.whiteDist, payload.blackDist);
  const budget = formatBudgetLine(payload);
  const distText =
    payload.whiteDist != null ? `path W${payload.whiteDist} · B${payload.blackDist}` : '';
  const pvLead = payload.live ? pvHeadline(payload.pv) : '';

  card.classList.toggle('think-card--live', !!payload.live);
  card.classList.toggle('think-card--idle', !!payload.idle);

  const pulse = card.querySelector('[data-field="pulse"]');
  if (pulse) {
    pulse.hidden = !payload.live;
  }

  setText(card.querySelector('[data-field="engine"]'), payload.engine ?? '—');

  const playedSlot = card.querySelector('[data-field="played-slot"]');
  const hasPlayed = Boolean(payload.move);
  if (playedSlot) {
    playedSlot.classList.toggle('think-card__slot--visible', hasPlayed);
    playedSlot.classList.toggle('think-card__slot--hidden', !hasPlayed);
  }
  if (hasPlayed) {
    setText(card.querySelector('[data-field="played-move"]'), payload.move);
    setText(card.querySelector('[data-field="played-ply"]'), payload.ply ? `ply ${payload.ply}` : '');
  }

  const heroEl = card.querySelector('[data-field="hero"]');
  if (heroEl) {
    heroEl.classList.remove('think-card__hero--good', 'think-card__hero--bad', 'think-card__hero--even');
    heroEl.classList.add(`think-card__hero--${hero.tone}`);
  }
  setText(card.querySelector('[data-field="hero-left"]'), hero.left);
  setText(card.querySelector('[data-field="hero-right"]'), hero.right);
  setText(
    card.querySelector('[data-field="hero-left-label"]'),
    hero.mode === 'winrate' ? 'win rate' : hero.mode === 'eval' ? 'eval (sq)' : hero.mode === 'searching' ? 'searching' : '—',
  );
  setText(card.querySelector('[data-field="hero-right-label"]'), hero.rightLabel);

  patchDepthFeed(card, payload);

  const pvLeadBlock = card.querySelector('[data-field="pv-lead"]');
  const showPvLead = Boolean(payload.live && pvLead);
  if (pvLeadBlock) {
    pvLeadBlock.hidden = !showPvLead;
  }
  if (showPvLead) {
    setText(card.querySelector('[data-field="pv-lead-move"]'), pvLead);
  }

  const raceBarEl = card.querySelector('[data-field="race-bar"]');
  if (raceBarEl) {
    raceBarEl.style.width = `${bar.pct}%`;
  }
  const raceLabel = card.querySelector('[data-field="race-label"]');
  if (raceLabel) {
    raceLabel.textContent = bar.label;
    raceLabel.classList.remove('think-card__race-label--white', 'think-card__race-label--black', 'think-card__race-label--even');
    raceLabel.classList.add(`think-card__race-label--${bar.side}`);
  }
  setText(card.querySelector('[data-field="race-dist"]'), distText);

  const pvBlock = card.querySelector('[data-field="pv-block"]');
  const showPv = Boolean(payload.pv && !(payload.live && pvLead));
  if (pvBlock) {
    pvBlock.hidden = !showPv;
  }
  if (showPv) {
    setText(card.querySelector('[data-field="pv-text"]'), payload.pv);
  }

  patchRoots(card, payload);
  setText(card.querySelector('[data-field="budget"]'), budget);
}

/** Legacy string render — prefer `updateEngineThinkCards` (in-place DOM). */
export function renderEngineThinkCards(state) {
  return '<div class="engine-think-cards-host" data-think-cards-host></div>';
}

export function updateEngineThinkCards(container, state) {
  const root = ensureThinkCardsHost(container);
  if (!root) {
    return;
  }
  for (const seat of [0, 1]) {
    const card = root.querySelector(`[data-seat="${seat}"]`);
    const payload = thinkPayloadForSeat(state, seat);
    if (card && payload) {
      patchThinkCard(card, seat, payload);
    }
  }
}
