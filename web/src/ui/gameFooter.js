import { formatCoordinate, toAlgebraic } from '../lib/gameLogic.js';
import { encodeReplayFromActions } from '../lib/replayCode.js';
import { playerColorName } from '../lib/playerColors.js';
import { formatVisits, formatWallClock, TIME_TO_MOVE_PRESETS, STRENGTH_LEVEL_PRESETS } from '../lib/timeControl.js';

const SETTINGS_FIELD_LABELS = {
  wallClockSeconds: 'wall clock',
  visitsBudget: 'visits cap',
  timeToMove: 'time preset',
  strength: 'strength',
};

export function renderGameFooter(container, state) {
  const {
    winner,
    playerToMove,
    actions,
    replay,
    uiMode,
    moveThinkLog,
  } = state;

  let turnText;
  if (uiMode === 'replay' && replay) {
    turnText = `Replay ply ${replay.index} / ${replay.total}`;
  } else if (winner) {
    turnText = `Game over — ${playerColorName(winner)} wins`;
  } else {
    turnText = `Turn: ${playerColorName(playerToMove)}`;
  }

  const moveText =
    actions.length === 0 ? '—' : actions.map((action) => toAlgebraic(action)).join(' ');

  const hasMoves = actions.length > 0;
  const hasReport = hasMoves || moveThinkLog?.length > 0;

  container.innerHTML = `
    <div class="game-footer__row game-footer__row--turn">
      <strong>${turnText}</strong>
    </div>
    <div class="game-footer__moves" title="${escapeHtml(moveText)}">${escapeHtml(moveText)}</div>
    <div class="game-footer__actions">
      <button type="button" class="btn btn--small" data-action="copy-game-code" ${hasMoves ? '' : 'disabled'}>Copy game</button>
      <button type="button" class="btn btn--small" data-action="copy-full-report" ${hasReport ? '' : 'disabled'}>Copy game report</button>
    </div>
  `;

  wireCopyButton(container, '[data-action="copy-game-code"]', () => buildGameCodeText(state), 'Copy game');
  wireCopyButton(container, '[data-action="copy-full-report"]', () => buildGameExportText(state), 'Copy game report');
}

function wireCopyButton(container, selector, getText, label) {
  const btn = container.querySelector(selector);
  if (!btn) {
    return;
  }
  btn.addEventListener('click', () => {
    copyText(getText());
    btn.textContent = 'Copied!';
    setTimeout(() => {
      btn.textContent = label;
    }, 1500);
  });
}

function copyText(text) {
  navigator.clipboard.writeText(text).catch(() => {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    document.execCommand('copy');
    document.body.removeChild(ta);
  });
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

function isMateScore(score) {
  return Math.abs(Number(score) || 0) >= 19_500;
}

function formatEngineScore(score) {
  if (score == null || !Number.isFinite(Number(score))) {
    return '?';
  }
  const n = Number(score);
  if (isMateScore(n)) {
    const sign = n > 0 ? '+' : '-';
    return `${sign}M${Math.max(0, 20_000 - Math.abs(n))}`;
  }
  const meters = n / 100;
  return `${meters > 0 ? '+' : ''}${meters.toFixed(2)}`;
}

function formatDepthLog(depthLog) {
  return depthLog
    .map((e) => {
      const pv = e.pv ? ` pv:${e.pv}` : '';
      return `d${e.depth}=${formatEngineScore(e.score)}${pv}`;
    })
    .join(' | ');
}

function isTitaniumThinkEntry(entry) {
  return entry.engine?.includes('Titanium');
}

/** Top root candidates for copied reports: `roots: d5=-991 W6/B5 g0; h3h=-803 W5/B6 g2` */
function formatThinkDuration(entry) {
  const ms = entry.thinkMs ?? entry.elapsedMs ?? entry.time;
  if (ms == null || !Number.isFinite(Number(ms))) {
    return '';
  }
  const n = Number(ms);
  if (n < 1000) {
    return ` ${Math.round(n)}ms`;
  }
  return ` ${(n / 1000).toFixed(2)}s`;
}

function formatRootMovesSummary(rootMoves) {
  if (!rootMoves?.length) {
    return '';
  }
  const roots = [...rootMoves]
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map((r) => `${r.move}=${formatEngineScore(r.score)} W${r.whiteDist}/B${r.blackDist} g${r.gain ?? 0}`)
    .join('; ');
  return ` roots: ${roots}`;
}

function formatSettingsValue(field, value) {
  if (value == null) {
    return '?';
  }
  if (field === 'wallClockSeconds') {
    return formatWallClock(Number(value));
  }
  if (field === 'visitsBudget') {
    return formatVisits(Number(value));
  }
  if (field === 'timeToMove') {
    return TIME_TO_MOVE_PRESETS.find((p) => p.id === value)?.label ?? String(value);
  }
  if (field === 'strength') {
    return STRENGTH_LEVEL_PRESETS.find((p) => p.id === value)?.label ?? String(value);
  }
  return String(value);
}

function formatSettingsChangelog(changelog) {
  if (!changelog?.length) {
    return '';
  }
  return changelog
    .map(
      (e) =>
        `  ply ${e.ply} · ${e.seat} · ${e.player}: ${SETTINGS_FIELD_LABELS[e.field] ?? e.field} ${formatSettingsValue(e.field, e.from)} → ${formatSettingsValue(e.field, e.to)}`,
    )
    .join('\n');
}

function formatThinkEntry(entry) {
  const who =
    entry.ply % 2 === 1 ? 'White' : 'Black';
  const engine = entry.engine ? ` [${entry.engine}]` : '';
  const budget = entry.budget ? ` budget=${entry.budget}` : '';
  const dist =
    entry.whiteDist != null && entry.blackDist != null
      ? ` W${entry.whiteDist} B${entry.blackDist}`
      : '';
  const think = formatThinkDuration(entry);

  if (entry.error) {
    return `ply${entry.ply} ${who}${engine} ERROR: ${entry.error}${budget}${dist}${think}`;
  }

  const isMcts =
    !isTitaniumThinkEntry(entry) &&
    entry.stoppedBy !== 'minimax' &&
    (entry.stoppedBy === 'mcts' ||
      entry.stoppedBy === 'time' ||
      entry.stoppedBy === 'visits' ||
      entry.stoppedBy === 'bridge' ||
      entry.stoppedBy === 'bridge-visits' ||
      entry.stoppedBy === 'forced' ||
      entry.stoppedBy === 'win-in-1' ||
      entry.stoppedBy === 'opening');

  const sims = entry.nodes > 0 ? ` ${entry.nodes.toLocaleString()}nodes` : '';
  const wr = entry.rootWinRate != null && isMcts
    ? ` wr=${(entry.rootWinRate * 100).toFixed(0)}%`
    : '';
  const rootCands =
    isTitaniumThinkEntry(entry) ? formatRootMovesSummary(entry.rootMoves) : '';

  if (isMcts && !entry.depthLog?.length) {
    const stopped = entry.stoppedBy ? ` (${entry.stoppedBy})` : '';
    return `ply${entry.ply} ${who}${engine} ${entry.move}${budget}${dist}${sims}${think}${wr}${stopped}${rootCands}`;
  }

  const depth = entry.searchDepth ? ` d${entry.searchDepth}` : '';
  const dlog =
    entry.depthLog?.length
      ? ' ' + formatDepthLog(entry.depthLog)
      : '';

  return `ply${entry.ply} ${who}${engine} ${entry.move}${budget}${dist}${depth}${sims}${think}${dlog}${rootCands}`;
}

function engineLabelForSlot(state, playerNum) {
  const playerType = state.settings?.players?.[playerNum - 1];
  const opt = state.playerOptions?.find((entry) => entry.value === playerType);
  return opt?.label ?? playerType ?? '?';
}

function formatMargin(margin) {
  if (margin == null || !Number.isFinite(margin)) {
    return '?';
  }
  return margin > 0 ? `+${margin}` : String(margin);
}

function raceVerdict(winner, loserDist, closestMargin) {
  if (!winner) {
    if (closestMargin != null && Math.abs(closestMargin) <= 1) {
      return 'live — race within 1 step';
    }
    if (closestMargin != null && Math.abs(closestMargin) <= 3) {
      return 'live — close race';
    }
    return 'in progress';
  }
  if (loserDist <= 1) {
    return 'photo finish — loser 0–1 steps from goal';
  }
  if (loserDist <= 3) {
    return 'close — loser within 3 steps of goal';
  }
  if (loserDist >= 8) {
    return 'blowout — loser far from goal';
  }
  return 'decisive';
}

function summarizeRaceFromLog(log) {
  let closestMargin = null;
  let maxWhiteLead = null;
  let maxBlackLead = null;
  for (const entry of log ?? []) {
    if (entry.whiteDist == null || entry.blackDist == null) {
      continue;
    }
    const margin = entry.blackDist - entry.whiteDist;
    if (closestMargin === null || Math.abs(margin) < Math.abs(closestMargin)) {
      closestMargin = margin;
    }
    if (maxWhiteLead === null || margin > maxWhiteLead) {
      maxWhiteLead = margin;
    }
    if (maxBlackLead === null || margin < maxBlackLead) {
      maxBlackLead = margin;
    }
  }
  return { closestMargin, maxWhiteLead, maxBlackLead };
}

function buildGameHeader(state) {
  const {
    winner,
    actions,
    playerToMove,
    playerPositions,
    wallsRemaining,
    eval: evalState,
    playReplayCode,
    timeBudgetHint,
    initialBudgetHint,
    settingsChangelog,
    moveThinkLog,
  } = state;

  const plies = actions?.length ?? 0;
  const whiteSq = playerPositions?.[0] ? formatCoordinate(playerPositions[0]) : '?';
  const blackSq = playerPositions?.[1] ? formatCoordinate(playerPositions[1]) : '?';
  const wDist = evalState?.whiteDist;
  const bDist = evalState?.blackDist;
  const margin = evalState?.margin;
  const wallsUsedW = wallsRemaining?.[0] != null ? 10 - wallsRemaining[0] : '?';
  const wallsUsedB = wallsRemaining?.[1] != null ? 10 - wallsRemaining[1] : '?';

  const { closestMargin, maxWhiteLead, maxBlackLead } = summarizeRaceFromLog(moveThinkLog);
  const loserDist =
    winner === 1 ? bDist : winner === 2 ? wDist : null;

  const lines = ['=== Quoridor game report ===', ''];

  if (winner) {
    lines.push(`Result: ${playerColorName(winner)} wins · ${plies} plies`);
  } else {
    lines.push(
      `Result: in progress · ply ${plies} · ${playerColorName(playerToMove)} to move`,
    );
  }

  lines.push(
    `White: ${engineLabelForSlot(state, 1)}`,
    `Black: ${engineLabelForSlot(state, 2)}`,
  );
  if (initialBudgetHint && initialBudgetHint !== timeBudgetHint) {
    lines.push(`Budget at start: ${initialBudgetHint}`);
  }
  if (timeBudgetHint) {
    lines.push(`Budget (final): ${timeBudgetHint}`);
  } else if (initialBudgetHint) {
    lines.push(`Budget: ${initialBudgetHint}`);
  }
  const changelogText = formatSettingsChangelog(settingsChangelog);
  if (changelogText) {
    lines.push('Settings changes during game:');
    lines.push(changelogText);
  }
  lines.push('');

  lines.push(
    `Final position: White=${whiteSq} Black=${blackSq}`,
    `Path distance: W=${wDist ?? '?'} B=${bDist ?? '?'} · margin=${formatMargin(margin)} (positive = White ahead)`,
    `Walls used: White=${wallsUsedW} Black=${wallsUsedB} · left W=${wallsRemaining?.[0] ?? '?'} B=${wallsRemaining?.[1] ?? '?'}`,
  );

  if (closestMargin != null) {
    lines.push(
      `Race swing: closest margin=${formatMargin(closestMargin)} · best White lead=${formatMargin(maxWhiteLead)} · best Black lead=${formatMargin(maxBlackLead)}`,
    );
  }

  lines.push(`Verdict: ${raceVerdict(winner, loserDist ?? 99, closestMargin)}`);

  if (playReplayCode) {
    lines.push('', `Replay: ${playReplayCode}`);
  } else if (actions?.length) {
    const compact = actions.map((a) => toAlgebraic(a)).join(' ');
    lines.push('', `Moves: ${compact}`);
  }

  if (state.aiThinking && state.thinkingPlayerType) {
    const seat = state.settings?.players?.indexOf(state.thinkingPlayerType);
    const who = seat === 0 ? 'White' : seat === 1 ? 'Black' : '?';
    const label = engineLabelForSlot(state, seat >= 0 ? seat + 1 : 1);
    lines.push(`Search: in progress — ${who} (${label}) thinking`);
  }

  return lines.join('\n');
}

function formatEngineErrorsBlock(state) {
  const lines = [];
  const errors = state.engineErrors ?? {};
  const status = state.engineStatus ?? {};
  const players = state.settings?.players ?? [];

  for (let i = 0; i < players.length; i++) {
    const msg = errors[players[i]];
    if (msg) {
      lines.push(
        `${engineLabelForSlot(state, i + 1)}: ${msg} (status=${status[players[i]] ?? 'error'})`,
      );
    }
  }

  for (const entry of state.moveThinkLog ?? []) {
    if (!entry.error) {
      continue;
    }
    const who = entry.ply % 2 === 1 ? 'White' : 'Black';
    const tagged = `ply${entry.ply} ${who}`;
    if (!lines.some((line) => line.includes(entry.error) && line.includes(tagged))) {
      lines.push(`${tagged} [${entry.engine}]: ${entry.error}`);
    }
  }

  if (!lines.length) {
    return '';
  }
  return `--- Engine errors ---\n${lines.join('\n')}`;
}

export function buildGameCodeText(state) {
  const { actions, winner } = state;
  if (!actions?.length) {
    return '';
  }
  const meta =
    winner != null
      ? {
        winner: winner === 1 ? 'white' : 'black',
        plies: actions.length,
      }
      : null;
  return encodeReplayFromActions(actions, meta);
}

export function buildThinkLogText(state) {
  const log = state.moveThinkLog;
  if (!log?.length) {
    return '';
  }
  return log.map(formatThinkEntry).join('\n');
}

export function buildGameExportText(state) {
  const header = buildGameHeader(state);
  const errorsBlock = formatEngineErrorsBlock(state);
  const log = state.moveThinkLog;
  const thinkSection = !log?.length
    ? '--- Think chain ---\n(no AI think log yet)'
    : `--- Think chain ---\n${log.map(formatThinkEntry).join('\n')}`;
  return [header, errorsBlock, thinkSection].filter(Boolean).join('\n\n');
}

export function renderThinkLogPanel(log) {
  if (!log?.length) {
    return '';
  }
  const rows = log
    .map((entry) => `<div class="think-log__row">${escapeHtml(formatThinkEntry(entry))}</div>`)
    .join('');
  return `
    <div class="think-log">
      <div class="think-log__header">Think log <span class="think-log__count">${log.length}</span></div>
      <div class="think-log__body" data-think-log-body>${rows}</div>
    </div>
  `;
}

export function pinThinkLogScroll(container, scrollTop) {
  const body = container.querySelector('[data-think-log-body]');
  if (body && scrollTop > 0) {
    body.scrollTop = scrollTop;
  }
}
