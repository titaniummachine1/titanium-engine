/**
 * Opponent registry — local, remote, and future competition targets.
 */

import { PlayerType, getEngineList } from './engineConfig.js';
import {
  STRENGTH_LEVEL_PRESETS,
  TIME_TO_MOVE_PRESETS,
  describeAiSettingsForPlayers,
  formatWallClock,
} from './timeControl.js';

export { STRENGTH_LEVEL_PRESETS, TIME_TO_MOVE_PRESETS };
/** @deprecated use TIME_TO_MOVE_PRESETS */
export const TIME_PRESETS = TIME_TO_MOVE_PRESETS;

const GORISANSON_ENGINE = {
  kind: 'local',
  name: 'Gorisanson (JS, original)',
  key: PlayerType.GorisansonMCTS,
  tooltip: 'Original JavaScript MCTS — first boss (github.com/gorisanson/quoridor-ai)',
  uctConst: 0.2,
};

const TITANIUM_ENGINE = {
  kind: 'titanium',
  name: 'Titanium (MCTS, Rust)',
  key: PlayerType.Titanium,
  engineMode: 'mcts',
  tooltip: 'Updated local Rust MCTS engine — `titanium genmove` (cargo build --release in engine/)',
};

const TITANIUM_MINIMAX_ENGINE = {
  kind: 'titanium',
  name: 'Titanium Hybrid (strongest)',
  key: PlayerType.TitaniumMinimax,
  engineMode: 'minimax',
  tooltip:
    'Full pipeline: MCTS opening/book → minimax+CAT v3 (`cargo build --release` in engine/)',
};

const PLACEHOLDER_ENGINES = [
  {
    kind: 'placeholder',
    name: 'pavlosdais (C αβ)',
    key: PlayerType.Pavlosdais,
    tooltip: 'Competition baseline — not wired yet',
    disabled: true,
  },
];

export function getAllEngineConfigs() {
  const remote = getEngineList().map((entry) => ({
    ...entry,
    kind: 'remote',
  }));
  return [GORISANSON_ENGINE, TITANIUM_ENGINE, TITANIUM_MINIMAX_ENGINE, ...remote, ...PLACEHOLDER_ENGINES];
}

export function getPlayerOptionGroups() {
  return [
    {
      label: 'Human',
      options: [{ value: PlayerType.Human, label: 'Human', disabled: false }],
    },
    {
      label: 'Local — beat these first',
      options: [
        {
          value: PlayerType.GorisansonMCTS,
          label: 'Gorisanson (JS, original)',
          disabled: false,
          tooltip: GORISANSON_ENGINE.tooltip,
        },
        {
          value: PlayerType.Titanium,
          label: 'Titanium (MCTS, Rust)',
          disabled: false,
          tooltip: TITANIUM_ENGINE.tooltip,
        },
        {
          value: PlayerType.TitaniumMinimax,
          label: 'Titanium Hybrid (strongest)',
          disabled: false,
          tooltip: TITANIUM_MINIMAX_ENGINE.tooltip,
        },
      ],
    },
    {
      label: 'Remote',
      options: [
        { value: PlayerType.IshtarV3, label: 'Ishtar', disabled: false },
        { value: PlayerType.KaAI, label: 'Ka', disabled: false },
      ],
    },
    {
      label: 'Competition (planned)',
      options: [
        { value: PlayerType.Pavlosdais, label: 'pavlosdais C', disabled: true },
      ],
    },
  ];
}

export function flattenPlayerOptions(groups) {
  return groups.flatMap((group) => group.options);
}

export function describeTimeBudget(players, playerAiSettings, engineConfigs) {
  return describeAiSettingsForPlayers(players, playerAiSettings, engineConfigs);
}

export function describeActiveSearchInfo(
  players,
  searchInfoByType,
  engineConfigs,
  { thinkingPlayerType = null, aiThinking = false } = {},
) {
  const aiTypes = players.filter((p) => p !== PlayerType.Human);
  const formatOne = (playerType) => {
    const line = describeSearchInfo(playerType, searchInfoByType[playerType], engineConfigs);
    if (!line) {
      return '';
    }
    if (aiThinking && thinkingPlayerType && playerType !== thinkingPlayerType) {
      return '';
    }
    if (aiTypes.length > 1 && !aiThinking) {
      const label = String(playerType).toLowerCase().includes('white') ? 'W' : 'B';
      return `${label} ${line}`;
    }
    return line;
  };
  return aiTypes.map(formatOne).filter(Boolean).join(' · ');
}

export function formatEngineScore(score) {
  if (score == null || !Number.isFinite(Number(score))) {
    return '?';
  }
  const n = Number(score);
  if (Math.abs(n) >= 19_500) {
    const sign = n > 0 ? '+' : '-';
    return `${sign}M${Math.max(0, 20_000 - Math.abs(n))}`;
  }
  const meters = n / 100;
  return `${meters > 0 ? '+' : ''}${meters.toFixed(2)}`;
}

const SEARCH_STOP_LABELS = {
  visits: 'hit cap',
  time: 'time',
  converged: 'converged',
  trivial: 'instant',
  opening: 'opening',
  minimax: 'minimax',
  mcts: 'MCTS',
  hybrid: 'hybrid',
  race: 'win path',
  searching: 'searching',
};

function pickSearchDepthSource(liveSearch, searchInfo) {
  if (liveSearch?.depthLog?.length > 0) {
    return { depthLog: liveSearch.depthLog, header: liveSearch, live: true };
  }
  if (searchInfo?.depthLog?.length > 0) {
    return { depthLog: searchInfo.depthLog, header: searchInfo, live: false };
  }
  if (liveSearch?.mode) {
    return { depthLog: null, header: liveSearch, live: true };
  }
  return null;
}

function buildSearchDepthHeader(header, { live }) {
  if (!header) {
    return '';
  }
  const parts = [];
  if (header.playerLabel) {
    parts.push(escapeHtml(header.playerLabel));
  }
  if (live) {
    parts.push('searching');
  } else if (header.time != null) {
    parts.push(formatWallClock(header.time / 1000));
  }
  if (header.nodes != null) {
    parts.push(`${Number(header.nodes).toLocaleString()} nodes`);
  } else if (header.simulations != null) {
    parts.push(`${Number(header.simulations).toLocaleString()} sims`);
  }
  if (header.whiteDist != null) {
    parts.push(`W${header.whiteDist} B${header.blackDist}`);
  }
  const stopKey = header.stoppedBy ?? header.mode;
  const stopLabel = SEARCH_STOP_LABELS[stopKey];
  if (stopLabel && !live) {
    parts.push(stopLabel);
  }
  if (!parts.length) {
    return '';
  }
  return `<div class="search-depth__header">${parts.join(' · ')}</div>`;
}

export function buildSearchDepthPanel(liveSearch, searchInfo) {
  const source = pickSearchDepthSource(liveSearch, searchInfo);
  if (!source) {
    return '';
  }
  const header = buildSearchDepthHeader(source.header, { live: source.live });
  if (!source.depthLog?.length) {
    return header;
  }

  const byDepth = new Map(source.depthLog.map((entry) => [entry.depth, entry]));
  const newestFirst = [...source.depthLog].sort((a, b) => b.depth - a.depth);
  const rows = newestFirst
    .map((entry) => {
      const prev = byDepth.get(entry.depth - 1);
      const delta =
        prev && Number.isFinite(entry.score) && Number.isFinite(prev.score)
          ? entry.score - prev.score
          : null;
      const deltaClass =
        delta == null ? '' : delta > 0 ? 'search-depth__delta--up' : delta < 0 ? 'search-depth__delta--down' : '';
      const deltaText =
        delta == null ? '' : ` (${delta > 0 ? '+' : ''}${formatEngineScore(delta)})`;
      const pv = entry.pv ? `<span class="search-depth__pv">${escapeHtml(entry.pv)}</span>` : '';
      return `<tr>
        <td class="search-depth__d">d${entry.depth}</td>
        <td class="search-depth__eval">${formatEngineScore(entry.score)}${deltaText ? `<span class="search-depth__delta ${deltaClass}">${escapeHtml(deltaText)}</span>` : ''}</td>
        <td class="search-depth__nodes">${Number(entry.nodes ?? 0).toLocaleString()}</td>
        <td class="search-depth__pvcell">${pv}</td>
      </tr>`;
    })
    .join('');
  const wrSource = source.live ? liveSearch : searchInfo;
  const wr =
    wrSource?.rootWinRate != null
      ? `<div class="search-depth__wr">MCTS win rate: <strong>${(wrSource.rootWinRate * 100).toFixed(1)}%</strong></div>`
      : '';
  return `${header}${wr}<table class="search-depth"><thead><tr><th>depth</th><th>eval</th><th>nodes</th><th>pv</th></tr></thead><tbody>${rows}</tbody></table>`;
}

/** Update depth panel in place during live search — avoids full sidebar re-render. */
export function updateSearchDepthPanel(container, state) {
  const host = container.querySelector('.status-panel');
  if (!host) {
    return;
  }
  const html = buildSearchDepthPanel(state.liveSearch, state.activeSearchInfo);
  const aiLine = host.querySelector('.status-line--search-info');
  if (aiLine) {
    aiLine.hidden = Boolean(html);
  }

  let panel = host.querySelector('.search-depth-panel');
  if (!html) {
    panel?.remove();
    return;
  }

  const pinTop = !panel || panel.scrollTop < 4;
  if (!panel) {
    panel = document.createElement('div');
    panel.className = 'search-depth-panel';
    host.appendChild(panel);
  }
  panel.innerHTML = html;
  if (pinTop) {
    panel.scrollTop = 0;
  }
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}

export function describeSearchInfo(playerType, searchInfo, engineConfigs) {
  if (!searchInfo || playerType === PlayerType.Human) {
    return '';
  }
  const config = engineConfigs.find((entry) => entry.key === playerType);
  if ((config?.kind === 'local' || config?.kind === 'titanium') && searchInfo.time != null) {
    const isMinimax = searchInfo.stoppedBy === 'minimax';
    const budgetLabel = isMinimax
      ? `${(searchInfo.nodes ?? 0).toLocaleString()} nodes`
      : `${searchInfo.simulations?.toLocaleString() ?? '?'} sims`;
    const winPart =
      !isMinimax && searchInfo.rootWinRate != null
        ? ` · wr ${(searchInfo.rootWinRate * 100).toFixed(0)}%`
        : '';
    const distPart =
      searchInfo.whiteDist != null
        ? ` · W${searchInfo.whiteDist} B${searchInfo.blackDist}`
        : '';
    const limit = SEARCH_STOP_LABELS[searchInfo.stoppedBy] ?? '';
    const suffix = limit ? ` (${limit})` : '';
    const profile =
      searchInfo.profileName && isMinimax ? ` · ${searchInfo.profileName}` : '';
    return `${formatWallClock(searchInfo.time / 1000)} · ${budgetLabel}${winPart}${distPart}${profile}${suffix}`;
  }
  if (config?.kind === 'remote') {
    const parts = [];
    if (searchInfo.time != null) {
      parts.push(`${searchInfo.time}ms`);
    }
    if (searchInfo.visits != null) {
      parts.push(`${searchInfo.visits.toLocaleString()} visits`);
    }
    return parts.length ? `Last think: ${parts.join(' · ')}` : '';
  }
  return '';
}
