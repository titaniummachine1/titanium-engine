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
  name: 'Gorisanson MCTS',
  key: PlayerType.GorisansonMCTS,
  tooltip: 'Local MCTS — first boss (github.com/gorisanson/quoridor-ai)',
  uctConst: 0.2,
};

const PLACEHOLDER_ENGINES = [
  {
    kind: 'placeholder',
    name: 'Titanium (Rust)',
    key: PlayerType.Titanium,
    tooltip: 'Our engine — αβ search coming in episode 07',
    disabled: true,
  },
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
  return [GORISANSON_ENGINE, ...remote, ...PLACEHOLDER_ENGINES];
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
          label: 'Gorisanson MCTS',
          disabled: false,
          tooltip: GORISANSON_ENGINE.tooltip,
        },
        {
          value: PlayerType.Titanium,
          label: 'Titanium (soon)',
          disabled: true,
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

export function describeActiveSearchInfo(players, searchInfoByType, engineConfigs) {
  const aiTypes = players.filter((p) => p !== PlayerType.Human);
  const lines = aiTypes
    .map((playerType) =>
      describeSearchInfo(playerType, searchInfoByType[playerType], engineConfigs),
    )
    .filter(Boolean);
  return lines.join(' · ');
}

export function describeSearchInfo(playerType, searchInfo, engineConfigs) {
  if (!searchInfo || playerType === PlayerType.Human) {
    return '';
  }
  const config = engineConfigs.find((entry) => entry.key === playerType);
  if (config?.kind === 'local' && searchInfo.time != null) {
    const sims = searchInfo.simulations?.toLocaleString() ?? '?';
    return `Last think: ${formatWallClock(searchInfo.time / 1000)} · ${sims} sims`;
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
