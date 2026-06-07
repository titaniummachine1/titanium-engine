/**
 * Per-player AI settings — matches scraped quoridor-ai.netlify.app controls.
 *
 * Remote (Ishtar/Ka): AI Strength (Beg→Alpha) + AI Time (Immediate→Long) sliders.
 * Local (Gorisanson): wall-clock + visit-budget sliders.
 */

import { PlayerType, StrengthLevel, TimeToMove } from './engineConfig.js';

/** Scraped StrengthLevel slider — legacy label, kept for remote UI parity. */
export const STRENGTH_LEVEL_PRESETS = [
  { id: StrengthLevel.Beginner, label: 'Beg.' },
  { id: StrengthLevel.Intermediate, label: 'Inter.' },
  { id: StrengthLevel.Advanced, label: 'Adv.' },
  { id: StrengthLevel.Expert, label: 'Expert' },
  { id: StrengthLevel.Alpha, label: 'Alpha' },
];

/** Scraped timeToMove slider — drives visit count on cloud engines. */
export const TIME_TO_MOVE_PRESETS = [
  { id: TimeToMove.Intuition, label: 'Immediate' },
  { id: TimeToMove.Short, label: 'Short' },
  { id: TimeToMove.Medium, label: 'Medium' },
  { id: TimeToMove.Long, label: 'Long' },
];

export const WALL_CLOCK_RANGE = {
  min: 0.5,
  max: 60,
  step: 0.5,
  defaultSeconds: 3,
};

export const LOCAL_VISITS_RANGE = {
  min: 1_000,
  max: 60_000,
  step: 500,
  default: 7_500,
};

export function getEngineConfig(playerType, engineConfigs) {
  return engineConfigs.find((entry) => entry.key === playerType);
}

export function isRemoteEngine(playerType, engineConfigs) {
  return getEngineConfig(playerType, engineConfigs)?.kind === 'remote';
}

export function isLocalEngine(playerType, engineConfigs) {
  return getEngineConfig(playerType, engineConfigs)?.kind === 'local';
}

export function defaultPlayerAiSettings(playerType, engineConfigs) {
  if (playerType === PlayerType.Human) {
    return null;
  }
  if (isLocalEngine(playerType, engineConfigs)) {
    return {
      wallClockSeconds: WALL_CLOCK_RANGE.defaultSeconds,
      visitsBudget: LOCAL_VISITS_RANGE.default,
    };
  }
  return {
    strengthLevel: StrengthLevel.Alpha,
    timeToMove: TimeToMove.Short,
  };
}

export function formatWallClock(seconds) {
  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(0)}ms`;
  }
  if (Number.isInteger(seconds)) {
    return `${seconds}s`;
  }
  return `${seconds.toFixed(1)}s`;
}

export function formatVisits(n) {
  return Number(n).toLocaleString();
}

function strengthLevelLabel(level) {
  return STRENGTH_LEVEL_PRESETS.find((preset) => preset.id === level)?.label ?? 'Alpha';
}

function timeToMoveLabel(timeMode) {
  return TIME_TO_MOVE_PRESETS.find((preset) => preset.id === timeMode)?.label ?? 'Short';
}

export function describePlayerAiSettings(playerType, aiSettings, engineConfigs) {
  if (playerType === PlayerType.Human || !aiSettings) {
    return '';
  }
  const config = getEngineConfig(playerType, engineConfigs);
  if (!config) {
    return '';
  }

  if (isLocalEngine(playerType, engineConfigs)) {
    const time = formatWallClock(aiSettings.wallClockSeconds ?? WALL_CLOCK_RANGE.defaultSeconds);
    const visits = formatVisits(aiSettings.visitsBudget ?? LOCAL_VISITS_RANGE.default);
    return `${config.name}: ${time} · ≤${visits} rollouts`;
  }

  if (isRemoteEngine(playerType, engineConfigs) && config.visits) {
    const timeMode = aiSettings.timeToMove ?? TimeToMove.Short;
    const visits = config.visits[timeMode];
    const parallelism = config.settings?.parallelism?.[timeMode];
    const strength = strengthLevelLabel(aiSettings.strengthLevel ?? StrengthLevel.Alpha);
    const time = timeToMoveLabel(timeMode);
    let text = `${config.name}: ${strength} · ${time} (~${visits.toLocaleString()} visits)`;
    if (parallelism) {
      text += ` · ${parallelism} threads`;
    }
    return text;
  }

  return config.name;
}

export function describeAiSettingsForPlayers(players, playerAiSettings, engineConfigs) {
  const lines = players
    .map((playerType, index) =>
      describePlayerAiSettings(playerType, playerAiSettings[index], engineConfigs),
    )
    .filter(Boolean);
  return lines.length ? lines.join(' · ') : 'No AI selected.';
}
