import {
  STRENGTH_LEVEL_PRESETS,
  TIME_TO_MOVE_PRESETS,
  formatVisitsCap,
  formatWallClock,
  visitsFromSliderPosition,
} from '../lib/timeControl.js';
import { playerColorLabel, playerColorName } from '../lib/playerColors.js';
import { renderDiscreteSlider } from './discreteSlider.js';
import { wireRangeSlider } from './sliderWire.js';
import './scrapedSlider.css';

export function renderPlayersPanel(container, state, controller) {
  const { settings, playerAiSettingsUi, playerOptionGroups, uiMode } = state;
  const isPlay = uiMode === 'play';
  const [p1Ui, p2Ui] = playerAiSettingsUi ?? [];

  container.innerHTML = `
    <section class="players-card ${isPlay ? '' : 'players-card--dim'}">
      <div class="player-slot player-slot--white">
        <label class="control-label">${playerColorLabel(1)}</label>
        ${renderPlayerSelect('player1', settings.players[0], playerOptionGroups)}
        ${renderPlayerAiSettings(p1Ui, 1)}
      </div>
      <div class="player-slot player-slot--black">
        <label class="control-label">${playerColorName(2)}</label>
        ${renderPlayerSelect('player2', settings.players[1], playerOptionGroups)}
        ${renderPlayerAiSettings(p2Ui, 2)}
      </div>
    </section>
  `;

  container.querySelector('[data-setting="player1"]')?.addEventListener('change', (event) => {
    controller.setPlayer(1, event.target.value);
  });
  container.querySelector('[data-setting="player2"]')?.addEventListener('change', (event) => {
    controller.setPlayer(2, event.target.value);
  });

  wirePlayerAiSettings(container, controller, 1);
  wirePlayerAiSettings(container, controller, 2);
}

function wirePlayerAiSettings(container, controller, playerNum) {
  const refresh = () => controller.onChange?.();

  wireRangeSlider(
    container,
    `[data-setting="strength-level-${playerNum}"]`,
    (value) => controller.setPlayerStrengthLevel(playerNum, value, { silent: true }),
    refresh,
  );

  wireRangeSlider(
    container,
    `[data-setting="time-to-move-${playerNum}"]`,
    (value) => controller.setPlayerTimeToMove(playerNum, value, { silent: true }),
    refresh,
  );

  wireRangeSlider(
    container,
    `[data-setting="wallclock-${playerNum}"]`,
    (value) => {
      controller.setPlayerWallClock(playerNum, value, { silent: true });
      const label = container.querySelector(`[data-wallclock-label="${playerNum}"]`);
      if (label) {
        label.textContent = formatWallClock(Number(value));
      }
    },
    refresh,
  );

  wireRangeSlider(
    container,
    `[data-setting="visits-${playerNum}"]`,
    (value) => {
      const visits = visitsFromSliderPosition(value);
      controller.setPlayerVisitsBudget(playerNum, visits, { silent: true });
      const label = container.querySelector(`[data-visits-label="${playerNum}"]`);
      if (label) {
        label.textContent = formatVisitsCap(visits);
      }
    },
    refresh,
  );
}

function renderPlayerAiSettings(ui, playerNum) {
  if (!ui || ui.isHuman) {
    return '';
  }

  if (ui.isLocalMcts) {
    const { min: tMin, max: tMax, step: tStep } = ui.wallclockRange;
    const { min: vMin, max: vMax, step: vStep } = ui.visitsRange;
    const isMinimax = ui.playerType === 'titanium-minimax';
    const budgetLabel = isMinimax ? 'Nodes' : 'Rollouts';
    return `
      <div class="player-ai-settings">
        ${ui.isTitanium
        ? renderDiscreteSlider({
          label: 'Strength',
          settingName: 'strength-level',
          playerNum,
          value: ui.strengthLevel,
          presets: STRENGTH_LEVEL_PRESETS,
        })
        : ''
      }
        <label class="control-label control-label--sub">Time</label>
        <div class="time-slider-row">
          <input
            type="range"
            class="time-slider scraped-slider"
            data-setting="wallclock-${playerNum}"
            min="${tMin}"
            max="${tMax}"
            step="${tStep}"
            value="${ui.wallClockSeconds}"
          />
          <output class="time-slider-value" data-wallclock-label="${playerNum}">${formatWallClock(ui.wallClockSeconds)}</output>
        </div>
        <label class="control-label control-label--sub">${budgetLabel}</label>
        <div class="time-slider-row">
          <input
            type="range"
            class="time-slider scraped-slider"
            data-setting="visits-${playerNum}"
            min="${vMin}"
            max="${vMax}"
            step="${vStep}"
            value="${ui.visitsSliderPosition}"
          />
          <output class="time-slider-value" data-visits-label="${playerNum}">${formatVisitsCap(ui.visitsBudget)}</output>
        </div>
      </div>`;
  }

  return `
    <div class="player-ai-settings">
      ${renderDiscreteSlider({
    label: 'Strength',
    settingName: 'strength-level',
    playerNum,
    value: ui.strengthLevel,
    presets: STRENGTH_LEVEL_PRESETS,
  })}
      ${renderDiscreteSlider({
    label: 'Time',
    settingName: 'time-to-move',
    playerNum,
    value: ui.timeToMove,
    presets: TIME_TO_MOVE_PRESETS,
  })}
    </div>`;
}

function renderPlayerSelect(name, value, groups) {
  const options = groups
    .map(
      (group) => `
      <optgroup label="${escapeHtml(group.label)}">
        ${group.options
          .map(
            (opt) =>
              `<option value="${opt.value}" ${opt.value === value ? 'selected' : ''} ${opt.disabled ? 'disabled' : ''}>${escapeHtml(opt.label)}</option>`,
          )
          .join('')}
      </optgroup>`,
    )
    .join('');

  return `<select class="control-select" data-setting="${name}">${options}</select>`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}
