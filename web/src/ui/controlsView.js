import {
  STRENGTH_LEVEL_PRESETS,
  TIME_TO_MOVE_PRESETS,
  formatVisits,
  formatWallClock,
} from '../lib/timeControl.js';
import { renderDiscreteSlider } from './discreteSlider.js';
import { wireRangeSlider } from './sliderWire.js';
import './scrapedSlider.css';

export function renderControls(container, state, controller) {
  const { settings, aiThinking, playerAiSettingsUi, playerOptionGroups, searchInfoLine } = state;
  const [p1Ui, p2Ui] = playerAiSettingsUi ?? [];

  container.innerHTML = `
    <section class="controls-card">
      <h1 class="app-title">Quoridor AI</h1>
      <p class="app-subtitle">Play · Human vs Ishtar, Ka, or local MCTS</p>

      <div class="control-group">
        <label class="control-label">Player 1 (moves first)</label>
        ${renderPlayerSelect('player1', settings.players[0], playerOptionGroups)}
        ${renderPlayerAiSettings(p1Ui, 1)}
      </div>

      <div class="control-group">
        <label class="control-label">Player 2</label>
        ${renderPlayerSelect('player2', settings.players[1], playerOptionGroups)}
        ${renderPlayerAiSettings(p2Ui, 2)}
      </div>

      <div class="button-row">
        <button class="btn btn--primary" data-action="new-game">New Game</button>
        <button class="btn" data-action="undo" ${aiThinking ? 'disabled' : ''}>Undo</button>
      </div>

      <div class="toggle-group">
        <label class="toggle"><input type="checkbox" data-toggle="rotate" ${settings.rotateBoard ? 'checked' : ''} /> Rotate board</label>
        <label class="toggle"><input type="checkbox" data-toggle="coordinates" ${settings.displayCoordinates ? 'checked' : ''} /> Coordinates</label>
        <label class="toggle"><input type="checkbox" data-toggle="walls" ${settings.displayRemainingWalls ? 'checked' : ''} /> Wall count</label>
        <label class="toggle"><input type="checkbox" data-toggle="eval" ${settings.displayEvalBar ? 'checked' : ''} /> Eval bar</label>
      </div>

      <div class="status-panel">
        <div class="status-line"><span>Turn</span><strong>Player ${state.playerToMove}</strong></div>
        <div class="status-line"><span>Eval (P1)</span><strong>${Math.round((state.eval.p1 ?? 0.5) * 100)}%</strong></div>
        ${searchInfoLine ? `<div class="status-line status-line--muted"><span>AI</span><strong>${escapeHtml(searchInfoLine)}</strong></div>` : ''}
        ${
          state.eval.pv?.length
            ? `<div class="pv-line">PV: ${state.eval.pv.map((move) => (move.coordinate ? formatMove(move) : '?')).join(' ')}</div>`
            : ''
        }
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

  container.querySelector('[data-action="new-game"]')?.addEventListener('click', () => {
    controller.newGame();
  });
  container.querySelector('[data-action="undo"]')?.addEventListener('click', () => {
    controller.undo();
  });

  container.querySelector('[data-toggle="rotate"]')?.addEventListener('change', () => {
    controller.toggleRotateBoard();
  });
  container.querySelector('[data-toggle="coordinates"]')?.addEventListener('change', () => {
    controller.toggleDisplayCoordinates();
  });
  container.querySelector('[data-toggle="walls"]')?.addEventListener('change', () => {
    controller.toggleDisplayRemainingWalls();
  });
  container.querySelector('[data-toggle="eval"]')?.addEventListener('change', () => {
    controller.toggleDisplayEvalBar();
  });
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
      controller.setPlayerVisitsBudget(playerNum, value, { silent: true });
      const label = container.querySelector(`[data-visits-label="${playerNum}"]`);
      if (label) {
        label.textContent = formatVisits(Number(value));
      }
    },
    refresh,
  );
}

function renderPlayerAiSettings(ui, playerNum) {
  if (!ui || ui.isHuman) {
    return '';
  }

  if (ui.isLocal) {
    const { min: tMin, max: tMax, step: tStep } = ui.wallclockRange;
    const { min: vMin, max: vMax, step: vStep } = ui.visitsRange;
    return `
      <div class="player-ai-settings">
        <label class="control-label control-label--sub">Time per move</label>
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
        <label class="control-label control-label--sub">Visit budget</label>
        <div class="time-slider-row">
          <input
            type="range"
            class="time-slider scraped-slider"
            data-setting="visits-${playerNum}"
            min="${vMin}"
            max="${vMax}"
            step="${vStep}"
            value="${ui.visitsBudget}"
          />
          <output class="time-slider-value" data-visits-label="${playerNum}">${formatVisits(ui.visitsBudget)}</output>
        </div>
        <p class="time-hint">${escapeHtml(ui.hint)}</p>
      </div>`;
  }

  return `
    <div class="player-ai-settings">
      ${renderDiscreteSlider({
        label: 'AI Strength',
        settingName: 'strength-level',
        playerNum,
        value: ui.strengthLevel,
        presets: STRENGTH_LEVEL_PRESETS,
      })}
      ${renderDiscreteSlider({
        label: 'AI Time',
        settingName: 'time-to-move',
        playerNum,
        value: ui.timeToMove,
        presets: TIME_TO_MOVE_PRESETS,
      })}
      <p class="time-hint">${escapeHtml(ui.hint)}</p>
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

function formatMove(action) {
  if (action.wallType) {
    const suffix = action.wallType === 'h' ? 'h' : 'v';
    return `${action.coordinate.column}${action.coordinate.row}${suffix}`;
  }
  return `${action.coordinate.column}${action.coordinate.row}`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}
