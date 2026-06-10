import { playerColorName } from '../lib/playerColors.js';
import { encodeReplayFromActions } from '../lib/replayCode.js';
import { updateEngineThinkCards } from './engineThinkView.js';

export { updateEngineThinkCards };
import './scrapedSlider.css';

export function renderControls(container, state, controller) {
  const {
    settings,
    aiThinking,
    uiMode,
    replay,
  } = state;
  const engineErrorLines = Object.entries(state.engineErrors ?? {})
    .filter(([, message]) => Boolean(message))
    .map(([playerType, message]) => `${playerType}: ${message}`)
    .join(' | ');
  const isReplay = uiMode === 'replay';
  const isAnalysis = uiMode === 'analysis';
  const isPlay = uiMode === 'play';
  const catStatus = renderCatStatusLine(state);
  const lmrStatus = renderLmrStatusLine(state);

  container.innerHTML = `
    <section class="controls-card">
      <h1 class="app-title">Quoridor AI</h1>

      <div class="mode-tabs">
        <button type="button" class="mode-tab ${isPlay ? 'mode-tab--active' : ''}" data-ui-mode="play">Play</button>
        <button type="button" class="mode-tab ${isAnalysis ? 'mode-tab--active' : ''}" data-ui-mode="analysis">Analysis</button>
        <button type="button" class="mode-tab ${isReplay ? 'mode-tab--active' : ''}" data-ui-mode="replay">Replay</button>
      </div>

      ${isReplay ? renderReplayPanel(replay) : ''}
      ${isAnalysis ? renderAnalysisPanel(state) : ''}
      ${!isReplay ? renderBoardToggles(settings, catStatus, lmrStatus) : ''}

      <div class="play-panel ${isPlay ? '' : 'play-panel--hidden'}">
      <div class="button-row">
        <button class="btn btn--primary" data-action="new-game">New Game</button>
        <button class="btn" data-action="undo" ${aiThinking ? 'disabled' : ''}>Undo</button>
        <button class="btn" data-action="redo" ${aiThinking || !state.canRedo ? 'disabled' : ''}>Redo</button>
      </div>

      <div class="status-panel">
        <div class="status-line">
          <span>Turn</span>
          <strong>${state.winner ? `Over (${playerColorName(state.winner)})` : playerColorName(state.playerToMove)}</strong>
        </div>
        <div class="status-line">
          <span>Dist (W−B)</span>
          <strong>${formatDistanceEval(state.eval)}</strong>
        </div>
        ${engineErrorLines ? `<div class="status-line status-line--error"><span>Error</span><strong>${escapeHtml(engineErrorLines)}</strong></div>` : ''}
      </div>
      <div class="engine-think-cards-host" data-think-cards-host></div>
      </div>
    </section>
  `;

  updateEngineThinkCards(container, state);

  container.querySelectorAll('[data-ui-mode]').forEach((btn) => {
    btn.addEventListener('click', () => {
      controller.setUiMode(btn.dataset.uiMode);
    });
  });

  wireReplayPanel(container, controller);
  wireAnalysisPanel(container, controller);

  container.querySelector('[data-action="new-game"]')?.addEventListener('click', () => {
    controller.newGame();
  });
  container.querySelector('[data-action="undo"]')?.addEventListener('click', () => {
    controller.undo();
  });
  container.querySelector('[data-action="redo"]')?.addEventListener('click', () => {
    controller.redo();
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
  container.querySelector('[data-toggle="cat-vision"]')?.addEventListener('change', (event) => {
    controller.toggleCatVision(event.target.checked);
  });
  container.querySelector('[data-toggle="lmr-vision"]')?.addEventListener('change', (event) => {
    controller.toggleLmrVision(event.target.checked);
  });
  container.querySelector('[data-toggle="lmr-shallow"]')?.addEventListener('change', (event) => {
    controller.toggleLmrShallow(event.target.checked);
  });
}

function renderCatStatusLine(state) {
  if (!state.settings.showCatVision) {
    return '';
  }
  if (state.catVizLoading) {
    return 'Loading…';
  }
  if (state.catVizError) {
    return `Error`;
  }
  const cat = state.catViz;
  if (!cat) {
    return '';
  }
  return `W${cat.whiteDist} B${cat.blackDist}`;
}

export function renderLmrStatusLine(state) {
  if (!state.settings.showLmrVision) {
    return '';
  }
  if (state.lmrVizLoading) {
    return 'Loading…';
  }
  if (state.lmrVizError) {
    const msg = String(state.lmrVizError);
    return msg.length > 28 ? `${msg.slice(0, 26)}…` : msg;
  }
  const viz = state.lmrViz;
  if (!viz) {
    return '';
  }
  if (state.settings.lmrVisionShallow) {
    const n = viz.moves?.length ?? 0;
    return n ? `${viz.label ?? 'pre-search'} · ${n} mv` : (viz.label ?? 'pre-search');
  }
  const searched = viz.searchedCount ?? viz.moves?.filter((m) => m.searched).length ?? 0;
  const total = viz.moves?.length ?? 0;
  const depth = viz.searchDepth ?? '?';
  return total ? `search d${depth} · ${searched}/${total}` : `search d${depth}`;
}

/** Patch LMR status text during live search without re-rendering controls. */
export function updateLmrToggleStatus(container, state) {
  const el = container.querySelector('.toggle-group__lmr-status');
  if (!el) {
    return;
  }
  const line = renderLmrStatusLine(state);
  el.textContent = line;
}

function renderBoardToggles(settings, catStatus, lmrStatus) {
  const catNote = catStatus ? `<span class="toggle-group__cat-status">${escapeHtml(catStatus)}</span>` : '';
  const lmrNote = lmrStatus ? `<span class="toggle-group__lmr-status">${escapeHtml(lmrStatus)}</span>` : '';
  return `
    <div class="toggle-group toggle-group--board">
      <label class="toggle"><input type="checkbox" data-toggle="rotate" ${settings.rotateBoard ? 'checked' : ''} /> Rotate</label>
      <label class="toggle"><input type="checkbox" data-toggle="coordinates" ${settings.displayCoordinates ? 'checked' : ''} /> Coords</label>
      <label class="toggle"><input type="checkbox" data-toggle="walls" ${settings.displayRemainingWalls ? 'checked' : ''} /> Walls</label>
      <label class="toggle"><input type="checkbox" data-toggle="eval" ${settings.displayEvalBar ? 'checked' : ''} /> Eval</label>
      <label class="toggle toggle--cat"><input type="checkbox" data-toggle="cat-vision" ${settings.showCatVision ? 'checked' : ''} /> CAT ${catNote}</label>
      <label class="toggle toggle--lmr"><input type="checkbox" data-toggle="lmr-vision" ${settings.showLmrVision ? 'checked' : ''} /> LMR ${lmrNote}</label>
      <label class="toggle toggle--lmr-shallow ${settings.showLmrVision ? '' : 'toggle--hidden'}"><input type="checkbox" data-toggle="lmr-shallow" ${settings.lmrVisionShallow ? 'checked' : ''} ${settings.showLmrVision ? '' : 'disabled'} /> Shallow</label>
    </div>`;
}

function renderAnalysisPanel(state) {
  const code = encodeReplayFromActionsSafe(state.actions);

  return `
    <div class="analysis-panel">
      <label class="control-label">Position (paste moves or load from replay)</label>
      <textarea class="replay-input" data-analysis-input rows="3" placeholder="tq1 e2 e8 e3v …">${escapeHtml(code)}</textarea>
      <div class="button-row">
        <button type="button" class="btn btn--primary" data-action="load-analysis">Load position</button>
        <button type="button" class="btn" data-action="analysis-undo" ${state.aiThinking ? 'disabled' : ''}>Undo</button>
        <button type="button" class="btn" data-action="analysis-redo" ${state.aiThinking || !state.canRedo ? 'disabled' : ''}>Redo</button>
        <button type="button" class="btn" data-action="analysis-start">Start</button>
      </div>
      <p class="time-hint">Move either side on the board — human vs human. Undo/redo walks the move tree. Load any <code>tq1</code> line to debug a position. Toggle <strong>CAT vision</strong> to overlay heat on the board.</p>
    </div>`;
}

function encodeReplayFromActionsSafe(actions) {
  if (!actions?.length) {
    return '';
  }
  return encodeReplayFromActions(actions);
}

function wireAnalysisPanel(container, controller) {
  container.querySelector('[data-action="load-analysis"]')?.addEventListener('click', () => {
    const text = container.querySelector('[data-analysis-input]')?.value ?? '';
    try {
      controller.loadAnalysisPosition(text);
    } catch (err) {
      window.alert(err.message ?? String(err));
    }
  });

  container.querySelector('[data-action="analysis-start"]')?.addEventListener('click', () => {
    controller.newGame();
    controller.setUiMode('analysis');
  });

  container.querySelector('[data-action="analysis-undo"]')?.addEventListener('click', () => {
    controller.undo();
  });
  container.querySelector('[data-action="analysis-redo"]')?.addEventListener('click', () => {
    controller.redo();
  });
}

function renderReplayPanel(replay) {
  const index = replay?.index ?? 0;
  const total = replay?.total ?? 0;
  const code = replay?.code ?? '';
  const metaLine = replay?.meta
    ? `<p class="replay-meta">${escapeHtml(JSON.stringify(replay.meta))}</p>`
    : '';

  return `
    <div class="replay-panel">
      <label class="control-label">Paste terminal replay code</label>
      <textarea class="replay-input" data-replay-input rows="4" placeholder="tq1 e2 e8 e3 …">${escapeHtml(code)}</textarea>
      ${metaLine}
      <div class="button-row">
        <button type="button" class="btn btn--primary" data-action="load-replay">Load</button>
        <button type="button" class="btn btn--accent" data-action="continue-replay" ${total ? '' : 'disabled'}>Play from here</button>
        <button type="button" class="btn" data-action="copy-replay" ${code ? '' : 'disabled'}>Copy</button>
      </div>
      <div class="replay-scrub">
        <button type="button" class="btn btn--icon" data-action="replay-start" title="Start" ${total ? '' : 'disabled'}>⏮</button>
        <button type="button" class="btn btn--icon" data-action="replay-prev" ${total ? '' : 'disabled'}>◀</button>
        <input type="range" class="replay-slider" data-replay-slider min="0" max="${total}" value="${index}" ${total ? '' : 'disabled'} />
        <button type="button" class="btn btn--icon" data-action="replay-next" ${total ? '' : 'disabled'}>▶</button>
        <button type="button" class="btn btn--icon" data-action="replay-end" title="End" ${total ? '' : 'disabled'}>⏭</button>
      </div>
      <p class="replay-status">Ply <strong>${index}</strong> / ${total}${total ? ` · ${replayStatusLabel(replay)}` : ' — load a code'}</p>
      <p class="time-hint">Terminal prints <code>tq1 …</code> after each benchmark game. Paste here to step through on the board. Use <strong>Play from here</strong> to leave replay mode and move as human. Supports <code>e3v</code> and <code>ve3</code> wall notation.</p>
    </div>`;
}

function replayStatusLabel(replay) {
  if (!replay || replay.total === 0) {
    return '';
  }
  if (replay.index === 0) {
    return 'start position';
  }
  if (replay.index >= replay.total) {
    return 'final position';
  }
  return `after move ${replay.index}`;
}

function wireReplayPanel(container, controller) {
  container.querySelector('[data-action="continue-replay"]')?.addEventListener('click', () => {
    controller.continueFromReplay();
  });

  container.querySelector('[data-action="load-replay"]')?.addEventListener('click', () => {
    const text = container.querySelector('[data-replay-input]')?.value ?? '';
    try {
      controller.loadReplay(text);
    } catch (err) {
      window.alert(err.message ?? String(err));
    }
  });

  container.querySelector('[data-action="copy-replay"]')?.addEventListener('click', async () => {
    const code = controller.exportReplayCode();
    try {
      await navigator.clipboard.writeText(code);
    } catch {
      window.prompt('Copy replay code:', code);
    }
  });

  container.querySelector('[data-action="replay-prev"]')?.addEventListener('click', () => {
    controller.replayStep(-1);
  });
  container.querySelector('[data-action="replay-next"]')?.addEventListener('click', () => {
    controller.replayStep(1);
  });
  container.querySelector('[data-action="replay-start"]')?.addEventListener('click', () => {
    controller.setReplayIndex(0);
  });
  container.querySelector('[data-action="replay-end"]')?.addEventListener('click', () => {
    const total = controller.replay?.actions.length ?? 0;
    controller.setReplayIndex(total);
  });

  const slider = container.querySelector('[data-replay-slider]');
  slider?.addEventListener('input', () => {
    controller.setReplayIndex(Number(slider.value));
  });
}

function formatDistanceEval(evalState) {
  const w = evalState.whiteDist;
  const b = evalState.blackDist;
  if (!Number.isFinite(w) || !Number.isFinite(b)) {
    return `${Math.round((evalState.p1 ?? 0.5) * 100)}%`;
  }
  const margin = evalState.margin ?? b - w;
  const sign = margin > 0 ? '+' : '';
  return `W${w} B${b} (${sign}${margin})`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}
