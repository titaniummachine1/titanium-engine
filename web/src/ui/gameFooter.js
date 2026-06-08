import { toAlgebraic } from '../lib/gameLogic.js';
import { playerColorName } from '../lib/playerColors.js';

export function renderGameFooter(container, state) {
  const { winner, playerToMove, actions, liveSearch, aiThinking, replay, uiMode, playReplayCode } =
    state;

  let turnText;
  if (uiMode === 'replay' && replay) {
    turnText = `Replay ply ${replay.index} / ${replay.total}`;
  } else if (winner) {
    turnText = `Game over — ${playerColorName(winner)} wins`;
  } else {
    turnText = `Turn: ${playerColorName(playerToMove)}`;
  }

  const moveText =
    actions.length === 0
      ? '—'
      : actions.map((action, index) => `${index + 1}. ${toAlgebraic(action)}`).join('  ');

  let liveLine = '';
  if (aiThinking && liveSearch) {
    const who = liveSearch.playerLabel ?? 'AI';
    const modeLabel = formatSearchMode(liveSearch.mode);
    if (liveSearch.mode) {
      liveLine = `${who}: ${modeLabel}`;
      if (liveSearch.searchDepth) {
        liveLine += ` depth=${liveSearch.searchDepth}`;
      }
      if (liveSearch.nodes) {
        liveLine += ` · ${liveSearch.nodes.toLocaleString()} nodes`;
      } else if (liveSearch.simulations) {
        liveLine += ` · ${liveSearch.simulations.toLocaleString()} sims`;
      }
      if (liveSearch.depthLog?.length) {
        const last = liveSearch.depthLog[liveSearch.depthLog.length - 1];
        const scoreStr = last.score > 0 ? '+' : '';
        liveLine += ` · eval=${scoreStr}${last.score}`;
      } else if (liveSearch.rootWinRate != null) {
        liveLine += ` · wr ${(liveSearch.rootWinRate * 100).toFixed(0)}%`;
      }
      if (liveSearch.bestMove) {
        liveLine += ` · best=${liveSearch.bestMove}`;
      }
    } else if (liveSearch.simulations) {
      liveLine = `${who}: thinking… ${liveSearch.simulations.toLocaleString()} sims`;
    } else {
      liveLine = `${who}: thinking…`;
    }
  }

  const replayBlock =
    playReplayCode && uiMode === 'play'
      ? `<div class="game-footer__replay"><span class="game-footer__replay-label">Replay code (paste in Replay tab):</span> <code class="game-footer__replay-code">${escapeHtml(playReplayCode)}</code></div>`
      : '';

  container.innerHTML = `
    <div class="game-footer__row game-footer__row--turn">
      <strong>${turnText}</strong>
      ${liveLine ? `<span class="game-footer__live">${escapeHtml(liveLine)}</span>` : ''}
    </div>
    <div class="game-footer__moves" title="${escapeHtml(moveText)}">${escapeHtml(moveText)}</div>
    ${replayBlock}
  `;
}

function formatSearchMode(mode) {
  const labels = {
    searching: 'searching',
    mcts: 'MCTS',
    minimax: 'αβ+LMR',
    hybrid: 'hybrid',
    race: 'win path',
    trivial: 'instant',
    converged: 'MCTS ✓',
    visits: 'MCTS cap',
    time: 'MCTS',
  };
  return labels[mode] ?? mode;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}
