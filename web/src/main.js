import { AppController } from './game/appController.js';
import { renderBoard } from './ui/boardView.js';
import { renderCatHint } from './ui/catHint.js';
import { renderControls, updateEngineThinkCards } from './ui/controlsView.js';
import { renderEvalBar } from './ui/evalBar.js';
import { renderGameFooter } from './ui/gameFooter.js';
import { renderPlayersPanel } from './ui/playerSetupView.js';

const appRoot = document.getElementById('app');
const controller = new AppController();

appRoot.innerHTML = `
  <div class="layout">
    <aside class="layout__players" id="players-root"></aside>
    <main class="layout__board" id="board-root">
      <div class="board-column">
        <div class="board-row">
          <aside class="board-row__eval" id="eval-root"></aside>
          <div class="board-row__grid" id="board-slot"></div>
        </div>
        <footer class="game-footer" id="game-footer"></footer>
      </div>
    </main>
    <aside class="layout__sidebar" id="controls-root"></aside>
  </div>
`;

const boardRoot = document.getElementById('board-root');
const boardSlot = document.getElementById('board-slot');
const controlsRoot = document.getElementById('controls-root');
const playersRoot = document.getElementById('players-root');
const evalRoot = document.getElementById('eval-root');
const footerRoot = document.getElementById('game-footer');

function renderBoardArea() {
  const state = controller.getState();
  renderEvalBar(evalRoot, state);
  renderBoard(boardSlot, state, controller);
  renderGameFooter(footerRoot, state);
  renderCatHint(boardRoot, state, controller);
}

function render() {
  const state = controller.getState();
  renderBoardArea();
  renderPlayersPanel(playersRoot, state, controller);
  renderControls(controlsRoot, state, controller);
}

function renderLiveSearch() {
  updateEngineThinkCards(controlsRoot, controller.getState());
}

controller.onChange = render;
controller.onLiveUpdate = renderLiveSearch;
render();
controller.maybeRequestAiMove();
