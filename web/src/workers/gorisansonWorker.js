/**
 * Gorisanson MCTS in a Web Worker — keeps UI responsive.
 * Supports fixed simulations (legacy) or wall-clock budget (preferred).
 */

import gameJs from '../../../_vendor/quoridor-mcts/src/js/game.js?raw';
import aiJs from '../../../_vendor/quoridor-mcts/src/js/ai.js?raw';

const bootstrap = new Function(
  'postMessage',
  'performance',
  `${gameJs}\n${aiJs}\n
  function chooseOpeningPawnMove(game) {
    if (game.turn >= 2) {
      return null;
    }
    const nextPosition = AI.chooseShortestPathNextPawnPosition(game);
    const pawnMoveTuple = nextPosition.getDisplacementPawnMoveTupleFrom(game.pawnOfTurn.position);
    if (pawnMoveTuple[1] === 0) {
      return [[nextPosition.row, nextPosition.col], null, null];
    }
    return null;
  }

  function searchForTime(game, uctConst, timeMs, maxSimulations) {
    const opening = chooseOpeningPawnMove(game);
    if (opening) {
      return { move: opening, simulations: 0 };
    }

    const mcts = new MonteCarloTreeSearch(game, uctConst);
    const started = performance.now();
    const deadline = started + timeMs;
    const batchSize = 50;
    let simulations = 0;
    let tick = 0;
    const simCap = Number.isFinite(maxSimulations) ? maxSimulations : Infinity;

    while (performance.now() < deadline && simulations < simCap) {
      const batch = Math.min(batchSize, simCap - simulations);
      mcts.search(batch);
      simulations += batch;
      tick += 1;
      if (tick % 5 === 0) {
        const elapsed = performance.now() - started;
        postMessage({ type: 'progress', value: Math.min(0.99, elapsed / timeMs) });
      }
    }

    const best = mcts.selectBestMove();
    return { move: best.move, simulations };
  }

  return { Game, AI, searchForTime };
  `,
);

const { Game, AI, searchForTime } = bootstrap(
  (msg) => {
    if (typeof msg === 'number') {
      self.postMessage({ type: 'progress', value: msg });
    }
  },
  performance,
);

self.onmessage = (event) => {
  const { gorisansonMoves, simulations, timeMs, maxSimulations, uctConst } = event.data;
  const game = new Game(true);
  for (const move of gorisansonMoves) {
    game.doMove(move, true);
  }

  if (game.winner !== null) {
    self.postMessage({ type: 'error', message: 'terminal position' });
    return;
  }

  if (Number.isFinite(timeMs) && timeMs > 0) {
    const result = searchForTime(game, uctConst ?? 0.2, timeMs, maxSimulations);
    self.postMessage({
      type: 'bestmove',
      move: result.move,
      simulations: result.simulations,
      timeMs,
    });
    return;
  }

  const ai = new AI(simulations, uctConst, false, true);
  const move = ai.chooseNextMove(game);
  self.postMessage({ type: 'bestmove', move, simulations });
};
