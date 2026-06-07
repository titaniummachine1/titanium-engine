/**
 * Local gorisanson MCTS — same surface as EngineClient for AppController.
 */

import GorisansonWorker from '../workers/gorisansonWorker.js?worker';
import { actionToGorisansonMove, gorisansonMoveToAction } from './gorisansonBridge.js';
import { LOCAL_VISITS_RANGE } from './timeControl.js';

export class GorisansonEngineClient {
  constructor(engineConfig) {
    this.config = engineConfig;
    this.worker = null;
    this.gorisansonMoves = [];
    this.pendingSearch = null;

    this.onInfo = null;
    this.onBestMove = null;
    this.onStatus = null;
    this.onError = null;
  }

  destroy() {
    this.worker?.terminate();
    this.worker = null;
    this.gorisansonMoves = [];
    this.setStatus('idle');
  }

  resetConnection() {
    this.destroy();
    this.gorisansonMoves = [];
  }

  makeMoves(actions) {
    for (const action of actions) {
      this.gorisansonMoves.push(actionToGorisansonMove(action));
    }
    this.setStatus('idle');
  }

  requestMove({ aiSettings, moveHistory, isFreshGame }) {
    if (isFreshGame) {
      this.gorisansonMoves = [];
    } else if (moveHistory?.length) {
      this.gorisansonMoves = moveHistory.map(actionToGorisansonMove);
    }

    const timeMs = Math.round((aiSettings?.wallClockSeconds ?? 3) * 1000);
    const maxSimulations = aiSettings?.visitsBudget ?? LOCAL_VISITS_RANGE.default;

    const runSearch = () => {
      this.setStatus('searching');
      const started = performance.now();

      this.worker?.terminate();
      this.worker = new GorisansonWorker();

      this.worker.onmessage = (event) => {
        const data = event.data;
        if (data.type === 'progress') {
          return;
        }
        if (data.type === 'error') {
          this.setStatus('error');
          this.onError?.(new Error(data.message));
          return;
        }
        if (data.type === 'bestmove') {
          const elapsed = performance.now() - started;
          this.onInfo?.({
            time: elapsed,
            simulations: data.simulations,
            progress: 1,
          });
          this.setStatus('idle');
          const action = gorisansonMoveToAction(data.move);
          this.gorisansonMoves.push(data.move);
          this.onBestMove?.(action);
        }
      };

      this.worker.onerror = (err) => {
        this.setStatus('error');
        this.onError?.(err);
      };

      this.worker.postMessage({
        gorisansonMoves: this.gorisansonMoves,
        timeMs,
        maxSimulations,
        uctConst: this.config.uctConst ?? 0.2,
      });
    };

    runSearch();
  }

  setStatus(status) {
    this.onStatus?.(status);
  }
}
