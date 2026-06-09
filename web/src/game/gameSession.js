import {
  QuoridorBoard,
  WallType,
  toAlgebraic,
  isWallAction,
  formatCoordinate,
} from '../lib/gameLogic.js';

import { PlayerType } from '../lib/engineConfig.js';

export class GameSession {
  constructor() {
    this.reset();
    this.listeners = new Set();
  }

  reset() {
    this.board = new QuoridorBoard();
    this.actions = [];
    this.wallsByPlayer = [];
    this.winner = null;
    this.lastAction = null;
    this.historyIndex = null;
    this.futureActions = [];
  }

  subscribe(listener) {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  notify() {
    for (const listener of this.listeners) {
      listener(this.getSnapshot());
    }
  }

  getSnapshot() {
    return {
      board: this.board,
      actions: [...this.actions],
      wallsByPlayer: [...this.wallsByPlayer],
      winner: this.winner,
      lastAction: this.lastAction,
      playerToMove: this.board.playerToMove(),
      playerPositions: this.board._playerPositions.map((coordinate) => ({ ...coordinate })),
      wallsRemaining: this.board._wallsRemaining.map((count) => count),
      validActions: this.board.validActions(),
      isTerminal: this.winner !== null,
      canRedo: this.futureActions.length > 0,
    };
  }

  getEngineSnapshot() {
    return {
      currentState: {
        playerToMove: this.board.playerToMove(),
        playerPositions: this.board._playerPositions.map((coordinate) => ({ ...coordinate })),
        wallsRemaining: this.board._wallsRemaining.map((count) => count),
        wallsByPlayer: [...this.wallsByPlayer],
      },
    };
  }

  canInteract(playerTypes, playerIndex) {
    if (this.winner) {
      return false;
    }
    return playerTypes[playerIndex] === PlayerType.Human;
  }

  isHumanTurn(playerTypes) {
    const playerIndex = this.board.playerToMove() - 1;
    return this.canInteract(playerTypes, playerIndex);
  }

  getCurrentPlayerType(playerTypes) {
    return playerTypes[this.board.playerToMove() - 1];
  }

  applyAction(action) {
    if (this.winner) {
      return false;
    }

    if (!this.board.isValid(action)) {
      return false;
    }

    const actingPlayer = this.board.playerToMove();

    this.board.takeAction(action);
    this.actions.push(structuredClone(action));
    this.lastAction = structuredClone(action);
    if (!this._skipClearFuture) {
      this.futureActions = [];
    }
    this._skipClearFuture = false;

    if (isWallAction(action)) {
      this.wallsByPlayer.push([
        actingPlayer,
        { ...action.coordinate },
        action.wallType,
      ]);
    }

    const terminal = this.board.terminal();
    if (terminal.isTerminal) {
      this.winner = terminal.playerNum;
    }

    this.notify();
    return true;
  }

  undo() {
    if (this.actions.length === 0) {
      return false;
    }

    const removed = this.actions[this.actions.length - 1];
    this.futureActions.push(structuredClone(removed));
    this.rebuildFromActions(this.actions.slice(0, -1));
    this.notify();
    return true;
  }

  redo() {
    if (this.futureActions.length === 0) {
      return false;
    }

    const action = this.futureActions.pop();
    this._skipClearFuture = true;
    const ok = this.applyAction(action);
    if (!ok) {
      this._skipClearFuture = false;
      this.futureActions.push(action);
    }
    return ok;
  }

  rebuildFromActions(actions) {
    this.board = new QuoridorBoard();
    this.actions = [];
    this.wallsByPlayer = [];
    this.winner = null;
    this.lastAction = null;
    this.futureActions = [];

    for (const action of actions) {
      const actingPlayer = this.board.playerToMove();
      this.board.takeAction(action);
      this.actions.push(structuredClone(action));
      this.lastAction = structuredClone(action);

      if (isWallAction(action)) {
        this.wallsByPlayer.push([
          actingPlayer,
          { ...action.coordinate },
          action.wallType,
        ]);
      }
    }

    const terminal = this.board.terminal();
    if (terminal.isTerminal) {
      this.winner = terminal.playerNum;
    }
  }

  getWallOwner(coordinate, wallType) {
    const key = `${formatCoordinate(coordinate)}${wallType === WallType.Horizontal ? 'h' : 'v'}`;
    for (const [playerNum, coord, type] of this.wallsByPlayer) {
      const entryKey = `${formatCoordinate(coord)}${type === WallType.Horizontal ? 'h' : 'v'}`;
      if (entryKey === key) {
        return playerNum;
      }
    }
    return 0;
  }

  actionToLabel(action) {
    return toAlgebraic(action);
  }
}
