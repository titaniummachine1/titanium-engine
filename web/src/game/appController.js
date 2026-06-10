import { GameSession } from './gameSession.js';
import { naiveDistanceEval } from '../lib/gameLogic.js';
import { decodeReplayCode, encodeReplayFromActions } from '../lib/replayCode.js';
import { fetchCatSnapshot, indexCatWalls } from '../lib/catHeatmap.js';
import { toAlgebraic } from '../lib/gameLogic.js';
import { EngineClient } from '../lib/engineClient.js';
import { GorisansonEngineClient, TitaniumEngineClient } from '../lib/localMctsEngine.js';
import { PlayerType, StrengthLevel, TimeToMove } from '../lib/engineConfig.js';
import {
  STRENGTH_LEVEL_PRESETS,
  TIME_TO_MOVE_PRESETS,
  getAllEngineConfigs,
  getPlayerOptionGroups,
  flattenPlayerOptions,
  describeTimeBudget,
  describeActiveSearchInfo,
} from '../lib/playerRegistry.js';

function mergeDepthLogs(existing, incoming) {
  const byDepth = new Map((existing ?? []).map((entry) => [entry.depth, entry]));
  for (const entry of incoming ?? []) {
    byDepth.set(entry.depth, entry);
  }
  return [...byDepth.values()].sort((a, b) => a.depth - b.depth);
}

function deepestDepthEntry(depthLog) {
  if (!depthLog?.length) {
    return null;
  }
  return depthLog.reduce((best, entry) => (entry.depth > (best?.depth ?? 0) ? entry : best));
}

function scoreFromDepthLog(depthLog, rootScore) {
  const deep = deepestDepthEntry(depthLog);
  if (deep?.score != null && Number.isFinite(Number(deep.score))) {
    return deep.score;
  }
  return rootScore ?? null;
}

function resolveThinkMs(info, thinkStartedAt) {
  if (info?.elapsedMs != null && Number.isFinite(Number(info.elapsedMs))) {
    return Math.round(Number(info.elapsedMs));
  }
  if (info?.time != null && Number.isFinite(Number(info.time))) {
    return Math.round(Number(info.time));
  }
  if (thinkStartedAt != null) {
    return Math.round(performance.now() - thinkStartedAt);
  }
  return null;
}

function buildThinkSeatSnapshot({
  engine,
  live = false,
  move = null,
  ply = null,
  depthLog,
  searchDepth,
  whiteDist,
  blackDist,
  rootScore,
  nodes,
  simulations,
  rootWinRate,
  stoppedBy,
  rootMoves,
  thinkMs,
}) {
  const deep = deepestDepthEntry(depthLog);
  return {
    live,
    engine,
    move,
    ply,
    whiteDist,
    blackDist,
    score: deep?.score ?? rootScore ?? null,
    depth: deep?.depth ?? searchDepth ?? null,
    pv: deep?.pv ?? '',
    nodes: nodes ?? simulations ?? 0,
    simulations: simulations ?? nodes ?? 0,
    rootWinRate,
    stoppedBy: stoppedBy ?? (live ? 'searching' : '?'),
    rootMoves: rootMoves ? [...rootMoves] : [],
    depthLog: depthLog ? [...depthLog] : [],
    thinkMs: thinkMs ?? null,
  };
}
import {
  WALL_CLOCK_RANGE,
  LOCAL_VISITS_RANGE,
  clampVisits,
  sliderPositionFromVisits,
  defaultPlayerAiSettings,
  describePlayerAiSettings,
  isLocalEngine,
  isLocalMctsEngine,
  isRemoteEngine,
  isTitaniumEngine,
} from '../lib/timeControl.js';
import { playerColorName } from '../lib/playerColors.js';
import { ponderCandidateSlots } from '../lib/enginePonder.js';

function isSavedSettingsValid(playerType, saved, engineConfigs) {
  if (isTitaniumEngine(playerType, engineConfigs)) {
    return (
      saved.strengthLevel != null &&
      saved.wallClockSeconds != null &&
      saved.visitsBudget != null
    );
  }
  if (isLocalEngine(playerType, engineConfigs)) {
    return saved.wallClockSeconds != null && saved.visitsBudget != null;
  }
  if (isRemoteEngine(playerType, engineConfigs)) {
    return saved.strengthLevel != null && saved.timeToMove != null;
  }
  return false;
}

export class AppController {
  constructor() {
    this.session = new GameSession();
    this.engines = new Map();
    this.engineConfigs = getAllEngineConfigs();

    this.settings = {
      players: [PlayerType.TitaniumMinimax, PlayerType.GorisansonMCTS],
      playerAiSettings: [
        defaultPlayerAiSettings(PlayerType.TitaniumMinimax, this.engineConfigs),
        defaultPlayerAiSettings(PlayerType.GorisansonMCTS, this.engineConfigs),
      ],
      playerAiSettingsMemory: [{}, {}],
      rotateBoard: false,
      displayCoordinates: true,
      displayRemainingWalls: true,
      displayEvalBar: true,
      showCatVision: false,
      uiMode: 'play',
    };

    this.replay = null;
    this.catViz = null;
    this.catVizLoading = false;
    this.catVizError = null;
    this._catFetchSeq = 0;
    this._catMovesKey = null;
    this.catHintDismissed = false;
    this.showCatHint = false;

    this.engineStatus = {};
    this.engineErrors = {};
    this.searchInfo = {};
    this.moveThinkLog = [];
    this.settingsChangelog = [];
    this.initialBudgetHint = null;
    this.lastThinkBySeat = [null, null];
    this.eval = { score: 0.5, p1: 0.5, pv: [] };
    this.aiThinking = false;
    this.liveSearch = null;
    this.thinkingPlayerType = null;
    this._moveRequestSeq = 0;
    this._thinkAiSettings = null;

    this.session.subscribe(() => this.onSessionChange());
    // First paint: Titanium vs Gorisanson AI should not always be White.
    this.maybeRandomizeTitaniumGorisansonSeats();
    this.initialBudgetHint = describeTimeBudget(
      this.settings.players,
      this.settings.playerAiSettings,
      this.engineConfigs,
    );
  }

  getState() {
    const snapshot = this.session.getSnapshot();
    const distanceEval = naiveDistanceEval(this.session.board);

    return {
      ...snapshot,
      settings: { ...this.settings },
      engineStatus: { ...this.engineStatus },
      engineErrors: { ...this.engineErrors },
      eval: {
        p1: distanceEval.p1,
        margin: distanceEval.margin,
        whiteDist: distanceEval.whiteDist,
        blackDist: distanceEval.blackDist,
        pv: this.eval.pv ?? [],
      },
      liveSearch: this.liveSearch,
      aiThinking: this.aiThinking,
      strengthLevelPresets: STRENGTH_LEVEL_PRESETS,
      timeToMovePresets: TIME_TO_MOVE_PRESETS,
      playerOptionGroups: getPlayerOptionGroups(),
      playerOptions: flattenPlayerOptions(getPlayerOptionGroups()),
      playerAiSettingsUi: this.getPlayerAiSettingsUi(),
      timeBudgetHint: describeTimeBudget(
        this.settings.players,
        this.settings.playerAiSettings,
        this.engineConfigs,
      ),
      thinkingPlayerType: this.thinkingPlayerType,
      searchInfoLine: describeActiveSearchInfo(
        this.settings.players,
        this.searchInfo,
        this.engineConfigs,
        {
          thinkingPlayerType: this.thinkingPlayerType,
          aiThinking: this.aiThinking,
        },
      ),
      activeSearchInfo: this.thinkingPlayerType
        ? this.searchInfo[this.thinkingPlayerType]
        : null,
      moveThinkLog: this.moveThinkLog,
      settingsChangelog: this.settingsChangelog,
      initialBudgetHint: this.initialBudgetHint,
      lastThinkBySeat: this.lastThinkBySeat,
      uiMode: this.settings.uiMode,
      catViz: this.catViz,
      catVizLoading: this.catVizLoading,
      catVizError: this.catVizError,
      showCatHint: this.showCatHint && this.settings.showCatVision,
      canRedo: snapshot.canRedo,
      replay: this.replay
        ? {
          index: this.replay.index,
          total: this.replay.actions.length,
          code: this.replay.code,
          meta: this.replay.meta,
        }
        : null,
      playReplayCode:
        this.session.actions.length > 0 && this.settings.uiMode === 'play'
          ? encodeReplayFromActions(
            this.session.actions,
            this.session.winner
              ? {
                winner: this.session.winner === 1 ? 'white' : 'black',
                plies: this.session.actions.length,
              }
              : null,
          )
          : null,
    };
  }

  onChange = null;
  onLiveUpdate = null;
  _liveUpdateLastMs = 0;

  setPlayer(playerNum, playerType) {
    if (playerType === PlayerType.Pavlosdais) {
      return;
    }
    this.settings.players[playerNum - 1] = playerType;
    this.ensurePlayerAiSettingsSlot(playerNum, playerType);

    if (playerType !== PlayerType.Human && playerType !== PlayerType.GorisansonMCTS) {
      this.syncRemoteEngine(playerType);
    }
    this.onChange?.();
    this.maybeRequestAiMove();
  }

  ensurePlayerAiSettingsSlot(playerNum, playerType) {
    const index = playerNum - 1;
    if (playerType === PlayerType.Human) {
      return;
    }

    const memory = this.settings.playerAiSettingsMemory[index] ?? {};
    let saved = memory[playerType];
    if (saved?.strength != null && saved.timeToMove == null) {
      saved = {
        strengthLevel: StrengthLevel.Alpha,
        timeToMove: saved.strength,
      };
      memory[playerType] = saved;
    }
    if (saved && isSavedSettingsValid(playerType, saved, this.engineConfigs)) {
      this.settings.playerAiSettings[index] = { ...saved };
      return;
    }

    const created = defaultPlayerAiSettings(playerType, this.engineConfigs);
    memory[playerType] = { ...created };
    this.settings.playerAiSettingsMemory[index] = memory;
    this.settings.playerAiSettings[index] = created;
  }

  rememberPlayerAiSettings(playerNum, aiSettings) {
    const index = playerNum - 1;
    const playerType = this.settings.players[index];
    if (playerType === PlayerType.Human || !aiSettings) {
      return;
    }
    const memory = this.settings.playerAiSettingsMemory[index] ?? {};
    memory[playerType] = { ...aiSettings };
    this.settings.playerAiSettingsMemory[index] = memory;
    this.settings.playerAiSettings[index] = { ...aiSettings };
  }

  recordSettingsChange(playerNum, field, from, to) {
    if (this.settings.uiMode !== 'play' || this.session.winner || from === to) {
      return;
    }
    const seat = playerColorName(playerNum);
    this.settingsChangelog.push({
      ply: this.session.actions.length,
      seat,
      player: this.engineLabel(this.settings.players[playerNum - 1]),
      field,
      from,
      to,
    });
  }

  getPlayerAiSettingsUiForSlot(playerNum) {
    const index = playerNum - 1;
    const playerType = this.settings.players[index];
    const current = this.settings.playerAiSettings[index];

    return {
      playerNum,
      playerType,
      isHuman: playerType === PlayerType.Human,
      isLocal: isLocalEngine(playerType, this.engineConfigs),
      isTitanium: isTitaniumEngine(playerType, this.engineConfigs),
      isLocalMcts: isLocalMctsEngine(playerType, this.engineConfigs),
      isRemote: isRemoteEngine(playerType, this.engineConfigs),
      strengthLevel: current?.strengthLevel ?? StrengthLevel.Alpha,
      timeToMove: current?.timeToMove ?? TimeToMove.Short,
      wallClockSeconds: current?.wallClockSeconds ?? WALL_CLOCK_RANGE.defaultSeconds,
      visitsBudget: clampVisits(current?.visitsBudget ?? LOCAL_VISITS_RANGE.default),
      visitsSliderPosition: sliderPositionFromVisits(
        current?.visitsBudget ?? LOCAL_VISITS_RANGE.default,
      ),
      wallclockRange: WALL_CLOCK_RANGE,
      visitsRange: {
        min: 0,
        max: LOCAL_VISITS_RANGE.sliderSteps,
        step: 1,
      },
      hint: describePlayerAiSettings(playerType, current, this.engineConfigs),
      engineName: this.engineLabel(playerType),
    };
  }

  getPlayerAiSettingsUi() {
    return [this.getPlayerAiSettingsUiForSlot(1), this.getPlayerAiSettingsUiForSlot(2)];
  }

  setPlayerStrengthLevel(playerNum, strengthLevel, { silent = false } = {}) {
    const index = playerNum - 1;
    const playerType = this.settings.players[index];
    if (!isRemoteEngine(playerType, this.engineConfigs) && !isTitaniumEngine(playerType, this.engineConfigs)) {
      return;
    }
    const current = this.settings.playerAiSettings[index] ?? {};
    const next = Number(strengthLevel);
    this.recordSettingsChange(playerNum, 'strength', current.strengthLevel, next);
    this.rememberPlayerAiSettings(playerNum, {
      ...current,
      strengthLevel: next,
    });
    if (!silent) {
      this.onChange?.();
    }
  }

  setPlayerTimeToMove(playerNum, timeToMove, { silent = false } = {}) {
    const index = playerNum - 1;
    const playerType = this.settings.players[index];
    if (!isRemoteEngine(playerType, this.engineConfigs)) {
      return;
    }
    const current = this.settings.playerAiSettings[index] ?? {};
    const next = Number(timeToMove);
    this.recordSettingsChange(playerNum, 'timeToMove', current.timeToMove, next);
    this.rememberPlayerAiSettings(playerNum, {
      ...current,
      timeToMove: next,
    });
    if (!silent) {
      this.onChange?.();
    }
  }

  setPlayerWallClock(playerNum, seconds, { silent = false } = {}) {
    const index = playerNum - 1;
    const playerType = this.settings.players[index];
    if (!isLocalMctsEngine(playerType, this.engineConfigs)) {
      return;
    }
    const current = this.settings.playerAiSettings[index] ?? {};
    const next = Number(seconds);
    this.recordSettingsChange(playerNum, 'wallClockSeconds', current.wallClockSeconds, next);
    this.rememberPlayerAiSettings(playerNum, {
      ...current,
      wallClockSeconds: next,
    });
    if (!silent) {
      this.onChange?.();
    }
  }

  setPlayerVisitsBudget(playerNum, visits, { silent = false } = {}) {
    const index = playerNum - 1;
    const playerType = this.settings.players[index];
    if (!isLocalMctsEngine(playerType, this.engineConfigs)) {
      return;
    }
    const current = this.settings.playerAiSettings[index] ?? {};
    const next = clampVisits(visits);
    this.recordSettingsChange(playerNum, 'visitsBudget', current.visitsBudget, next);
    this.rememberPlayerAiSettings(playerNum, {
      ...current,
      visitsBudget: next,
    });
    if (!silent) {
      this.onChange?.();
    }
  }

  toggleRotateBoard() {
    this.settings.rotateBoard = !this.settings.rotateBoard;
    this.onChange?.();
  }

  toggleDisplayCoordinates() {
    this.settings.displayCoordinates = !this.settings.displayCoordinates;
    this.onChange?.();
  }

  toggleDisplayRemainingWalls() {
    this.settings.displayRemainingWalls = !this.settings.displayRemainingWalls;
    this.onChange?.();
  }

  toggleDisplayEvalBar() {
    this.settings.displayEvalBar = !this.settings.displayEvalBar;
    this.onChange?.();
  }

  toggleCatVision(enabled = !this.settings.showCatVision) {
    this.settings.showCatVision = Boolean(enabled);
    if (this.settings.showCatVision) {
      if (!this.catHintDismissed) {
        this.showCatHint = true;
      }
      this.invalidateCatCache();
      this.refreshCatViz();
    } else {
      this._catFetchSeq += 1;
      this._catMovesKey = null;
      this.catViz = null;
      this.catVizError = null;
      this.catVizLoading = false;
      this.showCatHint = false;
    }
    this.onChange?.();
  }

  dismissCatHint() {
    this.catHintDismissed = true;
    this.showCatHint = false;
    this.onChange?.();
  }

  catMovesKey() {
    return this.session.actions.map((action) => toAlgebraic(action)).join('|');
  }

  invalidateCatCache() {
    this._catMovesKey = null;
  }

  scheduleCatRefresh() {
    if (!this.settings.showCatVision || this.settings.uiMode === 'replay') {
      return;
    }
    const key = this.catMovesKey();
    if (key === this._catMovesKey && this.catViz && !this.catVizError) {
      return;
    }
    this.refreshCatViz();
  }

  /** Titanium vs Gorisanson AI-vs-AI: 50/50 White/Black on load and each new game. */
  maybeRandomizeTitaniumGorisansonSeats() {
    const { players, playerAiSettings, playerAiSettingsMemory } = this.settings;
    if (players.includes(PlayerType.Human)) {
      return;
    }
    const titaniumSeat = (i) => isTitaniumEngine(players[i], this.engineConfigs);
    const gorisansonSeat = (i) => players[i] === PlayerType.GorisansonMCTS;
    const isTiGo =
      (titaniumSeat(0) && gorisansonSeat(1)) || (gorisansonSeat(0) && titaniumSeat(1));
    if (!isTiGo) {
      return;
    }
    // Pick Titanium on White or Black; Gorisanson takes the other seat.
    if (Math.random() >= 0.5) {
      return;
    }
    this.settings.players = [players[1], players[0]];
    this.settings.playerAiSettings = [playerAiSettings[1], playerAiSettings[0]];
    this.settings.playerAiSettingsMemory = [
      playerAiSettingsMemory[1],
      playerAiSettingsMemory[0],
    ];
  }

  newGame() {
    this.maybeRandomizeTitaniumGorisansonSeats();
    this._moveRequestSeq += 1;
    this.aiThinking = false;
    this.liveSearch = null;
    this.engineErrors = {};
    this.replay = null;
    this.moveThinkLog = [];
    this.settingsChangelog = [];
    this.initialBudgetHint = describeTimeBudget(
      this.settings.players,
      this.settings.playerAiSettings,
      this.engineConfigs,
    );
    this.lastThinkBySeat = [null, null];
    this.catHintDismissed = false;
    this.showCatHint = false;
    this.settings.uiMode = 'play';
    this.eval = { score: 0.5, p1: 0.5, pv: [] };
    for (const engine of this.engines.values()) {
      engine.resetConnection();
    }
    this.session.reset();
    this.invalidateCatCache();
    this.scheduleCatRefresh();
    this.onChange?.();
    this.maybeRequestAiMove();
  }

  isFreePlayMode() {
    return this.settings.uiMode === 'analysis';
  }

  setUiMode(mode) {
    this.settings.uiMode = mode;
    if (mode === 'analysis') {
      this._moveRequestSeq += 1;
      this.replay = null;
      this.aiThinking = false;
      this.thinkingPlayerType = null;
      this.liveSearch = null;
    }
    this.scheduleCatRefresh();
    this.onChange?.();
  }

  loadAnalysisPosition(code) {
    this._moveRequestSeq += 1;
    const trimmed = code.trim();
    const { actions } = decodeReplayCode(trimmed);
    this.replay = null;
    this.aiThinking = false;
    this.liveSearch = null;
    this.session.rebuildFromActions(actions);
    this.invalidateCatCache();
    this.scheduleCatRefresh();
    this.onChange?.();
  }

  async refreshCatViz() {
    if (!this.settings.showCatVision) {
      return;
    }
    const movesKey = this.catMovesKey();
    const seq = ++this._catFetchSeq;
    this.catVizLoading = true;
    this.catVizError = null;
    this.onChange?.();

    const moves = this.session.actions.map((action) => toAlgebraic(action));
    try {
      const data = await fetchCatSnapshot(moves);
      if (seq !== this._catFetchSeq) {
        return;
      }
      const squares = data.squares ?? [];
      const reachableRaw = data.reachable ?? [];
      const reachable =
        reachableRaw.length === 81
          ? reachableRaw.map((v) => v === 1 || v === true || v === '1')
          : null;
      const walls = data.walls ?? [];

      this.catViz = {
        squares,
        reachable,
        wallIndex: indexCatWalls(walls),
        whiteDist: data.whiteDist,
        blackDist: data.blackDist,
        hotCm: data.hotCm ?? 160,
        coldCm: data.coldCm ?? 60,
        maxCm: data.maxCm ?? 400,
        skippedSquares:
          data.skippedSquares ?? reachable?.filter((r) => !r).length ?? 0,
        skippedWallCount: walls.filter((w) => w.skip ?? w.pruned).length,
        searchableWallCount: walls.filter((w) => w.search ?? !(w.skip ?? w.pruned)).length,
      };
      this._catMovesKey = movesKey;
      this.catVizError = null;
    } catch (err) {
      if (seq !== this._catFetchSeq) {
        return;
      }
      this.catViz = null;
      this.catVizError = err.message ?? String(err);
    } finally {
      if (seq === this._catFetchSeq) {
        this.catVizLoading = false;
        this.onChange?.();
      }
    }
  }

  /** Leave replay scrubber but keep the current position — human can play from here. */
  continueFromReplay() {
    if (!this.replay) {
      return;
    }
    this._moveRequestSeq += 1;
    this.replay = null;
    this.settings.uiMode = 'play';
    this.aiThinking = false;
    this.thinkingPlayerType = null;
    this.liveSearch = null;
    this.moveThinkLog = [];
    this.onChange?.();
    this.maybeRequestAiMove();
  }

  loadReplay(code) {
    this._moveRequestSeq += 1;
    const trimmed = code.trim();
    const { actions, meta, algebraic } = decodeReplayCode(trimmed);
    this.replay = {
      actions,
      algebraic,
      index: actions.length,
      code: trimmed.startsWith('tq1') ? trimmed : encodeReplayFromActions(actions, meta),
      meta,
    };
    this.settings.uiMode = 'replay';
    this.aiThinking = false;
    this.liveSearch = null;
    this.engineErrors = {};
    for (const engine of this.engines.values()) {
      engine.resetConnection();
    }
    this.applyReplayIndex();
    this.onChange?.();
  }

  applyReplayIndex() {
    if (!this.replay) {
      return;
    }
    const slice = this.replay.actions.slice(0, this.replay.index);
    this.session.rebuildFromActions(slice);
  }

  setReplayIndex(index) {
    if (!this.replay) {
      return;
    }
    this.replay.index = Math.max(0, Math.min(index, this.replay.actions.length));
    this.applyReplayIndex();
    this.onChange?.();
  }

  replayStep(delta) {
    if (!this.replay) {
      return;
    }
    this.setReplayIndex(this.replay.index + delta);
  }

  exportReplayCode() {
    if (!this.replay) {
      return encodeReplayFromActions(this.session.actions);
    }
    return this.replay.code;
  }

  undo() {
    if (this.aiThinking || this.replay) {
      return;
    }
    this._moveRequestSeq += 1;
    this.liveSearch = null;
    this.engineErrors = {};
    if (!this.session.undo()) {
      return;
    }
    for (const engine of this.engines.values()) {
      engine.resetConnection();
    }
    this.onChange?.();
    if (!this.isFreePlayMode()) {
      this.maybeRequestAiMove();
    }
  }

  redo() {
    if (this.aiThinking || this.replay) {
      return;
    }
    this._moveRequestSeq += 1;
    this.liveSearch = null;
    if (!this.session.redo()) {
      return;
    }
    for (const engine of this.engines.values()) {
      engine.resetConnection();
    }
    this.onChange?.();
    if (!this.isFreePlayMode()) {
      this.maybeRequestAiMove();
    }
  }

  tryAction(action) {
    if (this.replay || this.aiThinking) {
      return;
    }

    const freePlay = this.isFreePlayMode();
    if (!freePlay && !this.session.isHumanTurn(this.settings.players)) {
      return;
    }

    const applied = this.session.applyAction(action);
    if (!applied) {
      return;
    }

    if (!freePlay) {
      this.syncRemoteEnginesAfterMove(action);
    }
    this.onChange?.();
    if (freePlay) {
      return;
    }
    this.maybeRequestAiMove();
    this.maybePonderInactiveEngines();
  }

  onSessionChange() {
    this.scheduleCatRefresh();
    this.onChange?.();
  }

  createEngineClient(config) {
    if (config.kind === 'local') {
      return new GorisansonEngineClient(config);
    }
    if (config.kind === 'titanium') {
      return new TitaniumEngineClient(config);
    }
    return new EngineClient(config);
  }

  getEngine(playerType) {
    if (playerType === PlayerType.Human) {
      return null;
    }

    if (!this.engines.has(playerType)) {
      const config = this.engineConfigs.find((entry) => entry.key === playerType);
      if (!config || config.disabled) {
        return null;
      }

      const engine = this.createEngineClient(config);
      engine.onStatus = (status) => {
        const prev = this.engineStatus[playerType];
        this.engineStatus[playerType] = status;
        if (prev !== status) {
          this.onChange?.();
        }
      };
      engine.onInfo = (info) => {
        const prev = this.searchInfo[playerType] ?? {};
        const depthLog = info.thinking
          ? info.depthLog?.length
            ? mergeDepthLogs(prev.depthLog, info.depthLog)
            : prev.depthLog
          : info.depthLog ?? [];
        this.searchInfo[playerType] = {
          ...prev,
          ...info,
          depthLog,
        };
        if (info.thinking) {
          const liveDepthLog = depthLog?.length
            ? depthLog
            : mergeDepthLogs(this.liveSearch?.depthLog, info.depthLog);
          const liveRootScore = scoreFromDepthLog(
            liveDepthLog,
            info.rootScore ?? this.liveSearch?.rootScore,
          );
          this.liveSearch = {
            playerType,
            playerLabel: this.engineLabel(playerType),
            simulations: info.simulations ?? this.liveSearch?.simulations,
            nodes: info.nodes ?? this.liveSearch?.nodes,
            progress: info.progress,
            mode:
              info.mode ??
              info.stoppedBy ??
              (isTitaniumEngine(playerType, this.engineConfigs) ? 'minimax' : 'mcts'),
            searchDepth: info.searchDepth ?? this.liveSearch?.searchDepth,
            depthLog: liveDepthLog,
            rootWinRate:
              info.rootWinRate != null ? info.rootWinRate : this.liveSearch?.rootWinRate,
            whiteDist: info.whiteDist ?? this.liveSearch?.whiteDist,
            blackDist: info.blackDist ?? this.liveSearch?.blackDist,
            rootMoves: info.rootMoves ?? this.liveSearch?.rootMoves,
            rootScore: liveRootScore,
          };
          const seat = this.seatIndexForPlayerType(playerType);
          this.snapshotThinkSeat(seat, {
            live: true,
            ply: this.session.actions.length + 1,
            depthLog: liveDepthLog,
            searchDepth: this.liveSearch.searchDepth,
            whiteDist: this.liveSearch.whiteDist,
            blackDist: this.liveSearch.blackDist,
            rootScore: this.liveSearch.rootScore,
            nodes: this.liveSearch.nodes,
            simulations: this.liveSearch.simulations,
            rootWinRate: this.liveSearch.rootWinRate,
            stoppedBy: this.liveSearch.mode,
            rootMoves: this.liveSearch.rootMoves,
            engine: this.liveSearch.playerLabel,
          });
          const now = performance.now();
          if (now - this._liveUpdateLastMs >= 16) {
            this._liveUpdateLastMs = now;
            (this.onLiveUpdate ?? this.onChange)?.();
          }
          return;
        }
        if (info.progress !== undefined && info.p1 === undefined && !info.pv && !info.stoppedBy) {
          return;
        }
        if (info.pv) {
          this.eval.pv = info.pv;
        }
        if (info.stoppedBy) {
          const si = this.searchInfo[playerType] ?? {};
          const seat = this.seatIndexForPlayerType(playerType);
          this.snapshotThinkSeat(seat, {
            live: false,
            ply: this.session.actions.length + 1,
            depthLog: si.depthLog,
            searchDepth: si.searchDepth,
            whiteDist: si.whiteDist,
            blackDist: si.blackDist,
            rootScore: si.rootScore,
            nodes: si.nodes,
            simulations: si.simulations,
            rootWinRate: si.rootWinRate,
            stoppedBy: info.stoppedBy,
            rootMoves: si.rootMoves,
            engine: this.engineLabel(playerType),
          });
        }
        this.onChange?.();
      };
      engine.onError = (err) => {
        this.aiThinking = false;
        const message = err?.message ?? String(err ?? 'Engine error');
        this.engineErrors[playerType] = message;
        this.engineStatus[playerType] = 'error';
        this.onChange?.();
      };
      this.engines.set(playerType, engine);
    }

    return this.engines.get(playerType);
  }

  /** Keep remote engines in sync after every ply (human or AI), matching scraped takeAction middleware. */
  syncRemoteEnginesAfterMove(action) {
    for (const playerType of this.settings.players) {
      if (
        playerType === PlayerType.Human ||
        isLocalEngine(playerType, this.engineConfigs) ||
        isTitaniumEngine(playerType, this.engineConfigs)
      ) {
        continue;
      }
      const engine = this.getEngine(playerType);
      engine?.makeMoves([action]);
    }
  }

  syncRemoteEngine(playerType) {
    const engine = this.getEngine(playerType);
    if (!engine?.syncGameState) {
      return;
    }

    const moveHistory = this.session.actions;
    engine.syncGameState({
      moveHistory,
      gameSnapshot: this.session.getEngineSnapshot(),
      isFreshGame: moveHistory.length === 0,
    });
  }

  /** Stop background ponder on all engines before a real search. Safe no-op until pondering ships. */
  stopAllPonders() {
    for (const engine of this.engines.values()) {
      engine.stopPonder?.();
    }
  }

  /**
   * Future: remote `go ponder` + local predicted-line MCTS (node cap only).
   * @see docs/video/09-pondering-prep.md
   */
  maybePonderInactiveEngines() {
    if (this.session.winner || this.aiThinking) {
      return;
    }
    const { playerToMove } = this.session.getSnapshot();
    for (const { playerType } of ponderCandidateSlots(this.settings.players, playerToMove)) {
      const engine = this.getEngine(playerType);
      if (!engine?.ponder) {
        continue;
      }
      // Not enabled yet — wire aiSettings + sync before calling engine.ponder(...)
    }
  }

  engineLabel(playerType) {
    const config = this.engineConfigs.find((entry) => entry.key === playerType);
    return config?.name ?? playerType;
  }

  seatIndexForPlayerType(playerType) {
    return this.settings.players.indexOf(playerType);
  }

  recordEngineFailure(playerType, { ply, error, budget }) {
    const message = error?.message ?? String(error ?? 'Engine error');
    console.error('Engine search failed', {
      playerType,
      ply,
      engine: this.engineLabel(playerType),
      message,
      stack: error?.stack,
    });

    this.engineErrors[playerType] = message;
    this.engineStatus[playerType] = 'error';
    this.aiThinking = false;
    this.thinkingPlayerType = null;
    this.liveSearch = null;
    this._thinkStartedAt = null;

    const si = this.searchInfo[playerType] ?? {};
    const thinkMs = resolveThinkMs(si, null);
    const seat = this.seatIndexForPlayerType(playerType);
    this.snapshotThinkSeat(seat, {
      live: false,
      ply,
      move: null,
      error: message,
      stoppedBy: 'error',
      engine: this.engineLabel(playerType),
      depthLog: si.depthLog,
      searchDepth: si.searchDepth,
      whiteDist: si.whiteDist,
      blackDist: si.blackDist,
      nodes: si.nodes ?? si.simulations,
      simulations: si.simulations ?? si.nodes,
      thinkMs,
    });
    this.moveThinkLog.push({
      ply,
      move: null,
      engine: this.engineLabel(playerType),
      budget: budget ?? '',
      error: message,
      stoppedBy: 'error',
      nodes: si.nodes ?? si.simulations ?? 0,
      searchDepth: si.searchDepth,
      whiteDist: si.whiteDist,
      blackDist: si.blackDist,
      depthLog: si.depthLog ? [...si.depthLog] : [],
      thinkMs,
    });
  }

  snapshotThinkSeat(seatIndex, fields) {
    if (seatIndex < 0) {
      return;
    }
    this.lastThinkBySeat[seatIndex] = buildThinkSeatSnapshot({
      engine: fields.engine ?? this.engineLabel(this.settings.players[seatIndex]),
      ...fields,
    });
  }

  maybeRequestAiMove() {
    if (this.replay || this.isFreePlayMode()) {
      this.aiThinking = false;
      return;
    }
    if (this.session.winner) {
      this.aiThinking = false;
      this.liveSearch = null;
      return;
    }

    this.stopAllPonders();

    const playerType = this.session.getCurrentPlayerType(this.settings.players);
    if (playerType === PlayerType.Human) {
      this.aiThinking = false;
      return;
    }

    const engine = this.getEngine(playerType);
    if (!engine) {
      this.aiThinking = false;
      return;
    }

    const requestSnapshot = this.session.getSnapshot();
    const requestSeq = ++this._moveRequestSeq;
    const requestPly = requestSnapshot.actions.length;
    const requestPlayerToMove = requestSnapshot.playerToMove;

    this.aiThinking = true;
    this.thinkingPlayerType = playerType;
    this._thinkStartedAt = performance.now();
    this.searchInfo[playerType] = { depthLog: [] };
    const seat = this.seatIndexForPlayerType(playerType);
    const prevThink = seat >= 0 ? this.lastThinkBySeat[seat] : null;
    this.snapshotThinkSeat(seat, {
      live: true,
      ply: requestPly + 1,
      depthLog: [],
      stoppedBy: 'searching',
      engine: this.engineLabel(playerType),
      move: null,
      ...(prevThink && !prevThink.live
        ? {
          whiteDist: prevThink.whiteDist,
          blackDist: prevThink.blackDist,
          score: prevThink.score,
          depth: prevThink.depth,
          pv: prevThink.pv,
          rootWinRate: prevThink.rootWinRate,
          rootMoves: prevThink.rootMoves,
        }
        : {}),
    });
    this.liveSearch = {
      playerType,
      playerLabel: this.engineLabel(playerType),
      mode: 'searching',
      depthLog: [],
    };
    this.onChange?.();

    engine.onBestMove = (action) => {
      const current = this.session.getSnapshot();
      const currentPlayerType = this.session.getCurrentPlayerType(this.settings.players);
      const stale =
        requestSeq !== this._moveRequestSeq ||
        current.actions.length !== requestPly ||
        current.playerToMove !== requestPlayerToMove ||
        currentPlayerType !== playerType;
      if (stale) {
        console.warn('Ignoring stale engine move response', {
          playerType,
          requestSeq,
          currentSeq: this._moveRequestSeq,
          requestPly,
          currentPly: current.actions.length,
          requestPlayerToMove,
          currentPlayerToMove: current.playerToMove,
          currentPlayerType,
          suggested: this.session.actionToLabel(action),
        });
        this.aiThinking = false;
        this.thinkingPlayerType = null;
        this.onChange?.();
        return;
      }

      this.aiThinking = false;
      this.thinkingPlayerType = null;
      this.liveSearch = null;

      const applied = this.session.applyAction(action);
      if (applied) {
        const plyNum = this.session.actions.length;
        const si = this.searchInfo[playerType] ?? {};
        const thinkMs = resolveThinkMs(si, this._thinkStartedAt);
        this._thinkStartedAt = null;
        const moveLabel = this.session.actionToLabel(action);
        const seat = this.seatIndexForPlayerType(playerType);
        const budgetHint = describePlayerAiSettings(
          playerType,
          this._thinkAiSettings ?? this.settings.playerAiSettings[seat],
          this.engineConfigs,
        );
        this._thinkAiSettings = null;
        this.snapshotThinkSeat(seat, {
          live: false,
          move: moveLabel,
          ply: plyNum,
          depthLog: si.depthLog,
          searchDepth: si.searchDepth,
          whiteDist: si.whiteDist,
          blackDist: si.blackDist,
          rootScore: si.rootScore,
          nodes: si.nodes,
          simulations: si.simulations,
          rootWinRate: si.rootWinRate,
          stoppedBy: si.stoppedBy ?? si.mode ?? '?',
          rootMoves: si.rootMoves,
          engine: this.engineLabel(playerType),
          thinkMs,
        });
        this.moveThinkLog.push({
          ply: plyNum,
          move: moveLabel,
          engine: this.engineLabel(playerType),
          budget: budgetHint,
          stoppedBy: si.stoppedBy ?? si.mode ?? '?',
          nodes: si.nodes ?? si.simulations ?? 0,
          searchDepth: si.searchDepth,
          whiteDist: si.whiteDist,
          blackDist: si.blackDist,
          rootScore: scoreFromDepthLog(si.depthLog, si.rootScore),
          rootWinRate: si.rootWinRate,
          depthLog: si.depthLog ? [...si.depthLog] : [],
          rootMoves: si.rootMoves ? [...si.rootMoves] : [],
          thinkMs,
        });
      }
      if (this.session.winner) {
        this.onChange?.();
        return;
      }
      if (!applied) {
        const snapshot = this.session.getSnapshot();
        const suggested = this.session.actionToLabel(action);
        const legal = snapshot.validActions.map((mv) => this.session.actionToLabel(mv));
        const seat = this.seatIndexForPlayerType(playerType);
        const si = this.searchInfo[playerType] ?? {};
        console.error('Engine produced illegal move', {
          playerType,
          suggested,
          ply: snapshot.actions.length + 1,
          playerToMove: snapshot.playerToMove,
          playerPositions: snapshot.playerPositions,
          wallsRemaining: snapshot.wallsRemaining,
          legalCount: legal.length,
          legalSample: legal.slice(0, 60),
        });
        this.searchInfo[playerType] = {
          ...(this.searchInfo[playerType] ?? {}),
          illegalMove: suggested,
          illegalMovePly: snapshot.actions.length + 1,
          legalMovesCount: legal.length,
        };
        const illegalMsg = `Illegal move ${suggested} on ply ${snapshot.actions.length + 1} (${legal.length} legal)`;
        this.engineErrors[playerType] = illegalMsg;
        this.engineStatus[playerType] = 'error';
        this.moveThinkLog.push({
          ply: snapshot.actions.length + 1,
          move: suggested,
          engine: this.engineLabel(playerType),
          budget: describePlayerAiSettings(
            playerType,
            this._thinkAiSettings ?? this.settings.playerAiSettings[seat],
            this.engineConfigs,
          ),
          error: illegalMsg,
          stoppedBy: 'error',
          thinkMs: resolveThinkMs(si, this._thinkStartedAt),
          nodes: si.nodes ?? si.simulations ?? 0,
          depthLog: si.depthLog ? [...si.depthLog] : [],
        });
        this.onChange?.();
        return;
      }

      this.engineErrors[playerType] = null;

      this.syncRemoteEnginesAfterMove(action);
      this.onChange?.();
      this.maybeRequestAiMove();
      this.maybePonderInactiveEngines();
    };

    engine.onError = (err) => {
      if (requestSeq !== this._moveRequestSeq) {
        return;
      }
      const seat = this.seatIndexForPlayerType(playerType);
      this.recordEngineFailure(playerType, {
        ply: requestPly + 1,
        error: err,
        budget: describePlayerAiSettings(
          playerType,
          this._thinkAiSettings ?? this.settings.playerAiSettings[seat],
          this.engineConfigs,
        ),
      });
      this.onChange?.();
    };

    const playerIndex = requestPlayerToMove - 1;
    let aiSettings = this.settings.playerAiSettings[playerIndex];
    if (!aiSettings) {
      aiSettings = defaultPlayerAiSettings(playerType, this.engineConfigs);
      this.settings.playerAiSettings[playerIndex] = aiSettings;
    }
    const moveHistory = this.session.actions;
    const isFreshGame = moveHistory.length === 0;
    this._thinkAiSettings = { ...aiSettings };

    engine.requestMove({
      aiSettings,
      gameSnapshot: this.session.getEngineSnapshot(),
      moveHistory,
      isFreshGame,
    });
  }
}
