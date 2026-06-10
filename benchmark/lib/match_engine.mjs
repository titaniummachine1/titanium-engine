/**
 * Rust Titanium vs local opponent (gorisanson MCTS or rust-titanium self-play).
 */

import { parseAlgebraic, toAlgebraic } from '../../web/src/lib/gameLogic.js';
import { actionToGorisansonMove, gorisansonMoveToAction } from './gorisanson_bridge.mjs';
import {
  applyGorisansonMove,
  chooseGorisansonMoveWithMeta,
  createGorisansonGame,
  winnerIndex,
} from './gorisanson_ai.mjs';
import { chooseTitaniumMove } from './titanium_ai.mjs';
import { chooseQuoridorV3Move } from './quoridor_v3_ai.mjs';
import { RUST_TITANIUM_ID, GORISANSON_ID, QUORIDOR_V3_ID, assertRustTitaniumId } from './engine_ids.mjs';
import { encodeReplayFromAlgebraic, formatReplayBlock } from './replay_code.mjs';
import { termLine, termThinking } from './terminal_log.mjs';
import { printPlyCompact, printFinalPosition, printSearchDepth } from './terminal_reporter.mjs';
import { GORISANSON_MAX_VISITS, resolveThinkBudget, TITANIUM_MAX_NODES } from './bench_limits.mjs';
import { validateMove, fallbackLegalMove } from './move_validate.mjs';
import { evalPosition } from './path_eval.mjs';
import { buildGameReport } from './game_report.mjs';
import { moveLabel } from './gorisanson_moves.mjs';

const MAX_PLIES = 250;

export function defaultPlayerConfigs({ timeSec = 10, gorisansonTimeSec = 10 } = {}) {
  return {
    titanium: {
      id: RUST_TITANIUM_ID,
      engine: 'minimax',
      timeSec,
      maxSimulations: Number(process.env.TITANIUM_MAX_NODES ?? TITANIUM_MAX_NODES),
      useCatGuidance: true,
    },
    gorisanson: {
      id: GORISANSON_ID,
      timeSec: gorisansonTimeSec,
      maxSimulations: Number(process.env.GORISANSON_MAX_VISITS ?? GORISANSON_MAX_VISITS),
    },
    quoridorV3: {
      id: QUORIDOR_V3_ID,
      timeSec: 0.5,
      maxDepth: 24,
    },
  };
}

function engineLabel(cfg, budget) {
  if (cfg.id === GORISANSON_ID) {
    return `Gorisanson MCTS (${budget.timeSec}s/${formatSimsCap(budget.maxSimulations)})`;
  }
  if (cfg.id === RUST_TITANIUM_ID) {
    const mode = cfg.engine === 'minimax' ? 'Minimax' : 'MCTS';
    return `Rust Titanium ${mode} (${budget.timeSec}s/${formatSimsCap(budget.maxSimulations)})`;
  }
  if (cfg.id === QUORIDOR_V3_ID) {
    return `Quoridor v3 αβ (${budget.timeMs}ms/d${cfg.maxDepth ?? 24})`;
  }
  return cfg.id;
}

function formatSimsCap(n) {
  if (n >= 1_000_000_000) {
    return `${(n / 1_000_000_000).toFixed(0)}B cap`;
  }
  if (n >= 1_000_000) {
    return `${(n / 1_000_000).toFixed(1)}M cap`;
  }
  return `${n} cap`;
}

function formatSims(n) {
  if (n >= 1_000_000) {
    return `${(n / 1_000_000).toFixed(1)}M`;
  }
  if (n >= 1000) {
    return `${(n / 1000).toFixed(1)}k`;
  }
  return String(n);
}

async function chooseMove(game, algebraicHistory, playerConfig, ply, options) {
  const logMoves = options.logMoves !== false && !options.quiet;
  const budget = resolveThinkBudget(options, playerConfig);
  const label = engineLabel(playerConfig, budget);

  if (logMoves) {
    termThinking({ ply, side: game.pawnOfTurn.index, engine: label });
  }

  if (playerConfig.id === GORISANSON_ID) {
    let lastProgressMs = -1;
    const { move, meta } = chooseGorisansonMoveWithMeta(game, {
      timeMs: budget.timeMs,
      maxSimulations: budget.maxSimulations,
      uct: playerConfig.uct,
      onProgress: logMoves
        ? (progress) => {
          const elapsedMs = progress.elapsedMs ?? 0;
          if (lastProgressMs >= 0 && elapsedMs - lastProgressMs < 900) {
            return;
          }
          lastProgressMs = elapsedMs;
          termLine(
            `      ply ${ply} progress ${playerConfig.id}: ${formatSims(progress.simulations ?? 0)} sims · ${(elapsedMs / 1000).toFixed(1)}s`,
          );
        }
        : undefined,
    });
    return { move, meta, elapsedMs: meta.elapsedMs };
  }

  if (playerConfig.id === RUST_TITANIUM_ID) {
    assertRustTitaniumId(playerConfig.id);
    const log = options.logSearch !== false;
    const started = performance.now();
    let lastProgressMs = -1;
    const engineMode = playerConfig.engine ?? options.engine;
    const { move: algebraic, meta } = await chooseTitaniumMove(algebraicHistory, {
      log,
      ply,
      engine: engineMode,
      timeSec: budget.timeSec,
      maxSims: budget.maxSimulations,
      uct: playerConfig.uct,
      disableBook: playerConfig.disableBook ?? options.disableBook,
      disableBridge: playerConfig.disableBridge ?? options.disableBridge,
      useCatGuidance: playerConfig.useCatGuidance ?? options.useCatGuidance,
      onDepth:
        logMoves && engineMode === 'minimax'
          ? (depth) => {
            printSearchDepth({ ply, ...depth });
          }
          : undefined,
      onProgress: logMoves
        ? (progress) => {
          const elapsedMs = progress.elapsedMs ?? 0;
          if (engineMode === 'minimax') {
            return;
          }
          if (lastProgressMs >= 0 && elapsedMs - lastProgressMs < 900) {
            return;
          }
          lastProgressMs = elapsedMs;
          termLine(
            `      ply ${ply} progress ${playerConfig.id}: ${formatSims(progress.simulations ?? 0)} sims · ${(elapsedMs / 1000).toFixed(1)}s`,
          );
        }
        : undefined,
    });
    const elapsedMs = performance.now() - started;
    return {
      move: actionToGorisansonMove(parseAlgebraic(algebraic)),
      meta,
      elapsedMs,
    };
  }

  if (playerConfig.id === QUORIDOR_V3_ID) {
    const started = performance.now();
    const { move: algebraic, meta } = chooseQuoridorV3Move(algebraicHistory, {
      timeMs: budget.timeMs,
      maxDepth: playerConfig.maxDepth ?? 24,
    });
    const elapsedMs = performance.now() - started;
    if (logMoves && meta.searchDepth != null) {
      termLine(
        `      ply ${ply} v3 depth ${meta.searchDepth} nodes ${formatSims(meta.nodes ?? 0)} · ${(elapsedMs / 1000).toFixed(2)}s`,
      );
    }
    return {
      move: actionToGorisansonMove(parseAlgebraic(algebraic)),
      meta,
      elapsedMs,
    };
  }

  throw new Error(`Unknown player id: ${playerConfig.id}`);
}

export async function playOneGame(playerA, playerB, options = {}) {
  let game = createGorisansonGame();
  const algebraicHistory = [];
  const moveThinkLog = [];
  const errors = [];
  let plies = 0;
  const stats = {
    byEngine: {
      [playerA.id]: { plies: 0, simulations: 0, nodes: 0, rootWinRateSum: 0, rootWinRateSamples: 0 },
      [playerB.id]: { plies: 0, simulations: 0, nodes: 0, rootWinRateSum: 0, rootWinRateSamples: 0 },
    },
  };
  const logMoves = options.logMoves !== false && !options.quiet;
  const tiBudget = resolveThinkBudget(options, playerA);
  const goBudget = resolveThinkBudget(options, playerB);

  while (winnerIndex(game) === null && plies < MAX_PLIES) {
    const side = game.pawnOfTurn.index;
    const cfg = side === 0 ? playerA : playerB;
    const ply = plies + 1;
    const playerBudget = resolveThinkBudget(options, cfg);
    const label = engineLabel(cfg, playerBudget);

    let move;
    let meta = {};
    let chooseError = null;

    try {
      const chosen = await chooseMove(game, algebraicHistory, cfg, ply, options);
      move = chosen.move;
      meta = chosen.meta ?? {};
      if (chosen.elapsedMs != null && meta.elapsedMs == null) {
        meta.elapsedMs = chosen.elapsedMs;
      }
    } catch (err) {
      chooseError = err?.message ?? String(err);
      const fb = fallbackLegalMove(game);
      if (!fb) {
        errors.push({
          ply,
          side: side === 0 ? 'White' : 'Black',
          engine: cfg.id,
          move: '(none)',
          reason: `search crashed: ${chooseError}`,
          fallback: null,
        });
        break;
      }
      move = fb;
      meta = { stoppedBy: 'crash-fallback', elapsedMs: 0 };
      errors.push({
        ply,
        side: side === 0 ? 'White' : 'Black',
        engine: cfg.id,
        move: '(search failed)',
        reason: chooseError,
        fallback: moveLabel(fb),
      });
    }

    const algebraic = moveLabel(move);
    const check = validateMove(game, move);
    let usedFallback = false;

    if (!check.ok) {
      const fb = fallbackLegalMove(game);
      errors.push({
        ply,
        side: side === 0 ? 'White' : 'Black',
        engine: cfg.id,
        move: algebraic,
        reason: check.reason,
        fallback: fb ? moveLabel(fb) : null,
      });
      if (!fb) {
        termLine(`  !! ply ${ply} ${cfg.id}: no legal fallback after illegal ${algebraic}`);
        break;
      }
      move = fb;
      usedFallback = true;
      meta.illegal = true;
    }

    if (stats.byEngine[cfg.id]) {
      stats.byEngine[cfg.id].plies += 1;
      stats.byEngine[cfg.id].simulations += meta?.simulations ?? 0;
      stats.byEngine[cfg.id].nodes += meta?.nodes ?? 0;
      if (meta?.rootWinRate != null && Number.isFinite(meta.rootWinRate)) {
        stats.byEngine[cfg.id].rootWinRateSum += meta.rootWinRate;
        stats.byEngine[cfg.id].rootWinRateSamples += 1;
      }
    }

    applyGorisansonMove(game, move);
    const applied = moveLabel(move);
    algebraicHistory.push(toAlgebraic(gorisansonMoveToAction(move)));
    plies += 1;

    const pos = evalPosition(game);
    const whiteDist = meta.whiteDist ?? pos.whiteDist;
    const blackDist = meta.blackDist ?? pos.blackDist;
    const margin =
      whiteDist < 200 && blackDist < 200 ? blackDist - whiteDist : pos.margin;
    const budgetHint = `${label}: ${playerBudget.timeSec}s · ≤${formatSimsCap(playerBudget.maxSimulations)}`;

    moveThinkLog.push({
      ply,
      engine: cfg.id === RUST_TITANIUM_ID ? 'Titanium αβ + CAT' : 'Gorisanson (JS, original)',
      move: applied,
      budgetHint,
      whiteDist,
      blackDist,
      margin,
      nodes: meta.nodes,
      simulations: meta.simulations,
      rootWinRate: meta.rootWinRate,
      rootScore: meta.rootScore,
      depthLog: meta.depthLog,
      searchDepth: meta.searchDepth,
      lmrReSearches: meta.lmrReSearches,
      stoppedBy: meta.stoppedBy,
      elapsedMs: meta.elapsedMs,
      error: chooseError ?? (check.ok ? null : check.reason),
      fallback: usedFallback ? applied : null,
    });

    if (typeof options.onPly === 'function') {
      options.onPly({
        ply,
        whiteId: playerA.id,
        blackId: playerB.id,
        algebraicHistory: [...algebraicHistory],
        margin,
      });
    }

    if (logMoves) {
      printPlyCompact({
        ply,
        who: side,
        engine: label,
        result: meta,
        move,
      });
      if (margin != null) {
        termLine(`      race margin=${margin > 0 ? `+${margin}` : margin} (W${pos.whiteDist} B${pos.blackDist})`);
      }
    } else if (options.verbose) {
      const wr =
        meta.rootWinRate != null ? ` wr=${Math.round(meta.rootWinRate * 100)}%` : '';
      termLine(
        `  ply ${ply} P${side + 1} (${cfg.id}): ${applied} · ${meta.simulations ?? meta.nodes ?? 0} · ${meta.stoppedBy}${wr} · margin=${margin}`,
      );
    }
  }

  const winner = winnerIndex(game);
  const replayCode = encodeReplayFromAlgebraic(algebraicHistory, {
    a: playerA.id,
    b: playerB.id,
    plies,
    winner: winner === null ? 'draw' : winner === 0 ? playerA.id : playerB.id,
  });

  const posEnd = evalPosition(game);
  const lastThink = moveThinkLog[moveThinkLog.length - 1];
  const finalPos = {
    whiteDist:
      lastThink?.whiteDist != null && lastThink.whiteDist < 200
        ? lastThink.whiteDist
        : posEnd.whiteDist,
    blackDist:
      lastThink?.blackDist != null && lastThink.blackDist < 200
        ? lastThink.blackDist
        : posEnd.blackDist,
    margin:
      lastThink?.margin != null && Math.abs(lastThink.margin) < 200
        ? lastThink.margin
        : posEnd.margin,
  };
  const report = buildGameReport({
    game,
    winnerPawn: winner,
    whiteId: playerA.id,
    blackId: playerB.id,
    replayCode,
    moveThinkLog,
    errors,
    tiBudget,
    goBudget,
    gameIndex: options.gameIndex,
  });

  const base = {
    plies,
    replayCode,
    algebraicHistory,
    game,
    stats,
    moveThinkLog,
    errors,
    report,
    finalPos,
  };

  if (winner === null) {
    return { result: 'draw', winner: null, ...base };
  }
  return {
    result: 'decided',
    winner: winner === 0 ? playerA.id : playerB.id,
    winnerPawn: winner,
    ...base,
  };
}

export async function playMatch(playerA, playerB, games, options = {}) {
  let scoreA = 0;
  let scoreB = 0;
  let draws = 0;
  const results = [];
  const swapColors = options.swapColors !== false;
  const logMoves = options.logMoves !== false && !options.quiet;
  const tiBudget = resolveThinkBudget(options, playerA);
  const goBudget = resolveThinkBudget(options, playerB);

  for (let i = 0; i < games; i++) {
    const swap = swapColors && i % 2 === 1;
    const light = swap ? playerB : playerA;
    const dark = swap ? playerA : playerB;

    if (logMoves || options.verbose) {
      termLine('');
      termLine(
        `── Game ${i + 1}/${games} · White=${light.id} · Black=${dark.id} · Ti ${tiBudget.timeSec}s/${formatSimsCap(tiBudget.maxSimulations)} · Go ${goBudget.timeSec}s/${formatSimsCap(goBudget.maxSimulations)} ──`,
      );
    }

    if (typeof options.onGameStart === 'function') {
      options.onGameStart({ gameIndex: i + 1, totalGames: games, whiteId: light.id, blackId: dark.id });
    }

    const outcome = await playOneGame(light, dark, { ...options, gameIndex: i + 1 });
    results.push(outcome);

    if (options.printReport !== false && outcome.report) {
      console.log('');
      console.log(outcome.report);
    }

    if (logMoves) {
      const winnerLabel =
        outcome.winner === null
          ? null
          : outcome.winner === playerA.id
            ? playerA.id
            : playerB.id;
      printFinalPosition(outcome.game, {
        winnerSide: outcome.winnerPawn ?? null,
        winnerLabel,
        algebraicHistory: outcome.algebraicHistory,
      });
    }

    if (options.logReplay !== false) {
      termLine(
        formatReplayBlock(outcome.replayCode, {
          label: `REPLAY game ${i + 1} — paste in web Replay tab`,
        }),
      );
    }

    if (outcome.result === 'draw') {
      draws += 1;
      scoreA += 0.5;
      scoreB += 0.5;
      continue;
    }

    if (outcome.winner === playerA.id) {
      scoreA += 1;
    } else if (outcome.winner === playerB.id) {
      scoreB += 1;
    }
  }

  return { playerA, playerB, games, scoreA, scoreB, draws, results };
}

export function eloFromMatch(scoreA, scoreB, games, ratingA = 1500, ratingB = 1500, k = 32) {
  const expectedA = 1 / (1 + 10 ** ((ratingB - ratingA) / 400));
  const actualA = scoreA / games;
  return {
    ratingA: ratingA + k * (actualA - expectedA),
    ratingB: ratingB + k * ((1 - actualA) - (1 - expectedA)),
    expectedA,
  };
}
