#!/usr/bin/env node
/**
 * Terminal minimax tuning harness — Titanium minimax vs Gorisanson MCTS.
 *
 *   node benchmark/tune_minimax.mjs --games 1 --time 10 --gorisanson-time 10
 *   node benchmark/tune_minimax.mjs --games 1 --time 10 --gorisanson-time 0.5 --report-dir benchmark/overnight
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { defaultPlayerConfigs, playMatch } from './lib/match_engine.mjs';
import { RUST_TITANIUM_ID, GORISANSON_ID } from './lib/engine_ids.mjs';
import { GORISANSON_MAX_VISITS, TITANIUM_MAX_NODES } from './lib/bench_limits.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const BASELINE_PATH = path.join(ROOT, 'benchmark', 'baseline_depths.json');

function parseArgs(argv) {
  const opts = {
    games: 4,
    timeSec: 10,
    gorisansonTimeSec: 10,
    quiet: true,
    disableBook: false,
    reportDir: null,
    swapColors: false,
  };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--games' && argv[i + 1]) opts.games = Number(argv[++i]);
    else if (arg === '--time' && argv[i + 1]) opts.timeSec = Number(argv[++i]);
    else if (arg === '--gorisanson-time' && argv[i + 1]) {
      opts.gorisansonTimeSec = Number(argv[++i]);
    } else if (arg === '--report-dir' && argv[i + 1]) opts.reportDir = argv[++i];
    else if (arg === '--swap-colors') opts.swapColors = true;
    else if (arg === '--no-book') opts.disableBook = true;
    else if (arg === '--verbose' || arg === '-v') opts.quiet = false;
    else if (arg === '--label' && argv[i + 1]) opts.label = argv[++i];
  }
  return opts;
}

function avgRootWinRate(stats) {
  if (!stats?.rootWinRateSamples) return null;
  return stats.rootWinRateSum / stats.rootWinRateSamples;
}

function summarizeGame(game, gameIndex, tiTime, goTime) {
  const ti = game.stats?.byEngine?.[RUST_TITANIUM_ID] ?? {};
  const go = game.stats?.byEngine?.[GORISANSON_ID] ?? {};
  return {
    gameIndex,
    winner: game.winner,
    winnerPawn: game.winnerPawn,
    plies: game.plies,
    finalMargin: game.finalPos?.margin,
    whiteDist: game.finalPos?.whiteDist,
    blackDist: game.finalPos?.blackDist,
    errors: game.errors?.length ?? 0,
    illegalMoves: game.errors,
    tiNodes: ti.nodes ?? 0,
    goSims: go.simulations ?? 0,
    goAvgWinRate: avgRootWinRate(go),
    tiAvgNodesPerMove: ti.plies ? Math.round((ti.nodes ?? 0) / ti.plies) : 0,
    goAvgSimsPerMove: go.plies ? Math.round((go.simulations ?? 0) / go.plies) : 0,
    replay: game.replayCode,
    reportPath: null,
  };
}

async function main() {
  const opts = parseArgs(process.argv);
  const label = opts.label ?? process.env.TUNE_LABEL ?? 'default';
  const players = defaultPlayerConfigs({
    timeSec: opts.timeSec,
    gorisansonTimeSec: opts.gorisansonTimeSec,
  });

  const titanium = {
    ...players.titanium,
    timeSec: opts.timeSec,
    maxSimulations: Number(process.env.TITANIUM_MAX_NODES ?? TITANIUM_MAX_NODES),
    disableBook: opts.disableBook,
  };
  const gorisanson = {
    ...players.gorisanson,
    timeSec: opts.gorisansonTimeSec,
    maxSimulations: Number(process.env.GORISANSON_MAX_VISITS ?? GORISANSON_MAX_VISITS),
  };

  if (!opts.quiet && opts.label) {
    console.error(
      `════ ${opts.label} · Ti ${opts.timeSec}s/10B vs Go ${opts.gorisansonTimeSec}s/66k ════`,
    );
  }

  const started = performance.now();
  const match = await playMatch(titanium, gorisanson, opts.games, {
    engine: 'minimax',
    disableBook: opts.disableBook,
    quiet: opts.quiet,
    logMoves: !opts.quiet,
    logReplay: !opts.quiet,
    logSearch: true,
    printReport: !opts.quiet,
    swapColors: opts.swapColors,
    useCatGuidance: true,
  });
  const wallSec = (performance.now() - started) / 1000;

  if (opts.reportDir) {
    fs.mkdirSync(opts.reportDir, { recursive: true });
  }

  const gameSummaries = [];
  let totalPlies = 0;
  let totalNodes = 0;
  let totalGoSims = 0;
  let totalErrors = 0;

  for (let i = 0; i < match.results.length; i++) {
    const game = match.results[i];
    totalPlies += game.plies ?? 0;
    totalErrors += game.errors?.length ?? 0;
    const ti = game.stats?.byEngine?.[RUST_TITANIUM_ID];
    const go = game.stats?.byEngine?.[GORISANSON_ID];
    if (ti) totalNodes += ti.nodes ?? 0;
    if (go) totalGoSims += go.simulations ?? 0;

    const summary = summarizeGame(game, i + 1, opts.timeSec, opts.gorisansonTimeSec);
    if (opts.reportDir && game.report) {
      const reportPath = path.join(opts.reportDir, `${label}-game${i + 1}.txt`);
      fs.writeFileSync(reportPath, game.report, 'utf8');
      summary.reportPath = reportPath;
      const jsonPath = path.join(opts.reportDir, `${label}-game${i + 1}.json`);
      fs.writeFileSync(
        jsonPath,
        JSON.stringify(
          {
            ...summary,
            moveThinkLog: game.moveThinkLog,
            errors: game.errors,
          },
          null,
          2,
        ),
        'utf8',
      );
    }
    gameSummaries.push(summary);
  }

  const summary = {
    label,
    games: opts.games,
    timeSec: opts.timeSec,
    gorisansonTimeSec: opts.gorisansonTimeSec,
    titaniumMaxNodes: Number(process.env.TITANIUM_MAX_NODES ?? TITANIUM_MAX_NODES),
    gorisansonMaxVisits: Number(process.env.GORISANSON_MAX_VISITS ?? GORISANSON_MAX_VISITS),
    score: `${match.scoreA}-${match.scoreB}`,
    draws: match.draws,
    winRate: opts.games ? match.scoreA / opts.games : 0,
    wallSec: Number(wallSec.toFixed(1)),
    avgPlies: opts.games ? totalPlies / opts.games : 0,
    avgNodesPerMove: totalPlies ? Math.round(totalNodes / totalPlies) : 0,
    avgGoSimsPerMove: totalPlies ? Math.round(totalGoSims / totalPlies) : 0,
    illegalMoveCount: totalErrors,
    games_detail: gameSummaries,
    bin: process.env.TITANIUM_BIN ?? 'default',
  };

  let baselineDelta = null;
  if (fs.existsSync(BASELINE_PATH)) {
    const baseline = JSON.parse(fs.readFileSync(BASELINE_PATH, 'utf8'));
    const opening = baseline.positions?.opening_ply0;
    if (opening?.searchDepth != null) {
      baselineDelta = {
        openingDepthBaseline: opening.searchDepth,
        openingDepthNow: reportOpeningDepth(match),
        note: 'Compare opening_ply0 searchDepth vs baseline_depths.json',
      };
    }
  }

  console.log(JSON.stringify({ ...summary, baselineDelta }));
  process.exit(match.scoreA > match.scoreB ? 0 : 1);
}

function reportOpeningDepth(match) {
  for (const game of match.results ?? []) {
    const first = game.moveThinkLog?.find((e) => e.searchDepth != null);
    if (first?.searchDepth != null) {
      return first.searchDepth;
    }
  }
  return null;
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(2);
});
