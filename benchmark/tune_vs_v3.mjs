#!/usr/bin/env node
/**
 * Rust Titanium (long think) vs Quoridor v3 (500ms default).
 *
 *   node benchmark/tune_vs_v3.mjs --games 1 --time 10 --v3-ms 500
 */

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { playMatch } from './lib/match_engine.mjs';
import { RUST_TITANIUM_ID, QUORIDOR_V3_ID } from './lib/engine_ids.mjs';
import { TITANIUM_MAX_NODES } from './lib/bench_limits.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');

function parseArgs(argv) {
  const opts = {
    games: 2,
    timeSec: 10,
    v3Ms: 500,
    quiet: true,
    reportDir: null,
    label: 'ti-vs-v3',
  };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--games' && argv[i + 1]) opts.games = Number(argv[++i]);
    else if (arg === '--time' && argv[i + 1]) opts.timeSec = Number(argv[++i]);
    else if (arg === '--v3-ms' && argv[i + 1]) opts.v3Ms = Number(argv[++i]);
    else if (arg === '--report-dir' && argv[i + 1]) opts.reportDir = argv[++i];
    else if (arg === '--label' && argv[i + 1]) opts.label = argv[++i];
    else if (arg === '--verbose' || arg === '-v') opts.quiet = false;
  }
  return opts;
}

function summarizeGame(game, gameIndex) {
  const ti = game.stats?.byEngine?.[RUST_TITANIUM_ID] ?? {};
  const v3 = game.stats?.byEngine?.[QUORIDOR_V3_ID] ?? {};
  return {
    gameIndex,
    winner: game.winner,
    winnerPawn: game.winnerPawn,
    plies: game.plies,
    errors: game.errors?.length ?? 0,
    tiNodes: ti.nodes ?? 0,
    v3Nodes: v3.nodes ?? 0,
    replay: game.replayCode,
  };
}

async function main() {
  const opts = parseArgs(process.argv);
  const titanium = {
    id: RUST_TITANIUM_ID,
    engine: 'minimax',
    timeSec: opts.timeSec,
    maxSimulations: Number(process.env.TITANIUM_MAX_NODES ?? TITANIUM_MAX_NODES),
    useCatGuidance: true,
  };
  const quoridorV3 = {
    id: QUORIDOR_V3_ID,
    timeSec: opts.v3Ms / 1000,
    maxDepth: 24,
  };

  const started = performance.now();
  const match = await playMatch(titanium, quoridorV3, opts.games, {
    engine: 'minimax',
    quiet: opts.quiet,
    logMoves: !opts.quiet,
    swapColors: true,
    useCatGuidance: true,
  });
  const wallSec = (performance.now() - started) / 1000;

  const gamesDetail = match.results.map((g, i) => summarizeGame(g, i + 1));
  const summary = {
    label: opts.label,
    titaniumTimeSec: opts.timeSec,
    v3Ms: opts.v3Ms,
    games: opts.games,
    wallSec,
    score: `${match.scoreA}-${match.scoreB}`,
    draws: match.draws,
    errors: gamesDetail.reduce((n, g) => n + g.errors, 0),
    gamesDetail,
  };

  if (opts.reportDir) {
    fs.mkdirSync(opts.reportDir, { recursive: true });
    fs.writeFileSync(
      path.join(opts.reportDir, `${opts.label}.json`),
      `${JSON.stringify(summary, null, 2)}\n`,
    );
  }

  console.log(JSON.stringify(summary));
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
