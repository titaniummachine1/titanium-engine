#!/usr/bin/env node
/**
 * Parallel Titanium vs Quoridor v3 — short regression batches.
 *
 *   node benchmark/parallel_vs_v3.mjs --workers 4 --games 4 --time 10 --v3-ms 500
 */

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { RUST_TITANIUM_ID, QUORIDOR_V3_ID } from './lib/engine_ids.mjs';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const WORKER = path.join(ROOT, 'benchmark', 'tune_vs_v3.mjs');

function parseArgs(argv) {
  const opts = {
    workers: 4,
    games: 4,
    timeSec: 10,
    v3Ms: 500,
    label: 'ti-vs-v3',
    reportDir: null,
  };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--workers' && argv[i + 1]) opts.workers = Number(argv[++i]);
    else if (arg === '--games' && argv[i + 1]) opts.games = Number(argv[++i]);
    else if (arg === '--time' && argv[i + 1]) opts.timeSec = Number(argv[++i]);
    else if (arg === '--v3-ms' && argv[i + 1]) opts.v3Ms = Number(argv[++i]);
    else if (arg === '--report-dir' && argv[i + 1]) opts.reportDir = argv[++i];
    else if (arg === '--label' && argv[i + 1]) opts.label = argv[++i];
  }
  return opts;
}

function runOneGame(gameIndex, opts) {
  return new Promise((resolve, reject) => {
    const reportDir = opts.reportDir
      ? path.join(opts.reportDir, `worker-g${gameIndex}`)
      : null;
    if (reportDir) fs.mkdirSync(reportDir, { recursive: true });

    const args = [
      WORKER,
      '--games',
      '1',
      '--time',
      String(opts.timeSec),
      '--v3-ms',
      String(opts.v3Ms),
      '--label',
      `${opts.label}-g${gameIndex}`,
    ];
    if (reportDir) args.push('--report-dir', reportDir);

    const child = spawn(process.execPath, args, {
      cwd: ROOT,
      env: { ...process.env, TITANIUM_ENGINE: 'minimax' },
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let jsonLine = '';
    child.stdout.on('data', (c) => {
      jsonLine += c;
    });
    child.stderr.on('data', (c) => {
      process.stderr.write(`[g${gameIndex}] ${c}`);
    });
    child.on('error', reject);
    child.on('close', (code) => {
      const line = jsonLine.trim().split(/\r?\n/).find((l) => l.startsWith('{'));
      if (!line) {
        reject(new Error(`vs-v3 game ${gameIndex}: no JSON`));
        return;
      }
      try {
        resolve({ gameIndex, code: code ?? 1, summary: JSON.parse(line) });
      } catch (err) {
        reject(err);
      }
    });
  });
}

async function runPool(opts) {
  const results = [];
  let next = 1;
  let inFlight = 0;
  const started = performance.now();

  return new Promise((resolve, reject) => {
    function launch() {
      while (inFlight < opts.workers && next <= opts.games) {
        const idx = next++;
        inFlight += 1;
        runOneGame(idx, opts)
          .then((r) => {
            inFlight -= 1;
            results.push(r);
            if (next > opts.games && inFlight === 0) {
              resolve({ results, wallSec: (performance.now() - started) / 1000 });
            } else {
              launch();
            }
          })
          .catch(reject);
      }
    }
    launch();
  });
}

function aggregate(results, wallSec, opts) {
  let tiWins = 0;
  let v3Wins = 0;
  let totalPlies = 0;
  let illegalMoveCount = 0;
  const games_detail = [];

  for (const { summary } of results) {
    const gd = summary.gamesDetail?.[0];
    if (gd) {
      games_detail.push(gd);
      if (gd.winner === RUST_TITANIUM_ID) tiWins += 1;
      else if (gd.winner === QUORIDOR_V3_ID) v3Wins += 1;
      totalPlies += gd.plies ?? 0;
      illegalMoveCount += gd.errors ?? 0;
    }
  }

  const games = results.length;
  return {
    label: opts.label,
    opponent: QUORIDOR_V3_ID,
    workers: opts.workers,
    games,
    timeSec: opts.timeSec,
    v3Ms: opts.v3Ms,
    score: `${tiWins}-${v3Wins}`,
    winRate: games ? tiWins / games : 0,
    wallSec: Number(wallSec.toFixed(1)),
    avgPlies: games ? Number((totalPlies / games).toFixed(1)) : 0,
    illegalMoveCount,
    games_detail,
  };
}

async function main() {
  const opts = parseArgs(process.argv);
  if (opts.reportDir) fs.mkdirSync(opts.reportDir, { recursive: true });

  const { results, wallSec } = await runPool(opts);
  const summary = aggregate(results, wallSec, opts);

  if (opts.reportDir) {
    fs.writeFileSync(
      path.join(opts.reportDir, `${opts.label}-aggregate.json`),
      JSON.stringify(summary, null, 2),
    );
  }

  console.log(`OVERNIGHT_JSON:${JSON.stringify(summary)}`);
  process.exit(summary.illegalMoveCount > 0 ? 2 : 0);
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(2);
});
