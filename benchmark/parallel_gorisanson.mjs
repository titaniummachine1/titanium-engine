#!/usr/bin/env node
/**
 * Parallel Titanium minimax vs Gorisanson — one game per worker process.
 *
 *   node benchmark/parallel_gorisanson.mjs --workers 4 --games 4 --time 10 --gorisanson-time 10
 *   node benchmark/parallel_gorisanson.mjs --workers 4 --games 4 --time 10 --report-dir benchmark/overnight
 */

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const WORKER = path.join(ROOT, 'benchmark', 'tune_minimax.mjs');

function parseArgs(argv) {
  const opts = {
    workers: 4,
    games: 4,
    timeSec: 10,
    gorisansonTimeSec: 10,
    verbose: false,
    label: 'parallel',
    reportDir: null,
  };

  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--workers' && argv[i + 1]) opts.workers = Number(argv[++i]);
    else if (arg === '--games' && argv[i + 1]) opts.games = Number(argv[++i]);
    else if (arg === '--time' && argv[i + 1]) opts.timeSec = Number(argv[++i]);
    else if (arg === '--gorisanson-time' && argv[i + 1]) {
      opts.gorisansonTimeSec = Number(argv[++i]);
    } else if (arg === '--report-dir' && argv[i + 1]) opts.reportDir = argv[++i];
    else if (arg === '--label' && argv[i + 1]) opts.label = argv[++i];
    else if (arg === '--verbose' || arg === '-v') opts.verbose = true;
  }

  return opts;
}

function prefixLines(stream, tag, onLine) {
  let carry = '';
  stream.setEncoding('utf8');
  stream.on('data', (chunk) => {
    carry += chunk;
    const parts = carry.split(/\r?\n/);
    carry = parts.pop() ?? '';
    for (const line of parts) {
      if (!line) {
        continue;
      }
      process.stdout.write(`${tag} ${line}\n`);
      onLine?.(line);
    }
  });
}

function runOneGame(gameIndex, opts) {
  return new Promise((resolve, reject) => {
    const tag = `[g${gameIndex}]`;
    const reportDir = opts.reportDir
      ? path.join(opts.reportDir, `worker-g${gameIndex}`)
      : null;
    if (reportDir) {
      fs.mkdirSync(reportDir, { recursive: true });
    }

    const args = [
      WORKER,
      '--games',
      '1',
      '--time',
      String(opts.timeSec),
      '--gorisanson-time',
      String(opts.gorisansonTimeSec),
      '--label',
      `${opts.label}-g${gameIndex}`,
    ];
    if (reportDir) {
      args.push('--report-dir', reportDir);
    }
    if (opts.verbose) {
      args.push('-v');
    }

    const env = {
      ...process.env,
      TITANIUM_ENGINE: 'minimax',
      TITANIUM_MAX_NODES: process.env.TITANIUM_MAX_NODES ?? '10000000000',
      GORISANSON_MAX_VISITS: process.env.GORISANSON_MAX_VISITS ?? '66000',
    };

    const child = spawn(process.execPath, args, {
      cwd: ROOT,
      env,
      stdio: ['ignore', 'pipe', 'pipe'],
    });

    let jsonLine = null;
    let reportText = null;

    prefixLines(child.stdout, tag, (line) => {
      if (line.startsWith('{')) {
        jsonLine = line;
      }
      if (line.startsWith('=== Quoridor game report ===')) {
        reportText = line;
      } else if (reportText != null && !line.startsWith('{')) {
        reportText += `\n${line}`;
      }
    });
    prefixLines(child.stderr, tag);

    child.on('error', reject);
    child.on('close', (code) => {
      if (!jsonLine) {
        reject(new Error(`game ${gameIndex}: no JSON output`));
        return;
      }

      try {
        const summary = JSON.parse(jsonLine);
        if (opts.reportDir && reportText) {
          const rp = path.join(opts.reportDir, `${opts.label}-g${gameIndex}-report.txt`);
          fs.writeFileSync(rp, reportText, 'utf8');
          summary.reportFile = rp;
        }
        resolve({ gameIndex, code: code ?? 1, summary });
      } catch (err) {
        reject(new Error(`game ${gameIndex}: bad JSON — ${err.message}\n${jsonLine}`));
      }
    });
  });
}

async function runPool(opts) {
  const results = [];
  let nextGame = 1;
  let inFlight = 0;
  const started = performance.now();

  return new Promise((resolve, reject) => {
    function launch() {
      while (inFlight < opts.workers && nextGame <= opts.games) {
        const gameIndex = nextGame++;
        inFlight += 1;

        process.stdout.write(`[start] game ${gameIndex}/${opts.games}\n`);

        runOneGame(gameIndex, opts)
          .then((result) => {
            inFlight -= 1;
            results.push(result);

            if (result.summary) {
              const s = result.summary;
              const gd = s.games_detail?.[0];
              process.stdout.write(
                `[done]  game ${gameIndex}: ${gd?.winner ?? s.score} · ${gd?.plies ?? s.avgPlies} plies · ` +
                  `margin=${gd?.finalMargin ?? '?'} · goWR=${gd?.goAvgWinRate != null ? (gd.goAvgWinRate * 100).toFixed(0) + '%' : '?'} · ` +
                  `illegal=${s.illegalMoveCount ?? 0}\n`,
              );
            }

            if (nextGame > opts.games && inFlight === 0) {
              resolve({ results, wallSec: (performance.now() - started) / 1000 });
              return;
            }

            launch();
          })
          .catch(reject);
      }
    }

    launch();
  });
}

function aggregate(results, wallSec, opts) {
  let titaniumWins = 0;
  let gorisansonWins = 0;
  let draws = 0;
  let totalPlies = 0;
  let totalNodes = 0;
  let totalGoSims = 0;
  let illegalMoveCount = 0;
  let nodeSamples = 0;
  const games_detail = [];

  for (const { summary } of results) {
    if (!summary) {
      continue;
    }
    const gd = summary.games_detail?.[0];
    if (gd) {
      games_detail.push(gd);
      if (gd.winner === 'rust-titanium') titaniumWins += 1;
      else if (gd.winner === 'gorisanson') gorisansonWins += 1;
      totalPlies += gd.plies ?? 0;
      totalNodes += gd.tiNodes ?? 0;
      totalGoSims += gd.goSims ?? 0;
      illegalMoveCount += gd.errors ?? 0;
      if (gd.tiAvgNodesPerMove) nodeSamples += gd.plies ?? 0;
    } else {
      const [a, b] = summary.score.split('-').map(Number);
      titaniumWins += a;
      gorisansonWins += b;
    }
    draws += summary.draws ?? 0;
  }

  const games = results.length;
  return {
    label: opts.label,
    workers: opts.workers,
    games,
    timeSec: opts.timeSec,
    gorisansonTimeSec: opts.gorisansonTimeSec,
    titaniumMaxNodes: Number(process.env.TITANIUM_MAX_NODES ?? 10_000_000_000),
    gorisansonMaxVisits: Number(process.env.GORISANSON_MAX_VISITS ?? 66_000),
    engine: 'minimax',
    score: `${titaniumWins}-${gorisansonWins}`,
    draws,
    winRate: games ? titaniumWins / games : 0,
    wallSec: Number(wallSec.toFixed(1)),
    avgPlies: games ? Number((totalPlies / games).toFixed(1)) : 0,
    avgNodesPerMove: nodeSamples ? Math.round(totalNodes / nodeSamples) : 0,
    avgGoSimsPerMove: totalPlies ? Math.round(totalGoSims / totalPlies) : 0,
    illegalMoveCount,
    games_detail,
  };
}

async function main() {
  const opts = parseArgs(process.argv);
  if (opts.reportDir) {
    fs.mkdirSync(opts.reportDir, { recursive: true });
  }

  console.log(
    `Parallel minimax vs Gorisanson — ${opts.games} games · ${opts.workers} workers · ` +
      `Ti ${opts.timeSec}s/10B · Go ${opts.gorisansonTimeSec}s/66k`,
  );
  console.log('');

  const { results, wallSec } = await runPool(opts);
  const summary = aggregate(results, wallSec, opts);

  if (opts.reportDir) {
    fs.writeFileSync(
      path.join(opts.reportDir, `${opts.label}-aggregate.json`),
      JSON.stringify(summary, null, 2),
      'utf8',
    );
  }

  console.log('');
  // Single-line marker for overnight_iterate (pretty JSON also on disk).
  console.log(`OVERNIGHT_JSON:${JSON.stringify(summary)}`);
  process.exit(summary.winRate > 0.5 ? 0 : 1);
}

main().catch((err) => {
  console.error(err?.stack || String(err));
  process.exit(2);
});
