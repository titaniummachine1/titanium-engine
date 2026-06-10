#!/usr/bin/env node
/**
 * Adaptive overnight tournament — design-driven LMR (no per-game heuristic tweaks).
 *
 *   node benchmark/overnight_iterate.mjs --hours 10 --workers 6
 *   node benchmark/overnight_iterate.mjs --resume
 *
 * Strategy:
 *   - PROBE: 2 games, fast time ladder → find interesting stress points
 *   - CONFIRM: 4–6 games only when probe isn't a clean blowout (WR<100% or margin≤5)
 *   - Perft gate on illegal moves only; LMR changes stay in engine (time_budget + stage_t)
 */

import { spawn } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const OUT_DIR = path.join(ROOT, 'benchmark', 'overnight');
const CHECKPOINT_DIR = path.join(OUT_DIR, 'checkpoints');
const PARALLEL = path.join(ROOT, 'benchmark', 'parallel_gorisanson.mjs');
const KEEP_AWAKE = path.join(ROOT, 'benchmark', 'keep_awake.ps1');
const LOG_PATH = path.join(OUT_DIR, 'overnight.log');
const CHECKPOINT_PATH = path.join(OUT_DIR, 'checkpoint.json');
const STATUS_PATH = path.join(OUT_DIR, 'STATUS.md');

/** Fast probes — many per night. */
const PROBES = [
  { label: 'probe-10v10', timeSec: 10, gorisansonTimeSec: 10 },
  { label: 'probe-5v10', timeSec: 5, gorisansonTimeSec: 10 },
  { label: 'probe-10v5', timeSec: 10, gorisansonTimeSec: 5 },
  { label: 'probe-3v10', timeSec: 3, gorisansonTimeSec: 10 },
  { label: 'probe-8v12', timeSec: 8, gorisansonTimeSec: 12 },
  { label: 'probe-10v15', timeSec: 10, gorisansonTimeSec: 15 },
  { label: 'probe-10v20', timeSec: 10, gorisansonTimeSec: 20 },
  { label: 'probe-2v10', timeSec: 2, gorisansonTimeSec: 10 },
];

function parseArgs(argv) {
  const opts = {
    hours: 10,
    workers: 6,
    probeGames: 2,
    confirmGames: 4,
    resume: false,
    skipPerft: false,
  };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--hours' && argv[i + 1]) opts.hours = Number(argv[++i]);
    else if (arg === '--workers' && argv[i + 1]) opts.workers = Number(argv[++i]);
    else if (arg === '--probe-games' && argv[i + 1]) opts.probeGames = Number(argv[++i]);
    else if (arg === '--confirm-games' && argv[i + 1]) opts.confirmGames = Number(argv[++i]);
    else if (arg === '--resume') opts.resume = true;
    else if (arg === '--skip-perft') opts.skipPerft = true;
  }
  return opts;
}

function log(msg) {
  const line = `[${new Date().toISOString()}] ${msg}`;
  console.log(line);
  fs.mkdirSync(OUT_DIR, { recursive: true });
  fs.appendFileSync(LOG_PATH, `${line}\n`, 'utf8');
}

function writeStatus(state) {
  const lines = [
    '# Overnight tournament',
    '',
    `Updated: ${new Date().toISOString()}`,
    '',
    `| Metric | Value |`,
    `|--------|-------|`,
    `| Step | ${state.stepIndex ?? 0} |`,
    `| Last | ${state.lastLabel ?? '—'} |`,
    `| Last score | ${state.lastScore ?? '—'} |`,
    `| Probes run | ${state.probesRun ?? 0} |`,
    `| Confirms run | ${state.confirmsRun ?? 0} |`,
    `| Workers | ${state.opts?.workers ?? '?'} |`,
    `| Deadline | ${state.deadline ? new Date(state.deadline).toISOString() : '—'} |`,
    '',
    'LMR is **not** micromanaged here — engine uses `apply_time_budget` + `stage_t` by design.',
    '',
    '## Resume',
    '```',
    'node benchmark/overnight_iterate.mjs --resume --workers 6',
    '```',
  ];
  fs.writeFileSync(STATUS_PATH, lines.join('\n'), 'utf8');
}

function saveCheckpoint(state) {
  fs.mkdirSync(CHECKPOINT_DIR, { recursive: true });
  fs.writeFileSync(CHECKPOINT_PATH, JSON.stringify(state, null, 2), 'utf8');
}

function loadCheckpoint() {
  if (!fs.existsSync(CHECKPOINT_PATH)) return null;
  return JSON.parse(fs.readFileSync(CHECKPOINT_PATH, 'utf8'));
}

function parseSummary(stdout, reportDir, label) {
  const marker = stdout
    .split(/\r?\n/)
    .find((l) => l.startsWith('OVERNIGHT_JSON:'));
  if (marker) {
    return JSON.parse(marker.slice('OVERNIGHT_JSON:'.length));
  }
  const agg = path.join(reportDir, `${label}-aggregate.json`);
  if (fs.existsSync(agg)) {
    return JSON.parse(fs.readFileSync(agg, 'utf8'));
  }
  const lines = stdout.split(/\r?\n/).filter((l) => l.startsWith('{'));
  if (lines.length) {
    try {
      return JSON.parse(lines.join('\n'));
    } catch {
      /* fall through */
    }
  }
  return null;
}

function runCmd(cmd, args, { cwd = ROOT, env = process.env, timeoutMs = 0 } = {}) {
  return new Promise((resolve, reject) => {
    const child = spawn(cmd, args, { cwd, env, stdio: ['ignore', 'pipe', 'pipe'] });
    let stdout = '';
    let stderr = '';
    let timer;
    if (timeoutMs > 0) {
      timer = setTimeout(() => {
        child.kill('SIGTERM');
        reject(new Error(`timeout ${timeoutMs}ms`));
      }, timeoutMs);
    }
    child.stdout.on('data', (c) => {
      stdout += c;
    });
    child.stderr.on('data', (c) => {
      stderr += c;
    });
    child.on('error', (e) => {
      if (timer) clearTimeout(timer);
      reject(e);
    });
    child.on('close', (code) => {
      if (timer) clearTimeout(timer);
      resolve({ code: code ?? 1, stdout, stderr });
    });
  });
}

let awakeChild = null;

function startKeepAwake() {
  if (awakeChild) return;
  awakeChild = spawn(
    'powershell',
    ['-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', KEEP_AWAKE],
    { cwd: ROOT, stdio: 'ignore', detached: true },
  );
  awakeChild.unref();
  log(`keep_awake started pid=${awakeChild.pid}`);
}

function stopKeepAwake() {
  if (awakeChild) {
    try {
      process.kill(awakeChild.pid);
    } catch {
      /* ignore */
    }
    awakeChild = null;
  }
}

async function buildEngine() {
  log('cargo build --release');
  const { code, stderr } = await runCmd('cargo', ['build', '--release'], {
    cwd: path.join(ROOT, 'engine'),
    timeoutMs: 600_000,
  });
  if (code !== 0) throw new Error(stderr.slice(-2000));
}

async function runPerftGate() {
  log('perft d4 gate');
  const { code, stderr, stdout } = await runCmd(
    'cargo',
    ['test', '--release', 'perft_depth4', '--', '--ignored', '--nocapture'],
    { cwd: path.join(ROOT, 'engine'), timeoutMs: 120_000 },
  );
  if (code !== 0) throw new Error(`perft failed:\n${stderr}\n${stdout}`);
  log('perft PASS');
}

function analyzeSummary(summary) {
  let minMargin = Infinity;
  let losses = 0;
  for (const g of summary.games_detail ?? []) {
    if (g.winner !== 'rust-titanium') losses += 1;
    if (g.finalMargin != null && g.finalMargin < 200) {
      minMargin = Math.min(minMargin, g.finalMargin);
    }
  }
  return {
    winRate: summary.winRate,
    losses,
    minMargin: Number.isFinite(minMargin) ? minMargin : null,
    illegalMoveCount: summary.illegalMoveCount ?? 0,
  };
}

function needsConfirm(analysis) {
  if (analysis.illegalMoveCount > 0) return true;
  if (analysis.losses > 0) return true;
  if (analysis.winRate < 1) return true;
  if (analysis.minMargin != null && analysis.minMargin <= 5) return true;
  return false;
}

async function runBatch(round, { games, workers, suffix = '' }) {
  const label = suffix ? `${round.label}${suffix}` : round.label;
  const reportDir = path.join(OUT_DIR, label);
  fs.mkdirSync(reportDir, { recursive: true });

  log(`BATCH ${label} · Ti ${round.timeSec}s Go ${round.gorisansonTimeSec}s · ${games}g ${workers}w`);

  const timeoutMs = Math.max(2_700_000, games * (round.timeSec + round.gorisansonTimeSec) * 50_000);

  const { code, stdout, stderr } = await runCmd(
    process.execPath,
    [
      PARALLEL,
      '--workers',
      String(Math.min(workers, games)),
      '--games',
      String(games),
      '--time',
      String(round.timeSec),
      '--gorisanson-time',
      String(round.gorisansonTimeSec),
      '--label',
      label,
      '--report-dir',
      reportDir,
    ],
    {
      env: {
        ...process.env,
        TITANIUM_ENGINE: 'minimax',
        TITANIUM_MAX_NODES: '10000000000',
        GORISANSON_MAX_VISITS: '66000',
      },
      timeoutMs,
    },
  );

  const summary = parseSummary(stdout, reportDir, label);
  if (!summary) {
    log(`BATCH ${label} FAILED parse — exit ${code}\n${stderr.slice(-1500)}`);
    return null;
  }

  const analysis = analyzeSummary(summary);
  log(
    `BATCH ${label}: ${summary.score} WR=${(summary.winRate * 100).toFixed(0)}% ` +
      `minMargin=${analysis.minMargin} illegal=${analysis.illegalMoveCount} wall=${summary.wallSec}s`,
  );

  fs.appendFileSync(
    path.join(OUT_DIR, 'history.jsonl'),
    `${JSON.stringify({ ts: new Date().toISOString(), label, games, workers, summary, analysis })}\n`,
  );

  return { summary, analysis, label };
}

async function gitCheckpoint(message) {
  const paths = [
    'benchmark/overnight/history.jsonl',
    'benchmark/overnight/checkpoint.json',
    'benchmark/overnight/STATUS.md',
    'engine/src/search/lmr_profile.rs',
    'engine/src/search/alphabeta.rs',
    'benchmark/overnight_iterate.mjs',
    'benchmark/parallel_gorisanson.mjs',
    'benchmark/lib/match_engine.mjs',
    'benchmark/lib/bench_limits.mjs',
  ];
  for (const p of paths) {
    const full = path.join(ROOT, p);
    if (fs.existsSync(full)) {
      await runCmd('git', ['add', p]);
    }
  }
  const { code, stderr, stdout } = await runCmd('git', ['commit', '-m', message]);
  if (code === 0) {
    const hash = stdout.match(/\[[\w/-]+ ([0-9a-f]+)\]/)?.[1] ?? 'ok';
    log(`git checkpoint ${hash}: ${message.split('\n')[0]}`);
  } else if (!`${stderr}${stdout}`.includes('nothing to commit')) {
    log(`git skip: ${stderr.slice(0, 200)}`);
  }
}

function ingestExistingResults(state) {
  const hist = path.join(OUT_DIR, 'history.jsonl');
  const existing = fs.existsSync(hist) ? fs.readFileSync(hist, 'utf8') : '';

  for (const [label, file] of [
    ['fair-10v10', 'fair-10v10-aggregate.json'],
    ['ti8-go12', 'ti8-go12-aggregate.json'],
  ]) {
    const agg = path.join(OUT_DIR, label, file);
    if (!fs.existsSync(agg) || existing.includes(`"${label}"`)) continue;
    const summary = JSON.parse(fs.readFileSync(agg, 'utf8'));
    const analysis = analyzeSummary(summary);
    fs.appendFileSync(
      hist,
      `${JSON.stringify({ ts: new Date().toISOString(), label, summary, analysis, note: 'recovered' })}\n`,
    );
    log(`Recovered ${label}: ${summary.score} WR=${summary.winRate}`);
    state.probesRun = (state.probesRun ?? 0) + 1;
    state.lastScore = summary.score;
    state.lastLabel = `${label} (recovered)`;
  }
}

async function main() {
  const opts = parseArgs(process.argv);
  let stepIndex = 0;
  let deadline = Date.now() + opts.hours * 3600 * 1000;
  const state = {
    stepIndex: 0,
    probesRun: 0,
    confirmsRun: 0,
    deadline,
    opts: { workers: opts.workers, probeGames: opts.probeGames, confirmGames: opts.confirmGames },
  };

  if (opts.resume) {
    const cp = loadCheckpoint();
    if (cp) {
      Object.assign(state, cp);
      stepIndex = cp.stepIndex ?? 0;
      deadline = cp.deadline ?? deadline;
      log(`Resume step ${stepIndex}`);
    }
  }

  fs.mkdirSync(OUT_DIR, { recursive: true });
  ingestExistingResults(state);
  startKeepAwake();

  log(`Overnight v2 · ${opts.hours}h · workers=${opts.workers} probe=${opts.probeGames} confirm=${opts.confirmGames}`);
  await buildEngine();
  if (!opts.skipPerft) await runPerftGate();

  writeStatus(state);
  saveCheckpoint(state);

  while (Date.now() < deadline) {
    const probe = PROBES[stepIndex % PROBES.length];
    stepIndex += 1;
    state.stepIndex = stepIndex;

    try {
      const probeWorkers = Math.min(2, opts.workers);
      const probeResult = await runBatch(probe, {
        games: opts.probeGames,
        workers: probeWorkers,
      });
      state.probesRun = (state.probesRun ?? 0) + 1;

      if (probeResult) {
        state.lastLabel = probeResult.label;
        state.lastScore = probeResult.summary.score;

        if (needsConfirm(probeResult.analysis)) {
          log(`  → confirm (${probeResult.analysis.losses}L margin=${probeResult.analysis.minMargin})`);
          const confirmResult = await runBatch(probe, {
            games: opts.confirmGames,
            workers: opts.workers,
            suffix: '-confirm',
          });
          state.confirmsRun = (state.confirmsRun ?? 0) + 1;
          if (confirmResult) {
            state.lastLabel = confirmResult.label;
            state.lastScore = confirmResult.summary.score;
            if (confirmResult.analysis.illegalMoveCount > 0 && !opts.skipPerft) {
              await runPerftGate();
            }
          }
        } else {
          log('  → skip confirm (clean blowout)');
        }
      }

      if (probeResult) {
        await gitCheckpoint(
          `overnight checkpoint: ${probeResult.label} ${probeResult.summary.score} WR=${(probeResult.summary.winRate * 100).toFixed(0)}%`,
        );
      }
    } catch (err) {
      log(`STEP error: ${err?.message ?? err}`);
    }

    saveCheckpoint(state);
    writeStatus(state);

    if (Date.now() >= deadline) break;
  }

  log(`Done · probes=${state.probesRun} confirms=${state.confirmsRun}`);
  stopKeepAwake();
  saveCheckpoint({ ...state, finished: true });
  writeStatus(state);
}

process.on('SIGINT', () => {
  stopKeepAwake();
  process.exit(130);
});

main().catch((err) => {
  log(`FATAL: ${err?.stack || err}`);
  stopKeepAwake();
  process.exit(2);
});
