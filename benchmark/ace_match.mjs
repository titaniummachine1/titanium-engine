#!/usr/bin/env node
/**
 * Head-to-head between titanium CLI engines (ace, ace-cat, minimax).
 *
 *   node benchmark/ace_match.mjs --white ace --black ace-cat --games 6 --time 1 --workers 3
 *
 * Colors swap every game. The engines themselves are the referee: both sides
 * emit only legal moves, so a 2-char pawn move onto the goal row ends the game.
 */

import { chooseTitaniumMove } from './lib/titanium_ai.mjs';

const MAX_PLIES = 250;

function parseArgs(argv) {
  const opts = { white: 'ace', black: 'ace-cat', games: 6, timeSec: 1, workers: 3 };
  for (let i = 2; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--white' && argv[i + 1]) opts.white = argv[++i];
    else if (arg === '--black' && argv[i + 1]) opts.black = argv[++i];
    else if (arg === '--games' && argv[i + 1]) opts.games = Number(argv[++i]);
    else if (arg === '--time' && argv[i + 1]) opts.timeSec = Number(argv[++i]);
    else if (arg === '--workers' && argv[i + 1]) opts.workers = Number(argv[++i]);
  }
  return opts;
}

function goalReached(move, ply) {
  if (move.length !== 2) return false;
  const isWhitePly = ply % 2 === 1; // white = first mover, races to row 9
  return isWhitePly ? move[1] === '9' : move[1] === '1';
}

async function playGame(gameIndex, whiteEngine, blackEngine, timeSec) {
  const history = [];
  const depths = { [whiteEngine]: [], [blackEngine]: [] };
  for (let ply = 1; ply <= MAX_PLIES; ply++) {
    const engine = ply % 2 === 1 ? whiteEngine : blackEngine;
    const { move, meta } = await chooseTitaniumMove(history, {
      engine,
      timeSec,
      log: false,
    });
    if (meta?.searchDepth != null) depths[engine].push(meta.searchDepth);
    history.push(move);
    if (goalReached(move, ply)) {
      const winner = ply % 2 === 1 ? 'white' : 'black';
      return { gameIndex, winner, plies: ply, history, depths };
    }
  }
  return { gameIndex, winner: 'draw', plies: MAX_PLIES, history, depths };
}

async function runPool(tasks, workers) {
  const results = [];
  let next = 0;
  async function worker() {
    while (next < tasks.length) {
      const idx = next++;
      results[idx] = await tasks[idx]();
    }
  }
  await Promise.all(Array.from({ length: Math.min(workers, tasks.length) }, worker));
  return results;
}

function avg(list) {
  if (!list.length) return 0;
  return list.reduce((a, b) => a + b, 0) / list.length;
}

async function main() {
  const opts = parseArgs(process.argv);
  console.log(
    `ace_match: ${opts.white} vs ${opts.black} · ${opts.games} games · ${opts.timeSec}s/move · ${opts.workers} workers`,
  );

  const tasks = [];
  for (let i = 0; i < opts.games; i++) {
    const swap = i % 2 === 1;
    const whiteEngine = swap ? opts.black : opts.white;
    const blackEngine = swap ? opts.white : opts.black;
    tasks.push(async () => {
      const r = await playGame(i + 1, whiteEngine, blackEngine, opts.timeSec);
      const winnerEngine =
        r.winner === 'draw' ? 'draw' : r.winner === 'white' ? whiteEngine : blackEngine;
      console.log(
        `  game ${r.gameIndex}: W=${whiteEngine} B=${blackEngine} → ${winnerEngine} in ${r.plies} plies` +
          ` · depth avg ${opts.white}=${avg(r.depths[opts.white] ?? []).toFixed(1)}` +
          ` ${opts.black}=${avg(r.depths[opts.black] ?? []).toFixed(1)}`,
      );
      return { ...r, whiteEngine, blackEngine, winnerEngine };
    });
  }

  const results = await runPool(tasks, opts.workers);

  let scoreA = 0;
  let scoreB = 0;
  let draws = 0;
  for (const r of results) {
    if (r.winnerEngine === 'draw') {
      draws += 1;
      scoreA += 0.5;
      scoreB += 0.5;
    } else if (r.winnerEngine === opts.white) {
      scoreA += 1;
    } else {
      scoreB += 1;
    }
  }
  console.log('');
  console.log(`RESULT ${opts.white} ${scoreA} — ${scoreB} ${opts.black} (draws ${draws})`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
