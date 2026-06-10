/**
 * Terminal game report — mirrors web gameFooter export.
 */

import { formatThinkBudget } from './bench_limits.mjs';
import { pawnSquare, wallsLeft, wallsUsed, evalPosition } from './path_eval.mjs';
import { moveLabel } from './gorisanson_moves.mjs';
import { RUST_TITANIUM_ID, GORISANSON_ID } from './engine_ids.mjs';

function formatMargin(m) {
  if (m == null || !Number.isFinite(m)) return '?';
  return m > 0 ? `+${m}` : String(m);
}

function formatScore(score) {
  if (score == null || !Number.isFinite(score)) return '?';
  if (Math.abs(score) >= 19_500) return score > 0 ? '+M' : '-M';
  return (score / 100).toFixed(2);
}

function summarizeRace(moveThinkLog) {
  let closestMargin = null;
  let maxWhiteLead = null;
  let maxBlackLead = null;
  for (const e of moveThinkLog ?? []) {
    const m = e.margin;
    if (m == null || !Number.isFinite(m)) continue;
    if (closestMargin == null || Math.abs(m) < Math.abs(closestMargin)) {
      closestMargin = m;
    }
    if (maxWhiteLead == null || m > maxWhiteLead) maxWhiteLead = m;
    if (maxBlackLead == null || m < maxBlackLead) maxBlackLead = m;
  }
  return { closestMargin, maxWhiteLead, maxBlackLead };
}

function raceVerdict(winnerPawn, loserDist, closestMargin) {
  if (winnerPawn == null) return 'in progress';
  if (loserDist != null && loserDist <= 3) return 'close — loser within 3 steps of goal';
  if (closestMargin != null && Math.abs(closestMargin) <= 2) return 'tight race — margins stayed small';
  return 'blowout — loser far from goal';
}

function engineName(id) {
  if (id === RUST_TITANIUM_ID) return 'Titanium αβ + CAT';
  if (id === GORISANSON_ID) return 'Gorisanson (JS, original)';
  return id;
}

function formatBudgetLine(tiSec, tiCap, goSec, goCap) {
  const ti = `Titanium αβ + CAT: ${tiSec}s · ≤${tiCap}`;
  const go = `Gorisanson (JS, original): ${goSec}s · ≤${goCap}`;
  return `${ti} · ${go}`;
}

function formatSimsCap(n) {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(0)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1000) return `${(n / 1000).toFixed(1)}k`;
  return String(n);
}

function formatThinkEntry(e) {
  const who = e.ply % 2 === 1 ? 'White' : 'Black';
  const parts = [`ply${e.ply} ${who} [${e.engine}] ${e.move}`];

  if (e.budgetHint) parts.push(`budget=${e.budgetHint}`);
  if (e.whiteDist != null) parts.push(`W${e.whiteDist} B${e.blackDist}`);
  if (e.margin != null) parts.push(`margin=${formatMargin(e.margin)}`);

  if (e.nodes != null) parts.push(`${e.nodes}nodes`);
  if (e.simulations != null) parts.push(`${e.simulations}sims`);
  if (e.elapsedMs != null) parts.push(`${(e.elapsedMs / 1000).toFixed(2)}s`);

  if (e.rootWinRate != null) {
    parts.push(`wr=${Math.round(e.rootWinRate * 100)}%`);
  }
  if (e.stoppedBy) parts.push(`(${e.stoppedBy})`);

  if (e.depthLog?.length) {
    const ds = e.depthLog.map((d) => `d${d.depth}=${formatScore(d.score)}`).join(' ');
    parts.push(ds);
  } else if (e.rootScore != null) {
    parts.push(`eval=${formatScore(e.rootScore)}`);
  }

  if (e.searchDepth) parts.push(`d${e.searchDepth}`);
  if (e.lmrReSearches) parts.push(`LMR↺${e.lmrReSearches}`);
  if (e.error) parts.push(`ERROR: ${e.error}`);
  if (e.fallback) parts.push(`FALLBACK→${e.fallback}`);

  return parts.join(' ');
}

export function buildGameReport({
  game,
  winnerPawn,
  whiteId,
  blackId,
  replayCode,
  moveThinkLog,
  errors,
  tiBudget,
  goBudget,
  gameIndex,
}) {
  const pos = evalPosition(game);
  const plies = moveThinkLog?.length ?? 0;
  const { closestMargin, maxWhiteLead, maxBlackLead } = summarizeRace(moveThinkLog);
  const loserDist = winnerPawn === 0 ? pos.blackDist : winnerPawn === 1 ? pos.whiteDist : null;

  const tiCap = formatSimsCap(tiBudget?.maxSimulations ?? 0);
  const goCap = formatSimsCap(goBudget?.maxSimulations ?? 0);
  const budgetHint = formatBudgetLine(tiBudget?.timeSec ?? 10, tiCap, goBudget?.timeSec ?? 10, goCap);

  const lines = ['=== Quoridor game report ===', ''];

  if (winnerPawn != null) {
    const winner = winnerPawn === 0 ? 'White' : 'Black';
    lines.push(`Result: ${winner} wins · ${plies} plies`);
  } else {
    lines.push(`Result: draw or unfinished · ${plies} plies`);
  }

  lines.push(`White: ${engineName(whiteId)}`, `Black: ${engineName(blackId)}`);
  lines.push(`Budget: ${budgetHint}`);
  if (gameIndex != null) lines.push(`Game: ${gameIndex}`);
  lines.push('');

  lines.push(
    `Final position: White=${pawnSquare(game, 0)} Black=${pawnSquare(game, 1)}`,
    `Path distance: W=${pos.whiteDist} B=${pos.blackDist} · margin=${formatMargin(pos.margin)} (positive = White ahead)`,
    `Walls used: White=${wallsUsed(game, 0)} Black=${wallsUsed(game, 1)} · left W=${wallsLeft(game, 0)} B=${wallsLeft(game, 1)}`,
  );

  if (closestMargin != null) {
    lines.push(
      `Race swing: closest margin=${formatMargin(closestMargin)} · best White lead=${formatMargin(maxWhiteLead)} · best Black lead=${formatMargin(maxBlackLead)}`,
    );
  }

  lines.push(`Verdict: ${raceVerdict(winnerPawn, loserDist, closestMargin)}`);

  if (replayCode) {
    lines.push('', `Replay: ${replayCode}`);
  }

  if (errors?.length) {
    lines.push('', '--- Engine errors ---');
    for (const err of errors) {
      lines.push(
        `ply${err.ply} ${err.side} [${err.engine}]: ${err.move} — ${err.reason}${err.fallback ? ` → fallback ${err.fallback}` : ''}`,
      );
    }
  }

  if (moveThinkLog?.length) {
    lines.push('', '--- Think chain ---');
    for (const e of moveThinkLog) {
      lines.push(formatThinkEntry(e));
    }
  }

  return lines.join('\n');
}

export function formatMoveLabel(move) {
  return moveLabel(move);
}
