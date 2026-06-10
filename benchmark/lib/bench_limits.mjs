/**
 * Shared benchmark think budget — both engines stop on whichever limit hits first.
 */

export const BENCH_TIME_SEC = 10;
export const BENCH_MAX_SIMULATIONS = 10_000_000_000;
export const BENCH_TIME_MS = BENCH_TIME_SEC * 1000;
/** Gorisanson web default visit cap (66k rollouts). */
export const GORISANSON_MAX_VISITS = 66_000;
/** Titanium node budget — time usually stops first. */
export const TITANIUM_MAX_NODES = 10_000_000_000;

export function resolveThinkBudget(options = {}, playerConfig = {}) {
  const timeSec = playerConfig.timeSec ?? options.timeSec ?? BENCH_TIME_SEC;
  const timeMs = playerConfig.timeMs ?? timeSec * 1000;
  const maxSimulations =
    playerConfig.maxSimulations ??
    playerConfig.simulations ??
    options.maxSimulations ??
    BENCH_MAX_SIMULATIONS;
  return { timeSec, timeMs, maxSimulations };
}

export function formatThinkBudget(budget) {
  const sims =
    budget.maxSimulations >= 1_000_000_000
      ? `${(budget.maxSimulations / 1_000_000_000).toFixed(0)}B`
      : budget.maxSimulations.toLocaleString();
  return `${budget.timeSec}s / ${sims} sims cap`;
}
