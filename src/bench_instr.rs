//! Benchmark-only counters and phase timers (`bench-instrument` feature).

use std::cell::Cell;
use std::time::{Duration, Instant};

#[derive(Clone, Copy, Default, Debug)]
pub struct OpStat {
    pub calls: u64,
    pub ns: u128,
}

impl OpStat {
    #[inline]
    pub fn record(&mut self, dt: Duration) {
        self.calls += 1;
        self.ns += dt.as_nanos();
    }

    pub fn ns_per_call(&self) -> f64 {
        if self.calls == 0 {
            0.0
        } else {
            self.ns as f64 / self.calls as f64
        }
    }
}

/// Per call-site refresh_dist accounting. Index == site id.
#[derive(Clone, Copy, Default, Debug)]
pub struct RefreshSiteStat {
    pub calls: u64,
    pub ns: u128,
    pub cheap: u64,
    pub incr: u64,
    pub full: u64,
    pub reflood: u64,
}

pub const REFRESH_SITE_UNKNOWN: u8 = 0;
pub const REFRESH_SITE_AB: u8 = 1;
pub const REFRESH_SITE_QSEARCH_UNMAKE: u8 = 2;
pub const REFRESH_SITE_QSEARCH_REMAKE: u8 = 3;
pub const REFRESH_SITE_CAT_PATH_LMR: u8 = 4;
pub const REFRESH_SITE_ROOT_DEF_INIT: u8 = 5;
pub const REFRESH_SITE_ROOT_DEF_BEFORE: u8 = 6;
pub const REFRESH_SITE_ROOT_DEF_AFTER: u8 = 7;
pub const REFRESH_SITE_ROOT_WIN_BEFORE: u8 = 8;
pub const REFRESH_SITE_ROOT_WIN_AFTER: u8 = 9;
pub const REFRESH_SITE_RACE_PICK: u8 = 10;
pub const REFRESH_SITE_PROGRESS: u8 = 11;
pub const REFRESH_SITE_THINK_GUARD: u8 = 12;
pub const REFRESH_SITE_THINK_FINAL: u8 = 13;
pub const REFRESH_SITE_EVAL_POSITION: u8 = 14;
pub const REFRESH_SITE_LAZY_ROOT: u8 = 15;
pub const REFRESH_SITE_EVAL_DUMP: u8 = 16;
pub const REFRESH_SITE_EVAL_PARITY: u8 = 17;
pub const REFRESH_SITE_OPENING: u8 = 18;
pub const REFRESH_SITE_CHEAP_CERT: u8 = 19;
pub const REFRESH_SITE_RACE_ROOT: u8 = 20;
pub const REFRESH_SITE_COUNT: usize = 21;

pub const REFRESH_SITE_META: [(&str, u32, &str); REFRESH_SITE_COUNT] = [
    ("unknown", 0, "?"),
    ("ab", 5164, "ab"),
    ("qsearch_unmake", 5078, "q_search_wall_dist_changed"),
    ("qsearch_remake", 5082, "q_search_wall_dist_changed"),
    ("cat_path_lmr", 5598, "ab"),
    ("root_def_init", 6094, "root_defense_verify"),
    ("root_def_before", 6108, "root_defense_verify"),
    ("root_def_after", 6125, "root_defense_verify"),
    ("root_win_before", 6263, "root_clean_win_verify"),
    ("root_win_after", 6280, "root_clean_win_verify"),
    ("race_pick", 4688, "race_root_pick"),
    ("progress", 3502, "emit_stream_progress"),
    ("think_guard", 7295, "think_search"),
    ("think_final", 7314, "think_search"),
    ("eval_position", 2871, "eval_position"),
    ("lazy_root", 2943, "ordered_root_moves_snapshot"),
    ("eval_dump", 3142, "eval_dump_json"),
    ("eval_parity", 3284, "eval_parity_trace_json"),
    ("opening", 6352, "think"),
    ("cheap_cert", 6413, "think"),
    ("race_root", 6508, "think"),
];

#[derive(Clone, Default, Debug)]
pub struct BenchInstr {
    pub search_nodes: u64,
    pub stop_reason: &'static str,
    pub evaluate: OpStat,
    pub eval_race_bound: OpStat,
    pub race_gate_cached: OpStat,
    pub race_winner_table: OpStat,
    pub eval_route_features: OpStat,
    pub eval_nnue_prep: OpStat,
    pub eval_nnue_infer: OpStat,
    pub eval_wall_cross: OpStat,
    pub eval_legal_wall_count: OpStat,
    pub eval_misc_scalar: OpStat,
    pub refresh_dist: OpStat,
    pub shortest_path: OpStat,
    pub dir_masks_from_ace: OpStat,
    pub flood_bit_sq: OpStat,
    pub flood_bit_index: OpStat,
    pub flood_sq_from_bit: OpStat,
    pub flood_scatter: OpStat,
    pub unpack_square: OpStat,
    pub wall_crossing_count: OpStat,
    pub collect_wall_orientation: OpStat,
    pub wall_legality: OpStat,
    pub wall_proof_skip: OpStat,
    pub wall_seal_skip: OpStat,
    pub can_step: OpStat,
    pub gen_moves: OpStat,
    pub tt_probe: OpStat,
    pub tt_hit: OpStat,
    pub tt_cutoff: OpStat,
    pub tt_store: OpStat,
    pub make_move: OpStat,
    pub unmake_move: OpStat,
    pub nnue_full_refresh: OpStat,
    pub nnue_incr_update: OpStat,
    pub eval_cache_hit: OpStat,
    pub eval_cache_miss: OpStat,
    pub refresh_cheap: OpStat,
    pub refresh_incr: OpStat,
    pub refresh_full: OpStat,
    pub dist_lru_hit: OpStat,
    pub dist_lru_miss: OpStat,
    pub dist_reflood: OpStat,
    pub eval_width_opp: OpStat,
    pub eval_cat_heat: OpStat,
    pub eval_tail: OpStat,
    pub mat_layers: OpStat,
    pub refresh_site_stats: [RefreshSiteStat; REFRESH_SITE_COUNT],
    pub refresh_ab_skipped: u64,
    /// Child `ab(ply+1)` entered immediately after `cat_path_lmr` refresh.
    pub cat_child_ab_entries: u64,
    /// At that entry, `cached_stamp` + dist keys already valid (duplicate refresh).
    pub cat_child_ab_dup_valid: u64,
    /// Control path: duplicate `refresh_dist` still invoked (cheap inner path).
    pub cat_child_ab_dup_refresh: u64,
    /// Reuse path: child `ab()` skipped refresh when state provably valid.
    pub cat_child_ab_dup_avoided: u64,
    /// CAT-refreshed child returned from TT cutoff before `evaluate()`.
    pub cat_child_ab_tt_cutoff_before_eval: u64,
    /// CAT `refresh_dist` incr path: wall did not cut either shortest-path edge.
    pub cat_incr_no_edge_cut: u64,
    /// CAT path-LMR skipped refresh because wall cut neither shortest-path edge.
    pub cat_no_edge_skip: u64,
    /// Shared edge-cut probe invocations at the CAT path-LMR site.
    pub cat_edge_test_calls: u64,
    search_t0: Option<Instant>,
    measured_ns: u128,
}

thread_local! {
    static BENCH: std::cell::RefCell<BenchInstr> = std::cell::RefCell::new(BenchInstr::default());
    static ACTIVE_REFRESH_SITE: Cell<u8> = Cell::new(REFRESH_SITE_UNKNOWN);
}

impl BenchInstr {
    pub fn begin_search(&mut self) {
        *self = Self {
            search_t0: Some(Instant::now()),
            ..Default::default()
        };
    }

    pub fn end_search(&mut self, nodes: u64) {
        self.search_nodes = nodes;
        if let Some(t0) = self.search_t0.take() {
            self.measured_ns = t0.elapsed().as_nanos();
        }
    }

    pub fn set_stop_reason(&mut self, reason: &'static str) {
        self.stop_reason = reason;
    }

    pub fn refresh_site_calls_total(&self) -> u64 {
        self.refresh_site_stats.iter().map(|s| s.calls).sum()
    }

    pub fn to_json(&self) -> String {
        fn row(name: &str, s: &OpStat, nodes: u64, total_ns: u128) -> String {
            let cpn = if nodes == 0 {
                0.0
            } else {
                s.calls as f64 / nodes as f64
            };
            let pct = if total_ns == 0 {
                0.0
            } else {
                100.0 * s.ns as f64 / total_ns as f64
            };
            format!(
                r#"{{"op":"{name}","calls":{calls},"calls_per_node":{cpn:.4},"total_ns":{ns},"ns_per_call":{npc:.1},"pct_measured":{pct:.2}}}"#,
                name = name,
                calls = s.calls,
                cpn = cpn,
                ns = s.ns,
                npc = s.ns_per_call(),
                pct = pct,
            )
        }
        let nodes = self.search_nodes;
        let total_ns = self.measured_ns;
        let ops: [(&str, &OpStat); 45] = [
            ("evaluate", &self.evaluate),
            ("eval_race_bound", &self.eval_race_bound),
            ("race_gate_cached", &self.race_gate_cached),
            ("race_winner_table", &self.race_winner_table),
            ("eval_route_features", &self.eval_route_features),
            ("eval_nnue_prep", &self.eval_nnue_prep),
            ("eval_nnue_infer", &self.eval_nnue_infer),
            ("eval_wall_cross", &self.eval_wall_cross),
            ("eval_legal_wall_count", &self.eval_legal_wall_count),
            ("eval_misc_scalar", &self.eval_misc_scalar),
            ("refresh_dist", &self.refresh_dist),
            ("shortest_path", &self.shortest_path),
            ("dir_masks_from_ace", &self.dir_masks_from_ace),
            ("flood_bit_sq", &self.flood_bit_sq),
            ("flood_bit_index", &self.flood_bit_index),
            ("flood_sq_from_bit", &self.flood_sq_from_bit),
            ("flood_scatter", &self.flood_scatter),
            ("unpack_square", &self.unpack_square),
            ("wall_crossing_count", &self.wall_crossing_count),
            ("collect_wall_orientation", &self.collect_wall_orientation),
            ("wall_legality", &self.wall_legality),
            ("wall_proof_skip", &self.wall_proof_skip),
            ("wall_seal_skip", &self.wall_seal_skip),
            ("can_step", &self.can_step),
            ("gen_moves", &self.gen_moves),
            ("tt_probe", &self.tt_probe),
            ("tt_hit", &self.tt_hit),
            ("tt_cutoff", &self.tt_cutoff),
            ("tt_store", &self.tt_store),
            ("make_move", &self.make_move),
            ("unmake_move", &self.unmake_move),
            ("nnue_full_refresh", &self.nnue_full_refresh),
            ("nnue_incr_update", &self.nnue_incr_update),
            ("eval_cache_hit", &self.eval_cache_hit),
            ("eval_cache_miss", &self.eval_cache_miss),
            ("refresh_cheap", &self.refresh_cheap),
            ("refresh_incr", &self.refresh_incr),
            ("refresh_full", &self.refresh_full),
            ("dist_lru_hit", &self.dist_lru_hit),
            ("dist_lru_miss", &self.dist_lru_miss),
            ("dist_reflood", &self.dist_reflood),
            ("eval_width_opp", &self.eval_width_opp),
            ("eval_cat_heat", &self.eval_cat_heat),
            ("eval_tail", &self.eval_tail),
            ("mat_layers", &self.mat_layers),
        ];
        let parts: Vec<String> = ops
            .iter()
            .map(|(name, s)| row(name, s, nodes, total_ns))
            .collect();

        let site_rows: Vec<String> = (0..REFRESH_SITE_COUNT)
            .map(|i| {
                let s = &self.refresh_site_stats[i];
                let (label, line, func) = REFRESH_SITE_META[i];
                let cpn = if nodes == 0 {
                    0.0
                } else {
                    s.calls as f64 / nodes as f64
                };
                format!(
                    r#"{{"site_id":{i},"label":"{label}","line":{line},"function":"{func}","calls":{calls},"calls_per_node":{cpn:.6},"total_ns":{ns},"ns_per_call":{npc:.1},"cheap":{cheap},"incr":{incr},"full":{full},"reflood":{reflood}}}"#,
                    i = i,
                    label = label,
                    line = line,
                    func = func,
                    calls = s.calls,
                    cpn = cpn,
                    ns = s.ns,
                    npc = if s.calls == 0 { 0.0 } else { s.ns as f64 / s.calls as f64 },
                    cheap = s.cheap,
                    incr = s.incr,
                    full = s.full,
                    reflood = s.reflood,
                )
            })
            .collect();

        format!(
            r#"{{"search_nodes":{nodes},"measured_ns":{total_ns},"stop_reason":"{}","refresh_dist_calls":{},"refresh_site_calls_sum":{},"refresh_ab_skipped":{},"cat_path_lmr":{{"child_ab_entries":{},"dup_valid":{},"dup_refresh":{},"dup_avoided":{},"tt_cutoff_before_eval":{},"incr_no_edge_cut":{},"no_edge_skip":{},"edge_test_calls":{}}},"ops":[{}],"refresh_sites":[{}]}}"#,
            self.stop_reason,
            self.refresh_dist.calls,
            self.refresh_site_calls_total(),
            self.refresh_ab_skipped,
            self.cat_child_ab_entries,
            self.cat_child_ab_dup_valid,
            self.cat_child_ab_dup_refresh,
            self.cat_child_ab_dup_avoided,
            self.cat_child_ab_tt_cutoff_before_eval,
            self.cat_incr_no_edge_cut,
            self.cat_no_edge_skip,
            self.cat_edge_test_calls,
            parts.join(","),
            site_rows.join(","),
        )
    }
}

pub fn with_bench<F, R>(f: F) -> R
where
    F: FnOnce(&mut BenchInstr) -> R,
{
    BENCH.with(|c| f(&mut c.borrow_mut()))
}

#[inline(always)]
pub fn record<F, R>(pick: fn(&mut BenchInstr) -> &mut OpStat, body: F) -> R
where
    F: FnOnce() -> R,
{
    #[cfg(feature = "bench-instrument")]
    {
        let t0 = Instant::now();
        let out = body();
        with_bench(|b| pick(b).record(t0.elapsed()));
        out
    }
    #[cfg(not(feature = "bench-instrument"))]
    body()
}

#[inline(always)]
pub fn count<F, R>(pick: fn(&mut BenchInstr) -> &mut OpStat, body: F) -> R
where
    F: FnOnce() -> R,
{
    #[cfg(feature = "bench-instrument")]
    {
        let out = body();
        with_bench(|b| pick(b).calls += 1);
        out
    }
    #[cfg(not(feature = "bench-instrument"))]
    body()
}

#[inline(always)]
pub fn bump_u64(pick: fn(&mut BenchInstr) -> &mut u64) {
    #[cfg(feature = "bench-instrument")]
    with_bench(|b| {
        *pick(b) += 1;
    });
}

#[inline(always)]
pub fn bump(pick: fn(&mut BenchInstr) -> &mut OpStat) {
    #[cfg(feature = "bench-instrument")]
    with_bench(|b| {
        pick(b).calls += 1;
    });
}

#[inline(always)]
pub fn active_refresh_site() -> u8 {
    #[cfg(feature = "bench-instrument")]
    {
        ACTIVE_REFRESH_SITE.get()
    }
    #[cfg(not(feature = "bench-instrument"))]
    {
        REFRESH_SITE_UNKNOWN
    }
}

#[inline(always)]
pub fn refresh_site_call_start(site: u8) {
    #[cfg(feature = "bench-instrument")]
    {
        let idx = site as usize;
        ACTIVE_REFRESH_SITE.set(site);
        if idx < REFRESH_SITE_COUNT {
            with_bench(|b| b.refresh_site_stats[idx].calls += 1);
        }
    }
    #[cfg(not(feature = "bench-instrument"))]
    let _ = site;
}

#[inline(always)]
pub fn refresh_site_call_end(site: u8, dt: Duration) {
    #[cfg(feature = "bench-instrument")]
    {
        let idx = site as usize;
        let ns = dt.as_nanos();
        if idx < REFRESH_SITE_COUNT {
            with_bench(|b| {
                b.refresh_site_stats[idx].ns += ns;
                b.refresh_dist.record(dt);
            });
        } else {
            with_bench(|b| b.refresh_dist.record(dt));
        }
        ACTIVE_REFRESH_SITE.set(REFRESH_SITE_UNKNOWN);
    }
    #[cfg(not(feature = "bench-instrument"))]
    {
        let _ = (site, dt);
    }
}

#[inline(always)]
pub fn refresh_site_path(path: u8) {
    #[cfg(feature = "bench-instrument")]
    {
        let site = ACTIVE_REFRESH_SITE.get() as usize;
        if site < REFRESH_SITE_COUNT {
            with_bench(|b| match path {
                0 => {
                    b.refresh_cheap.calls += 1;
                    b.refresh_site_stats[site].cheap += 1;
                }
                1 => {
                    b.refresh_incr.calls += 1;
                    b.refresh_site_stats[site].incr += 1;
                }
                _ => {
                    b.refresh_full.calls += 1;
                    b.refresh_site_stats[site].full += 1;
                }
            });
        }
    }
    #[cfg(not(feature = "bench-instrument"))]
    let _ = path;
}

#[inline(always)]
pub fn refresh_site_reflood() {
    #[cfg(feature = "bench-instrument")]
    {
        let site = ACTIVE_REFRESH_SITE.get() as usize;
        bump(|b| &mut b.dist_reflood);
        if site < REFRESH_SITE_COUNT {
            with_bench(|b| b.refresh_site_stats[site].reflood += 1);
        }
    }
}

#[inline(always)]
pub fn bump_ab_refresh_skipped() {
    #[cfg(feature = "bench-instrument")]
    with_bench(|b| b.refresh_ab_skipped += 1);
}

pub fn begin_search() {
    #[cfg(feature = "bench-instrument")]
    with_bench(|b| b.begin_search());
}

pub fn end_search(nodes: u64) {
    #[cfg(feature = "bench-instrument")]
    with_bench(|b| b.end_search(nodes));
}

pub fn set_stop_reason(reason: &'static str) {
    #[cfg(feature = "bench-instrument")]
    with_bench(|b| b.set_stop_reason(reason));
}

pub fn take_json_report() -> Option<String> {
    #[cfg(feature = "bench-instrument")]
    {
        return Some(with_bench(|b| b.to_json()));
    }
    #[cfg(not(feature = "bench-instrument"))]
    None
}

pub struct OpTimer {
    #[cfg(feature = "bench-instrument")]
    pick: fn(&mut BenchInstr) -> &mut OpStat,
    #[cfg(feature = "bench-instrument")]
    t0: Instant,
}

impl OpTimer {
    #[inline(always)]
    pub fn start(pick: fn(&mut BenchInstr) -> &mut OpStat) -> Self {
        #[cfg(feature = "bench-instrument")]
        {
            Self {
                pick,
                t0: Instant::now(),
            }
        }
        #[cfg(not(feature = "bench-instrument"))]
        {
            let _ = pick;
            Self {}
        }
    }
}

impl Drop for OpTimer {
    fn drop(&mut self) {
        #[cfg(feature = "bench-instrument")]
        {
            let dt = self.t0.elapsed();
            with_bench(|b| (self.pick)(b).record(dt));
        }
    }
}
