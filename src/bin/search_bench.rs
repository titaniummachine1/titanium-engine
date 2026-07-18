//! Authoritative native Titanium search benchmark — JSON on stdout only.
//!
//! Build (baseline / flamegraph):
//!   $env:RUSTFLAGS="-C target-cpu=native -C force-frame-pointers=yes"
//!   cargo build --profile profiling -p titanium --bin search_bench --manifest-path engine\Cargo.toml
//!
//! Instrumented build:
//!   cargo build --profile profiling -p titanium --bin search_bench --features bench-instrument --manifest-path engine\Cargo.toml

use std::fs;
use std::hint::black_box;
use std::process::Command;
use std::time::Instant;

use sha2::{Digest, Sha256};
use titanium::algebraic_to_move_id;
use titanium::bench_instr;
use titanium::cat::build::build_impact_heatmap;
use titanium::cat::CorridorAttention;
use titanium::core::board::Board;
use titanium::movegen::prewarm;
use titanium::titanium::lazy_seal::{dump_lazy_seal_stats, reset_lazy_seal_stats};
use titanium::titanium::net::live_weights_sha256;
use titanium::titanium::session::apply_session_experiment_flags;
use titanium::titanium::{move_id_to_algebraic, GameState, ThinkResult, TitaniumSearch};

/// Reported + logged engine label. The actual search mode is selected in
/// `fresh_search` (TITANIUM_BENCH_V16=1 → v17 CAT-LMR), so the label must
/// follow the same env var or profiles get misattributed.
fn engine_mode() -> &'static str {
    let base = if let Ok(flag) = std::env::var("TITANIUM_BENCH_ENGINE") {
        flag
    } else if std::env::var("TITANIUM_BENCH_V16").as_deref() == Ok("1") {
        "titanium-v17".into()
    } else {
        "titanium-v17".into()
    };
    let lazy = std::env::var("TITANIUM_BENCH_LAZY_WALLS").as_deref() == Ok("1");
    let seal_mode = std::env::var("TITANIUM_LAZY_SEAL_MODE").unwrap_or_default();
    if lazy {
        let seal = if seal_mode.is_empty() {
            "deferred"
        } else {
            &seal_mode
        };
        Box::leak(format!("{base}-lazy-{seal}").into_boxed_str())
    } else {
        Box::leak(base.into_boxed_str())
    }
}
const TT_BITS: usize = 20;
const MAX_DEPTH: i32 = 30;

fn hex32(bytes: &[u8; 32]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn json_str(s: &str) -> String {
    let esc = s.replace('\\', "\\\\").replace('"', "\\\"");
    format!("\"{esc}\"")
}

fn binary_sha256() -> String {
    let exe = std::env::current_exe().expect("exe path");
    hex32(&Sha256::digest(fs::read(&exe).expect("read binary")).into())
}

fn git_commit() -> String {
    Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".into())
}

fn cargo_features() -> String {
    let mut feats = Vec::new();
    if cfg!(feature = "parallel") {
        feats.push("parallel");
    }
    if cfg!(feature = "bench-instrument") {
        feats.push("bench-instrument");
    }
    if cfg!(feature = "eval_cache_baseline") {
        feats.push("eval_cache_baseline");
    }
    if cfg!(feature = "dist_layers_full81") {
        feats.push("dist_layers_full81");
    }
    if cfg!(feature = "dist_layers_inline12") {
        feats.push("dist_layers_inline12");
    }
    if cfg!(feature = "embed-tables") {
        feats.push("embed-tables");
    }
    feats.join(",")
}

fn median_u64(v: &[u64]) -> u64 {
    let mut s = v.to_vec();
    s.sort_unstable();
    s[s.len() / 2]
}

fn median_f64(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    s[s.len() / 2]
}

fn tt_mb(bits: usize) -> f64 {
    let entries = 1usize << bits;
    entries as f64 * 25.0 / (1024.0 * 1024.0)
}

fn position_moves(name: &str) -> &'static [&'static str] {
    match name {
        "startpos" => &[],
        "c3h-midgame" => &["e2", "e8", "e3", "e7", "e4", "e6", "c3h"],
        "wall-maze" => &[
            "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "e4h", "d4", "c4h", "e5v", "a5h", "h8h",
            "d6", "b5v", "f3v", "e7v", "c3h", "d7h", "b2v", "h6h",
        ],
        "low-wall" => &["e2", "e8", "e3", "e7", "d4h"],
        "endgame-c5" => &[
            "e2", "e8", "e3", "e7", "e4", "e6", "f3h", "c3h", "d1h", "d4v", "e2v", "f6h", "d6v",
            "d8v", "h3h", "e7h", "e4h", "h6h", "b1h", "f6", "f7v", "f5", "e3", "g5", "g4h", "h5",
            "d3", "i5", "c3", "i4", "b3", "h4", "b4", "g4", "b5", "f4", "b6", "b6h", "a2v", "e4",
            "b5", "e3",
        ],
        "dense-maze" => &[
            "e2", "e8", "e3", "e7", "e4", "e6", "e3h", "f6h", "c3h", "d6h", "a3h", "h6h", "d4",
            "f6", "f5v", "c4v", "d5", "d5h", "g3h", "b6h", "e4h", "f5", "b5h",
        ],
        other => panic!(
            "unknown position {other}; use startpos|c3h-midgame|wall-maze|low-wall|endgame-c5|dense-maze"
        ),
    }
}

fn load_position(name: &str) -> GameState {
    let mut g = GameState::new();
    for mv in position_moves(name) {
        g.make_move(algebraic_to_move_id(mv));
    }
    g
}

fn load_board(name: &str, ply: Option<usize>) -> Board {
    let moves = position_moves(name);
    let limit = ply.unwrap_or(moves.len()).min(moves.len());
    let mut board = Board::new();
    for mv in &moves[..limit] {
        board.apply_algebraic(mv);
    }
    board
}

fn load_position_from_moves(moves: &str) -> GameState {
    let mut g = GameState::new();
    for mv in moves.split_whitespace() {
        g.make_move(algebraic_to_move_id(mv));
    }
    g
}

fn fresh_search(position: &str, moves: Option<&str>) -> Box<TitaniumSearch> {
    let g = match moves {
        Some(raw) if !raw.trim().is_empty() => load_position_from_moves(raw),
        _ => load_position(position),
    };
    let lazy_walls = std::env::var("TITANIUM_BENCH_LAZY_WALLS").as_deref() == Ok("1");
    if let Ok(flag) = std::env::var("TITANIUM_BENCH_ENGINE") {
        let mut search = if lazy_walls {
            TitaniumSearch::grafted_v17_lazy_walls_for_bench(g, Some(TT_BITS), 1000)
        } else {
            TitaniumSearch::grafted_v17_with_ceiling(g, Some(TT_BITS), 1000)
        };
        apply_session_experiment_flags(search.as_mut(), &flag);
        return search;
    }
    // TITANIUM_BENCH_V16=1 profiles the v17 CAT-LMR engine (default ceiling 1000)
    // so we can A/B the CAT cost vs the v15 baseline on identical positions.
    if std::env::var("TITANIUM_BENCH_V16").as_deref() == Ok("1") {
        if lazy_walls {
            TitaniumSearch::grafted_v17_lazy_walls_for_bench(g, Some(TT_BITS), 1000)
        } else {
            TitaniumSearch::grafted_v17_with_ceiling(g, Some(TT_BITS), 1000)
        }
    } else if lazy_walls {
        TitaniumSearch::grafted_lazy_walls_for_bench(g, Some(TT_BITS))
    } else {
        TitaniumSearch::grafted(g, Some(TT_BITS))
    }
}

fn run_think(
    search: &mut TitaniumSearch,
    time_ms: u64,
    max_depth: i32,
    full: bool,
    log: bool,
    threads: usize,
) -> ThinkResult {
    search.think_with_threads(time_ms, max_depth, full, log, engine_mode(), threads)
}

fn common_meta(position: &str, threads: usize) -> String {
    format!(
        ",\"commit\":{commit},\"binary_sha256\":{bin},\"weights_sha256\":{wt},\"position\":{pos},\"engine_mode\":{em},\"threads\":{th},\"tt_bits\":{ttb},\"tt_mb\":{ttm:.2},\"rustc\":{rustc},\"bmi2_build\":{bmi2},\"cargo_features\":{cf},\"max_depth\":{md},\"full_search\":{{full}}",
        commit = json_str(&git_commit()),
        bin = json_str(&binary_sha256()),
        wt = json_str(&hex32(&live_weights_sha256())),
        pos = json_str(position),
        em = json_str(engine_mode()),
        th = threads,
        ttb = TT_BITS,
        ttm = tt_mb(TT_BITS),
        rustc = json_str(&rustc_version()),
        bmi2 = if cfg!(all(target_arch = "x86_64", target_feature = "bmi2")) {
            "true"
        } else {
            "false"
        },
        cf = json_str(&cargo_features()),
        md = MAX_DEPTH,
    )
}

fn emit_result(
    bench_type: &str,
    position: &str,
    result: &ThinkResult,
    wall_ms: u64,
    full: bool,
    log: bool,
    threads: usize,
    extra: &str,
) {
    let nps = if wall_ms > 0 {
        (result.nodes as f64) * 1000.0 / wall_ms as f64
    } else {
        0.0
    };
    let meta =
        common_meta(position, threads).replace("{full}", if full { "true" } else { "false" });
    let helper_nodes = result
        .helper_nodes
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let helper_depths = result
        .helper_completed_depths
        .iter()
        .map(|d| d.to_string())
        .collect::<Vec<_>>()
        .join(",");
    let root_widths = result
        .root_widths
        .iter()
        .map(|p| {
            format!(
                "{{\"worker_id\":{},\"root_value_threshold_pct\":{},\"root_moves_before_filter\":{},\"root_moves_retained\":{},\"root_moves_retained_pct\":{:.1}}}",
                p.worker_id,
                p.root_value_threshold_pct,
                p.root_moves_before_filter,
                p.root_move_count,
                p.root_moves_retained_pct(),
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    let root_visits = result
        .root_visits
        .iter()
        .map(|visits| {
            format!(
                "[{}]",
                visits
                    .iter()
                    .map(|idx| idx.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )
        })
        .collect::<Vec<_>>()
        .join(",");
    println!(
        "{{\"bench_type\":{bt}{meta},\"elapsed_ms\":{wall},\"nodes\":{nodes},\"nps\":{nps:.0},\"depth\":{depth},\"move\":{mv},\"score\":{score},\"stop_reason\":{sr},\"node_source\":\"search.nodes\",\"main_thread_nodes\":{main_nodes},\"helper_nodes\":[{helper_nodes}],\"total_nodes\":{total_nodes},\"main_completed_depth\":{main_depth},\"helper_completed_depths\":[{helper_depths}],\"root_widths\":[{root_widths}],\"root_visits\":[{root_visits}],\"log_during_search\":{log}{extra}}}",
        bt = json_str(bench_type),
        wall = wall_ms,
        nodes = result.nodes,
        nps = nps,
        depth = result.depth,
        mv = json_str(&move_id_to_algebraic(result.mv)),
        score = result.score,
        sr = json_str(result.stop_reason),
        main_nodes = result.main_thread_nodes,
        total_nodes = result.total_nodes,
        main_depth = result.main_completed_depth,
        helper_nodes = helper_nodes,
        helper_depths = helper_depths,
        root_widths = root_widths,
        root_visits = root_visits,
        log = if log { "true" } else { "false" },
        extra = extra,
    );
}

fn bench_time(sec: u64, runs: usize, position: &str, full: bool, log: bool, threads: usize) {
    reset_lazy_seal_stats();
    let time_ms = sec * 1000;
    let g = load_position(position);
    let mut search = fresh_search(position, None);
    let _ = run_think(&mut search, time_ms, MAX_DEPTH, full, false, threads);
    search.set_position(g);

    let mut nodes = Vec::new();
    let mut nps = Vec::new();
    let mut walls = Vec::new();
    let mut depths = Vec::new();
    let mut results = Vec::new();
    let mut run_json = String::new();

    for i in 0..runs {
        search.set_position(load_position(position));
        let t0 = Instant::now();
        let r = run_think(&mut search, time_ms, MAX_DEPTH, full, log, threads);
        let wall_ms = t0.elapsed().as_millis() as u64;
        let n = if wall_ms > 0 {
            (r.nodes as f64) * 1000.0 / wall_ms as f64
        } else {
            0.0
        };
        nodes.push(r.nodes);
        nps.push(n);
        walls.push(wall_ms);
        depths.push(r.depth as u64);
        results.push(r.clone());
        if i > 0 {
            run_json.push(',');
        }
        run_json.push_str(&format!(
            "{{\"run\":{},\"nodes\":{},\"nps\":{:.0},\"depth\":{},\"move\":{},\"score\":{},\"elapsed_ms\":{},\"stop_reason\":{}}}",
            i + 1,
            r.nodes,
            n,
            r.depth,
            json_str(&move_id_to_algebraic(r.mv)),
            r.score,
            wall_ms,
            json_str(r.stop_reason),
        ));
    }

    let median_nodes = median_u64(&nodes);
    let median_idx = nodes.iter().position(|&n| n == median_nodes).unwrap_or(0);
    let mut rep = results[median_idx].clone();
    rep.ms = median_u64(&walls);
    let extra = format!(
        ",\"runs\":{runs},\"time_sec\":{sec},\"median_nodes\":{},\"median_nps\":{:.0},\"median_elapsed_ms\":{},\"median_depth\":{},\"runs_detail\":[{}]",
        median_nodes,
        median_f64(&nps),
        median_u64(&walls),
        median_u64(&depths),
        run_json,
    );
    emit_result(
        "fixed_time",
        position,
        &rep,
        median_u64(&walls),
        full,
        log,
        threads,
        &extra,
    );
    eprintln!("{}", dump_lazy_seal_stats());
}

fn bench_depth(target_depth: i32, position: &str, full: bool, threads: usize) {
    reset_lazy_seal_stats();
    let mut search = fresh_search(position, None);
    let _ = run_think(
        &mut search,
        60_000,
        target_depth.saturating_sub(1).max(1),
        full,
        false,
        threads,
    );
    search.set_position(load_position(position));
    let t0 = Instant::now();
    let r = run_think(&mut search, 600_000, target_depth, full, false, threads);
    let wall_ms = t0.elapsed().as_millis() as u64;
    let extra = format!(",\"target_depth\":{target_depth}");
    emit_result(
        "fixed_depth",
        position,
        &r,
        wall_ms,
        full,
        false,
        threads,
        &extra,
    );
    if let Some(instr) = bench_instr::take_json_report() {
        println!("{instr}");
    }
    eprintln!("{}", dump_lazy_seal_stats());
}

fn bench_profile(sec: u64, position: &str, moves: Option<&str>, full: bool, threads: usize) {
    let mut search = fresh_search(position, moves);
    eprintln!(
        "profile: {}s position={} full={} tt_bits={TT_BITS}",
        sec, position, full
    );
    let t0 = Instant::now();
    let r = run_think(&mut search, sec * 1000, MAX_DEPTH, full, false, threads);
    let wall_ms = t0.elapsed().as_millis() as u64;
    let nps = if wall_ms > 0 {
        r.nodes as f64 * 1000.0 / wall_ms as f64
    } else {
        0.0
    };
    eprintln!(
        "profile done: nodes={} depth={} move={} elapsed_ms={} nps={:.0} stop={}",
        r.nodes,
        r.depth,
        move_id_to_algebraic(r.mv),
        wall_ms,
        nps,
        r.stop_reason,
    );
    emit_result(
        "profile",
        position,
        &r,
        wall_ms,
        full,
        false,
        threads,
        &format!(",\"time_sec\":{sec}"),
    );
    if let Some(instr) = bench_instr::take_json_report() {
        println!("{instr}");
    }
    println!("{}", r.race_outcome_stats.to_json());
    bench_instr::with_bench(|b| {
        eprintln!(
            "race profile: gate_cached_ns={} race_tbl_solve_ns={} gate1_hit_rate={:.1}% race_tbl_rebuilds={} race_tbl_hits={}",
            b.race_gate_cached.ns,
            b.race_winner_table.ns,
            r.race_outcome_stats.gate1_hit_rate_pct(),
            r.race_outcome_stats.race_tbl_lru_rebuilds,
            r.race_outcome_stats.race_tbl_lru_hits,
        );
    });
}

/// Single cold think for per-move flamegraph replay.
fn bench_think_once(time_ms: u64, moves: Option<&str>, full: bool, threads: usize) {
    reset_lazy_seal_stats();
    let moves_s = moves.unwrap_or("").trim();
    let mut search = fresh_search("startpos", if moves_s.is_empty() { None } else { Some(moves_s) });
    let t0 = Instant::now();
    let r = run_think(&mut search, time_ms, MAX_DEPTH, full, false, threads);
    let wall_ms = t0.elapsed().as_millis() as u64;
    emit_result(
        "think",
        "startpos",
        &r,
        wall_ms,
        full,
        false,
        threads,
        &format!(
            ",\"time_ms\":{time_ms},\"moves\":{}",
            json_str(moves_s)
        ),
    );
    println!("{}", r.race_outcome_stats.to_json());
    if let Some(instr) = bench_instr::take_json_report() {
        println!("{instr}");
    }
}

fn cat_micro_run(board: &Board, iterations: usize) -> (u128, u64) {
    let mut checksum = 0u64;
    let t0 = Instant::now();
    for i in 0..iterations {
        let cat = build_impact_heatmap(black_box(board));
        let sq = (i * 37) % 81;
        checksum =
            checksum.wrapping_add(u64::from(cat.square_heat((sq / 9) as u8, (sq % 9) as u8)));
        black_box(&cat);
    }
    (t0.elapsed().as_nanos(), checksum)
}

fn bench_cat_micro(position: &str, ply: Option<usize>, iterations: usize, rounds: usize) {
    let board = load_board(position, ply);
    for _ in 0..100 {
        black_box(build_impact_heatmap(&board));
    }

    let mut catv6_ns = Vec::with_capacity(rounds);
    for round in 0..rounds {
        let (elapsed_ns, checksum) = cat_micro_run(&board, iterations);
        let ns_per_call = elapsed_ns as f64 / iterations as f64;
        let calls_per_sec = 1_000_000_000.0 / ns_per_call;
        println!(
            "{{\"bench_type\":\"cat_micro_raw\",\"position\":{},\"ply\":{},\"round\":{},\"variant\":\"catv6\",\"iterations\":{},\"elapsed_ns\":{},\"ns_per_call\":{:.2},\"calls_per_sec\":{:.0},\"checksum\":{}}}",
            json_str(position),
            ply.map_or_else(|| "null".to_string(), |v| v.to_string()),
            round + 1,
            iterations,
            elapsed_ns,
            ns_per_call,
            calls_per_sec,
            checksum,
        );
        catv6_ns.push(ns_per_call);
    }

    let catv6_median = median_f64(&catv6_ns);
    println!(
        "{{\"bench_type\":\"cat_micro_summary\",\"position\":{},\"ply\":{},\"rounds\":{},\"iterations_per_round\":{},\"catv6_median_ns_per_call\":{:.2},\"catv6_calls_per_sec\":{:.0}}}",
        json_str(position),
        ply.map_or_else(|| "null".to_string(), |v| v.to_string()),
        rounds,
        iterations,
        catv6_median,
        1_000_000_000.0 / catv6_median,
    );
}

fn dump_cat_planes(position: &str, ply: Option<usize>) {
    let board = load_board(position, ply);
    let catv6 = build_impact_heatmap(&board);
    let values = |cat: &CorridorAttention| {
        (0..81usize)
            .map(|sq| cat.square_heat((sq / 9) as u8, (sq % 9) as u8).to_string())
            .collect::<Vec<_>>()
            .join(",")
    };
    let moves = position_moves(position);
    let limit = ply.unwrap_or(moves.len()).min(moves.len());
    println!(
        "{{\"bench_type\":\"cat_dump\",\"position\":{},\"ply\":{},\"moves\":{},\"catv6\":[{}]}}",
        json_str(position),
        limit,
        json_str(&moves[..limit].join(" ")),
        values(&catv6),
    );
}

fn parse_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|a| a == name)
}

fn parse_usize(args: &[String], flag: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_threads(args: &[String]) -> usize {
    let threads = parse_usize(args, "--threads", 1);
    if threads == 0 {
        eprintln!("error --threads must be a positive integer");
        std::process::exit(2);
    }
    threads.min(16)
}

fn parse_u64(args: &[String], flag: &str, default: u64) -> u64 {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_i32(args: &[String], flag: &str, default: i32) -> i32 {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn parse_string<'a>(args: &'a [String], flag: &str, default: &'a str) -> &'a str {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or(default)
}

fn main() {
    prewarm();
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("time");
    let position = parse_string(&args, "--position", "startpos");
    let moves = args
        .iter()
        .position(|a| a == "--moves")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str());
    let full = parse_flag(&args, "--full");
    let threads = parse_threads(&args);
    match mode {
        "time" => {
            let sec = parse_u64(&args, "--sec", 10);
            let runs = parse_usize(&args, "--runs", 5);
            let log = parse_flag(&args, "--log");
            bench_time(sec, runs, position, full, log, threads);
        }
        "depth" => {
            let depth = parse_i32(&args, "--depth", 6);
            bench_depth(depth, position, full, threads);
        }
        "profile" => {
            let sec = parse_u64(&args, "--sec", 30);
            bench_profile(sec, position, moves, full, threads);
        }
        "instr" => {
            let sec = parse_u64(&args, "--sec", 10);
            bench_profile(sec, position, moves, full, threads);
        }
        "think" => {
            let ms = parse_u64(&args, "--ms", 1000);
            bench_think_once(ms, moves, full, threads);
        }
        "catmicro" => {
            let iterations = parse_usize(&args, "--iters", 10_000);
            let rounds = parse_usize(&args, "--runs", 7);
            let ply = args
                .iter()
                .position(|a| a == "--ply")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok());
            bench_cat_micro(position, ply, iterations, rounds);
        }
        "catdump" => {
            let ply = args
                .iter()
                .position(|a| a == "--ply")
                .and_then(|i| args.get(i + 1))
                .and_then(|s| s.parse().ok());
            dump_cat_planes(position, ply);
        }
        other => {
            eprintln!(
                "unknown mode {other}; use time|depth|profile|think|instr|catmicro|catdump [--position NAME] [--moves ...] [--ms N] [--full] [--sec N] [--runs N]"
            );
            std::process::exit(2);
        }
    }
}
