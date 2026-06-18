//! Flood-fill regime study: one-step PBFF (`expand_wave`) vs Kogge-Stone PBFF.
//! Run: `cargo bench --bench flood_modes`
//! Native (recommended): `$env:RUSTFLAGS='-C target-cpu=native'; cargo bench --bench flood_modes`
//!
//! WHY THIS EXISTS — the two floods have a crossover:
//!   * Open board (opening, long unobstructed runs): KS doubling fills a whole
//!     corridor in O(log w) shifts → can be far faster.
//!   * Dense board (middlegame, short snaking paths): the one-step flood
//!     converges in a couple of rings, and KS's per-call `KsProp::new` setup
//!     (24 shift/AND ops, paid twice — P1 + P2) never amortises → KS loses.
//!
//! The aggregate-corpus bench hid this by averaging densities together. This
//! version reports PER REGIME so the crossover is visible:
//!   * `startpos`  — wide-open board ("billion-times first move" regime).
//!   * `canta-gNN` — 15 real middlegame positions (15 plies of play, walls down).
//!
//! For each regime we measure ONLY the walls that actually trigger an L3 flood
//! in the hot path (`l12 & topo` — topology-touching legal candidates), because
//! that is exactly the per-node cost perft pays. We also print the open-board
//! "all 128 slots" raw throughput so the best case for KS is on record.

use std::time::Instant;

use titanium::core::board::Board;
use titanium::core::board::WallOrientation;
use titanium::movegen::o1::{wall_masks, WallMasks};
use titanium::oracle::canta::board_after_canta_game;
use titanium::path::parallel::{
    pawn_bit, pbff_ks_wall_legal, pbff_wall_legal, wall_delta, WallGrids,
};

/// A precomputed flood trial: base grids with one candidate wall's delta applied.
struct Trial {
    grids: WallGrids,
}

struct Regime {
    name: String,
    walls_on_board: u32,
    p1: u128,
    p2: u128,
    /// Hot-path candidates: legal walls that touch topology (these actually flood).
    hot: Vec<Trial>,
    /// All physically-placeable wall deltas (raw flood throughput at this density).
    all: Vec<Trial>,
}

fn bit_to_rc(bit: u32) -> (u8, u8) {
    ((bit / 8) as u8, (bit % 8) as u8)
}

fn build_trials(board: &Board, mask_hot_h: u64, mask_hot_v: u64) -> (Vec<Trial>, Vec<Trial>) {
    let base = WallGrids::from_board(board);
    let mut hot = Vec::new();
    let mut all = Vec::new();

    // Hot: topology-touching legal candidates (what the hot path floods).
    let mut h = mask_hot_h;
    while h != 0 {
        let (r, c) = bit_to_rc(h.trailing_zeros());
        h &= h - 1;
        let mut g = base;
        g.place(wall_delta(r, c, WallOrientation::Horizontal));
        hot.push(Trial { grids: g });
    }
    let mut v = mask_hot_v;
    while v != 0 {
        let (r, c) = bit_to_rc(v.trailing_zeros());
        v &= v - 1;
        let mut g = base;
        g.place(wall_delta(r, c, WallOrientation::Vertical));
        hot.push(Trial { grids: g });
    }

    // All: every wall slot (64 H + 64 V), regardless of legality — pure flood load.
    for slot in 0..64u32 {
        let (r, c) = bit_to_rc(slot);
        for o in [WallOrientation::Horizontal, WallOrientation::Vertical] {
            let mut g = base;
            g.place(wall_delta(r, c, o));
            all.push(Trial { grids: g });
        }
    }

    (hot, all)
}

fn make_regime(name: String, board: &Board) -> Regime {
    let WallMasks {
        l12_h,
        l12_v,
        topo_h,
        topo_v,
    } = wall_masks(board);
    let (hot, all) = build_trials(board, l12_h & topo_h, l12_v & topo_v);
    let (r1, c1) = board.pawns[0];
    let (r2, c2) = board.pawns[1];
    Regime {
        name,
        walls_on_board: (board.horizontal_walls | board.vertical_walls).count_ones(),
        p1: pawn_bit(r1, c1),
        p2: pawn_bit(r2, c2),
        hot,
        all,
    }
}

/// Time both floods over a trial set; return (step_secs, ks_secs, mismatches, sink).
fn time_set(p1: u128, p2: u128, trials: &[Trial], passes: u32) -> (f64, f64, usize, u64) {
    if trials.is_empty() {
        return (0.0, 0.0, 0, 0);
    }
    let mut mism = 0usize;
    for t in trials {
        if pbff_wall_legal(p1, p2, &t.grids) != pbff_ks_wall_legal(p1, p2, &t.grids) {
            mism += 1;
        }
    }

    let t0 = Instant::now();
    let mut sink_step = 0u64;
    for _ in 0..passes {
        for t in trials {
            sink_step += pbff_wall_legal(p1, p2, &t.grids) as u64;
        }
    }
    let step = t0.elapsed().as_secs_f64();

    let t1 = Instant::now();
    let mut sink_ks = 0u64;
    for _ in 0..passes {
        for t in trials {
            sink_ks += pbff_ks_wall_legal(p1, p2, &t.grids) as u64;
        }
    }
    let ks = t1.elapsed().as_secs_f64();

    assert_eq!(sink_step, sink_ks, "sink mismatch");
    (step, ks, mism, sink_step)
}

fn ns_per_query(secs: f64, n: usize, passes: u32) -> f64 {
    if n == 0 {
        return 0.0;
    }
    secs * 1e9 / (n as f64 * passes as f64)
}

fn main() {
    titanium::movegen::prewarm();

    let mut regimes = Vec::new();
    regimes.push(make_regime("startpos".to_string(), &Board::new()));
    for g in 0..15 {
        regimes.push(make_regime(
            format!("canta-g{g:02}"),
            &board_after_canta_game(g),
        ));
    }

    const PASSES: u32 = 20_000;

    println!("=== HOT flood candidates (l12 & topo — what perft actually floods) ===");
    println!("| regime | walls | #hot | step ns | ks ns | KS speedup | winner |");
    println!("|--------|------:|-----:|--------:|------:|-----------:|:-------|");

    let mut total_mism = 0usize;
    let mut ks_wins_hot = 0usize;
    let mut step_wins_hot = 0usize;

    for r in &regimes {
        let (step, ks, mism, _) = time_set(r.p1, r.p2, &r.hot, PASSES);
        total_mism += mism;
        let step_ns = ns_per_query(step, r.hot.len(), PASSES);
        let ks_ns = ns_per_query(ks, r.hot.len(), PASSES);
        let speedup = if ks > 0.0 { step / ks } else { 0.0 };
        let winner = if r.hot.is_empty() {
            "—"
        } else if speedup > 1.0 {
            ks_wins_hot += 1;
            "KS"
        } else {
            step_wins_hot += 1;
            "step"
        };
        println!(
            "| {} | {} | {} | {:.1} | {:.1} | {:.3}x | {} |",
            r.name,
            r.walls_on_board,
            r.hot.len(),
            step_ns,
            ks_ns,
            speedup,
            winner
        );
    }

    println!("\n=== ALL 128 wall slots (raw flood throughput at each density) ===");
    println!("| regime | walls | step ns | ks ns | KS speedup | winner |");
    println!("|--------|------:|--------:|------:|-----------:|:-------|");

    for r in &regimes {
        let (step, ks, mism, _) = time_set(r.p1, r.p2, &r.all, PASSES);
        total_mism += mism;
        let step_ns = ns_per_query(step, r.all.len(), PASSES);
        let ks_ns = ns_per_query(ks, r.all.len(), PASSES);
        let speedup = if ks > 0.0 { step / ks } else { 0.0 };
        let winner = if speedup > 1.0 { "KS" } else { "step" };
        println!(
            "| {} | {} | {:.1} | {:.1} | {:.3}x | {} |",
            r.name, r.walls_on_board, step_ns, ks_ns, speedup, winner
        );
    }

    assert_eq!(total_mism, 0, "KS disagreed with step flood somewhere");
    println!(
        "\nHOT-candidate verdict: KS faster in {ks_wins_hot} regimes, step faster in {step_wins_hot}."
    );
    println!("(0 mismatches across all regimes — algorithms agree.)");
}
