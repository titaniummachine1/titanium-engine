//! Work comparison: V10 wall-trial machinery vs V11 parallel flood,
//! measured on the 15 canta midgame positions (wall-heavy, realistic).
use std::time::Instant;
use titanium::core::board::{Board, Player, WallOrientation};
use titanium::oracle::canta::board_after_canta_game;
use titanium::path::flood::{flood_to_goal, goal_square_mask};
use titanium::path::masks::DirMasks;
use titanium::path::parallel::{
    expand_wave, pawn_bit, wall_delta, WallGrids, P1_GOAL_BITS, P2_GOAL_BITS,
};
use titanium::path::BfsScratch;
use titanium::util::grid::{has_wall, set_wall, square_index, FLOOD_PLAYABLE};

fn collides(b: &Board, r: u8, c: u8, o: WallOrientation) -> bool {
    let p = match o {
        WallOrientation::Horizontal => WallOrientation::Vertical,
        WallOrientation::Vertical => WallOrientation::Horizontal,
    };
    if has_wall(b, r, c, o) || has_wall(b, r, c, p) {
        return true;
    }
    match o {
        WallOrientation::Horizontal => {
            (c > 0 && has_wall(b, r, c - 1, o)) || (c < 7 && has_wall(b, r, c + 1, o))
        }
        WallOrientation::Vertical => {
            (r > 0 && has_wall(b, r - 1, c, o)) || (r < 7 && has_wall(b, r + 1, c, o))
        }
    }
}

/// Candidate walls that survive L2 collision (every one runs flood in L3).
fn flood_candidates(b: &Board) -> Vec<(u8, u8, WallOrientation)> {
    let mut out = Vec::new();
    for o in [WallOrientation::Horizontal, WallOrientation::Vertical] {
        for r in 0..8u8 {
            for c in 0..8u8 {
                if !collides(b, r, c, o) {
                    out.push((r, c, o));
                }
            }
        }
    }
    out
}

/// V11 flood with iteration/theft counters (mirror of path::parallel logic).
fn v11_counted(p1: u128, p2: u128, g: &WallGrids, iters: &mut u64, thefts: &mut u64) -> bool {
    let (ok1, p1_vis) = {
        let mut visited = p1 & FLOOD_PLAYABLE;
        let mut wave = visited;
        let mut ok = visited & P1_GOAL_BITS != 0;
        while !ok && wave != 0 {
            *iters += 1;
            wave = expand_wave(wave, g) & !visited;
            ok = wave & P1_GOAL_BITS != 0;
            visited |= wave;
        }
        (ok, visited)
    };
    if !ok1 {
        return false;
    }
    let mut visited = p2 & FLOOD_PLAYABLE;
    if visited & P2_GOAL_BITS != 0 {
        return true;
    }
    let mut wave = visited;
    let mut pool = p1_vis & !visited;
    while wave != 0 {
        if wave & pool != 0 {
            *thefts += 1;
            if pool & P2_GOAL_BITS != 0 {
                return true;
            }
            visited |= pool;
            wave |= pool;
            pool = 0;
        }
        *iters += 1;
        wave = expand_wave(wave, g) & !visited;
        if wave & P2_GOAL_BITS != 0 {
            return true;
        }
        visited |= wave;
    }
    false
}

fn main() {
    let boards: Vec<Board> = (0..15).map(board_after_canta_game).collect();
    let cands: Vec<Vec<_>> = boards.iter().map(flood_candidates).collect();
    let total_trials: usize = cands.iter().map(|v| v.len()).sum();
    let reps = 2000u32;

    // ── V10: per-trial set_wall + DirMasks rebuild + two independent floods ──
    let mut scratch = BfsScratch::new();
    let mut v10_legal = 0u64;
    let t0 = Instant::now();
    for _ in 0..reps {
        v10_legal = 0;
        for (b, cs) in boards.iter().zip(&cands) {
            let mut board = b.clone();
            for &(r, c, o) in cs {
                set_wall(&mut board, r, c, o, true);
                scratch.invalidate_dir_masks();
                if scratch.both_players_reach_goals(&board) {
                    v10_legal += 1;
                }
                set_wall(&mut board, r, c, o, false);
            }
        }
    }
    let v10_dt = t0.elapsed();

    // ── V10 floods alone: DirMasks reused across trials (NOT correct for
    //    trials — pure lower bound on the old flood machinery's cost) ──
    let t0 = Instant::now();
    for _ in 0..reps {
        for (b, cs) in boards.iter().zip(&cands) {
            let masks = DirMasks::from_board(b);
            let (r1, c1) = b.pawn(Player::One);
            let (r2, c2) = b.pawn(Player::Two);
            for _ in cs {
                let (ok1, _) = std::hint::black_box(flood_to_goal(
                    square_index(r1, c1),
                    masks,
                    goal_square_mask(Player::One),
                ));
                if ok1 {
                    std::hint::black_box(flood_to_goal(
                        square_index(r2, c2),
                        masks,
                        goal_square_mask(Player::Two),
                    ));
                }
            }
        }
    }
    let v10_floodonly_dt = t0.elapsed();

    // ── V11 with counters ──
    let mut v11_legal = 0u64;
    let mut iters = 0u64;
    let mut thefts = 0u64;
    let t0 = Instant::now();
    for _ in 0..reps {
        v11_legal = 0;
        iters = 0;
        thefts = 0;
        for (b, cs) in boards.iter().zip(&cands) {
            let mut grids = WallGrids::from_board(b);
            let (r1, c1) = b.pawn(Player::One);
            let (r2, c2) = b.pawn(Player::Two);
            let (p1, p2) = (pawn_bit(r1, c1), pawn_bit(r2, c2));
            for &(r, c, o) in cs {
                let d = wall_delta(r, c, o);
                grids.place(d);
                if v11_counted(p1, p2, &grids, &mut iters, &mut thefts) {
                    v11_legal += 1;
                }
                grids.remove(d);
            }
        }
    }
    let v11_dt = t0.elapsed();

    // ── V11 pure timing (production path, no counters) ──
    let t0 = Instant::now();
    for _ in 0..reps {
        for (b, cs) in boards.iter().zip(&cands) {
            let mut grids = WallGrids::from_board(b);
            let (r1, c1) = b.pawn(Player::One);
            let (r2, c2) = b.pawn(Player::Two);
            let (p1, p2) = (pawn_bit(r1, c1), pawn_bit(r2, c2));
            for &(r, c, o) in cs {
                let d = wall_delta(r, c, o);
                grids.place(d);
                std::hint::black_box(titanium::path::pbff_wall_legal(p1, p2, &grids));
                grids.remove(d);
            }
        }
    }
    let v11_pure_dt = t0.elapsed();

    let n = (total_trials as u32 * reps) as f64;
    println!(
        "15 canta midgames, {} flood-gated wall trials/pass, {} reps",
        total_trials, reps
    );
    println!(
        "V10 trial (set_wall + DirMasks rebuild + 2 floods): {:>8.1} ns/trial  (legal={})",
        v10_dt.as_nanos() as f64 / n,
        v10_legal
    );
    println!(
        "V10 floods alone (masks prebuilt, lower bound):     {:>8.1} ns/trial",
        v10_floodonly_dt.as_nanos() as f64 / n
    );
    println!(
        "V11 trial (mask flip + parallel flood + theft):     {:>8.1} ns/trial  (legal={})",
        v11_pure_dt.as_nanos() as f64 / n,
        v11_legal
    );
    println!(
        "V11 with counters:                                  {:>8.1} ns/trial",
        v11_dt.as_nanos() as f64 / n
    );
    println!();
    println!(
        "work reduction, full trial: {:.1}x   vs floods-alone lower bound: {:.1}x",
        v10_dt.as_nanos() as f64 / v11_pure_dt.as_nanos() as f64,
        v10_floodonly_dt.as_nanos() as f64 / v11_pure_dt.as_nanos() as f64
    );
    println!("V10 can_step calls per trial (DirMasks rebuild): 324");
    println!(
        "V11 dilation iterations per trial: {:.2}",
        iters as f64 / total_trials as f64
    );
    println!(
        "V11 bit-theft hit rate: {:.1}% of trials reuse P1's flood for P2",
        100.0 * thefts as f64 / total_trials as f64
    );
}
