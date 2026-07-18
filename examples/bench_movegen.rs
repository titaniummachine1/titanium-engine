//! Wall-movegen throughput: perft over the 15 canta midgame positions.
use std::time::Instant;
use titanium::validation::canta::board_after_canta_game;
use titanium::util::perft::perft_fast;

fn main() {
    // Warmup
    for game in 0..15 {
        let mut b = board_after_canta_game(game);
        let _ = perft_fast(&mut b, 2);
    }
    let start = Instant::now();
    let mut total = 0u64;
    for game in 0..15 {
        let mut b = board_after_canta_game(game);
        total += perft_fast(&mut b, 3);
    }
    let dt = start.elapsed();
    println!(
        "perft d3 x15 canta: {} nodes in {:?} ({:.2} Mnodes/s)",
        total,
        dt,
        total as f64 / dt.as_secs_f64() / 1e6
    );

    let start = Instant::now();
    let mut b = titanium::core::board::Board::new();
    let n = perft_fast(&mut b, 4);
    let dt = start.elapsed();
    println!(
        "perft d4 startpos: {} nodes in {:?} ({:.2} Mnodes/s)",
        n,
        dt,
        n as f64 / dt.as_secs_f64() / 1e6
    );
}
