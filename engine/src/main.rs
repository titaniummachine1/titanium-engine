//! Checkpoint 03 CLI — perft only.

use std::env;
use titanium::{perft, Board};

fn main() {
    let depth: u32 = env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1);
    let board = Board::new();
    println!("perft {} {}", depth, perft(&board, depth));
}
