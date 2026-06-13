//! Generate O(1) movegen lookup tables (separate from `cargo build`).
//!
//! Run from the engine directory:
//!   cargo run --bin movegen-o1-gen

#[path = "../../build/movegen_o1/mod.rs"]
mod movegen_o1;

use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("src/movegen/o1/generated_tables_data.rs");
    eprintln!("movegen-o1-gen → {}", out.display());
    movegen_o1::generate(&out);
    eprintln!("done.");
}
