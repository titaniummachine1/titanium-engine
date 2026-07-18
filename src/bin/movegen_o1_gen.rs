//! Generate O(1) movegen lookup tables (separate from `cargo build`).
//!
//! Run from the engine directory:
//!   cargo run --bin movegen-o1-gen

use std::path::PathBuf;
use titanium::movegen::o1::gen::generate;

fn main() {
    let out =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/movegen/o1/generated_tables_data.rs");
    eprintln!("movegen-o1-gen → {}", out.display());
    generate(&out);
    eprintln!("done.");
}
