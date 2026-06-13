//! Engine build does NOT generate movegen tables — run `cargo run --bin movegen-o1-gen`.

fn main() {
    let tables = "src/movegen/o1/generated_tables_data.rs";
    let remap = "src/movegen/o1/generated_remap.bin";
    let missing =
        !std::path::Path::new(tables).exists() || !std::path::Path::new(remap).exists();
    if missing {
        println!("cargo:warning=missing movegen-o1 tables — run: cargo run --bin movegen-o1-gen");
    }
    println!("cargo:rerun-if-changed={tables}");
    println!("cargo:rerun-if-changed={remap}");
    println!("cargo:rerun-if-changed=build/movegen_o1/mod.rs");
}
