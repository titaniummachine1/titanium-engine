//! Engine build does NOT generate movegen tables — run `cargo run --bin movegen-o1-gen`.
//!
//! Suboptimal-build guard: on x86_64, compiling without BMI2/PEXT produces a
//! binary that runs the scalar pawn-LUT fallback (~4× slower). The build script
//! hard-errors to prevent accidentally shipping such a build.
//!
//! BYPASS (cross-compile / intentional scalar build):
//!   TITANIUM_ALLOW_SUBOPTIMAL=1 cargo build [...]

fn main() {
    // ── Movegen table presence check ─────────────────────────────────────────
    let tables = "src/movegen/o1/generated_tables_data.rs";
    let remap = "src/movegen/o1/generated_remap.bin";
    if !std::path::Path::new(tables).exists() || !std::path::Path::new(remap).exists() {
        println!("cargo:warning=missing movegen-o1 tables — run: cargo run --bin movegen-o1-gen");
    }
    println!("cargo:rerun-if-changed={tables}");
    println!("cargo:rerun-if-changed={remap}");
    println!("cargo:rerun-if-changed=build/movegen_o1/mod.rs");

    // ── Suboptimal-build guard ────────────────────────────────────────────────
    // Re-run whenever the bypass env var changes so flipping it triggers recheck.
    println!("cargo:rerun-if-env-changed=TITANIUM_ALLOW_SUBOPTIMAL");

    let arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let features = std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    let has_bmi2 = features.split(',').any(|f| f.trim() == "bmi2");

    if arch == "x86_64" && !has_bmi2 {
        let bypass = std::env::var("TITANIUM_ALLOW_SUBOPTIMAL")
            .map(|v| v == "1")
            .unwrap_or(false);

        let msg = "\
\n\
╔══════════════════════════════════════════════════════════════════════╗\n\
║  TITANIUM BUILD ERROR — SUBOPTIMAL TARGET FEATURES                  ║\n\
║                                                                      ║\n\
║  Building for x86_64 without BMI2/PEXT produces a binary that runs  ║\n\
║  the scalar pawn-LUT fallback, which is ~4× slower than the native  ║\n\
║  PEXT path. This is almost certainly not what you want.              ║\n\
║                                                                      ║\n\
║  FIX — compile with native CPU features:                             ║\n\
║    RUSTFLAGS='-C target-cpu=native' cargo build --release            ║\n\
║                                                                      ║\n\
║  BYPASS — intentional scalar / cross-compile build:                  ║\n\
║    TITANIUM_ALLOW_SUBOPTIMAL=1 cargo build [...]                     ║\n\
╚══════════════════════════════════════════════════════════════════════╝\n";

        if bypass {
            // Still loud, but let it through.
            println!("cargo:warning={}", msg.replace('\n', " | "));
        } else {
            eprintln!("{msg}");
            std::process::exit(1);
        }
    }
}
