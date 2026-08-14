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

    emit_git_commit();
}

/// Stamp the build with the commit it came from.
///
/// `wasm.rs` already reads `GIT_COMMIT_HASH`, but nothing ever set it, so every
/// build reported "unknown". Without it a gate result cannot be tied back to
/// source: binaries are not bit-reproducible here (MSVC embeds paths and
/// timestamps), so rebuilding a commit yields a different file and hashing the
/// executable proves nothing. The commit has to be carried inside the binary.
///
/// A dirty tree is marked, because "built from abc1234" is a lie if the working
/// copy had uncommitted edits — which is exactly how an untraceable binary got
/// into a gate earlier today.
fn emit_git_commit() {
    use std::process::Command;

    let hash = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=no"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(false);

    let stamp = if dirty { format!("{hash}-dirty") } else { hash };
    println!("cargo:rustc-env=GIT_COMMIT_HASH={stamp}");

    // Rebuild when HEAD moves, so the stamp cannot go stale behind a cached build.
    for p in ["../.git/HEAD", "../.git/index", ".git/HEAD", ".git/index"] {
        if std::path::Path::new(p).exists() {
            println!("cargo:rerun-if-changed={p}");
        }
    }
}
