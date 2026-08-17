//! The engine repo is strength-only. It holds the engine and what is needed to
//! build, test and benchmark it — nothing else.
//!
//! Session handoffs, training scaffolding, run logs, corpora, weights and built
//! binaries belong to the workspace above this repo (`archive/engine-handoffs/`,
//! `training/`, `results/`), never here. This test is the guardrail: `.gitignore`
//! stops the accidental `git add`, and this stops the deliberate one.
//!
//! It reads `git ls-files`, so it fails on what is TRACKED, not on what happens
//! to sit in the working tree.

use std::path::Path;
use std::process::Command;

/// Path prefixes that must never be tracked here.
const FORBIDDEN_DIRS: &[&str] = &[
    "training/", // training pipeline lives in the workspace above
    "runs/",     // run logs and metrics
    "gates/",    // built gate binaries
    "data/",     // corpora
    "weights/",  // net checkpoints (src/weights/ is exempted below)
];

/// Filename patterns that must never be tracked here. `*` matches any run of
/// characters; matching is case-insensitive.
const FORBIDDEN_FILES: &[&str] = &[
    "*handoff*",       // session handoff notes
    "morning_brief*",  // daily briefs
    "session_*",       // session archives
    "brief_*",
    "progress.txt",
    "baseline_v*.txt", // bench dumps
    "*.log",
    "*.jsonl",
    "*.db",
    "*.exe",
    "*.dll",
    "*.pt",
    "*.pth",
    "*.onnx",
    "*.npz",
    "*.patch",
];

/// Paths allowed through despite matching a rule above.
const EXEMPT: &[&str] = &[
    "src/weights/", // the compiled-in eval weights ARE the engine
];

fn glob_match(pattern: &str, name: &str) -> bool {
    // Only `*` is supported, which is all these patterns need.
    let parts: Vec<&str> = pattern.split('*').collect();
    let mut rest = name;

    if let Some(first) = parts.first() {
        if !rest.starts_with(first) {
            return false;
        }
        rest = &rest[first.len()..];
    }
    if let Some(last) = parts.last() {
        if parts.len() > 1 {
            if rest.len() < last.len() || !rest.ends_with(last) {
                return false;
            }
            rest = &rest[..rest.len() - last.len()];
        }
    }
    for mid in parts.iter().skip(1).take(parts.len().saturating_sub(2)) {
        match rest.find(mid) {
            Some(i) => rest = &rest[i + mid.len()..],
            None => return false,
        }
    }
    true
}

fn reason(path: &str) -> Option<String> {
    let lower = path.to_ascii_lowercase();

    if EXEMPT.iter().any(|e| lower.starts_with(e)) {
        return None;
    }
    if let Some(dir) = FORBIDDEN_DIRS.iter().find(|d| lower.starts_with(**d)) {
        return Some(format!("lives under `{dir}`"));
    }
    let name = lower.rsplit('/').next().unwrap_or(&lower);
    if let Some(pat) = FORBIDDEN_FILES.iter().find(|p| glob_match(p, name)) {
        return Some(format!("matches `{pat}`"));
    }
    None
}

#[test]
fn engine_repo_is_strength_only() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));

    let out = match Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(root)
        .output()
    {
        Ok(o) if o.status.success() => o,
        // No git, or a source tarball with no history: nothing to check.
        _ => return,
    };

    let tracked = String::from_utf8_lossy(&out.stdout);
    let offenders: Vec<String> = tracked
        .split('\0')
        .filter(|p| !p.is_empty())
        .filter_map(|p| reason(p).map(|why| format!("  {p}  ({why})")))
        .collect();

    assert!(
        offenders.is_empty(),
        "\n\
         {} file(s) tracked in the engine repo that do not belong here:\n\
         {}\n\n\
         The engine repo is strength-only: engine sources, build, tests, benches, docs.\n\
         Move these to the workspace above and commit them there instead:\n\
         \x20 session notes / handoffs / briefs -> ../archive/engine-handoffs/<yyyy-mm>/\n\
         \x20 training pipeline and run output  -> ../training/\n\
         \x20 bench dumps and match results     -> ../results/\n\
         \x20 built binaries                    -> not versioned at all\n\n\
         If a path is genuinely part of the engine, add it to EXEMPT in this file\n\
         with a comment saying why.\n",
        offenders.len(),
        offenders.join("\n"),
    );
}

#[test]
fn glob_match_works() {
    assert!(glob_match("*handoff*", "training_handoff.md"));
    assert!(glob_match("*handoff*", "handoff.md"));
    assert!(glob_match("*.log", "loop.log"));
    assert!(glob_match("progress.txt", "progress.txt"));
    assert!(glob_match("baseline_v*.txt", "baseline_v18.txt"));
    assert!(glob_match("morning_brief*", "morning_brief_2026-08-16.md"));

    assert!(!glob_match("*handoff*", "readme.md"));
    assert!(!glob_match("*.log", "logic.rs"));
    assert!(!glob_match("progress.txt", "progress.md"));
    assert!(!glob_match("baseline_v*.txt", "baseline.md"));
}
