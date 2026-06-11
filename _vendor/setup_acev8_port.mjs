import { execSync } from 'node:child_process';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));

function writeFromGit(relPath, ref) {
  const txt = execSync(`git show ${ref}:${relPath}`, { cwd: root, encoding: 'utf8' });
  fs.writeFileSync(path.join(root, relPath), txt, 'utf8');
}

writeFromGit('engine/src/ace/search.rs', 'acev7-port');
writeFromGit('engine/src/ace/perft.rs', 'acev7-port');

let search = fs.readFileSync(path.join(root, 'engine/src/ace/search.rs'), 'utf8');
search = search.replace(
  `                    if !full
                        && d >= 6
                        && stable >= 2
                        && t0.elapsed().as_millis() as u64 > time_ms / 10
                    {
                        break; // easy move
                    }`,
  `                    // v8 easy-move stop (acev8_engine.js)
                    if !full
                        && d >= 9
                        && stable >= 3
                        && last_score > -120
                        && t0.elapsed().as_millis() as u64 > time_ms * 3 / 10
                    {
                        break;
                    }`,
);
search = search.replace(
  '            if t0.elapsed().as_millis() as f64 > time_ms as f64 * 0.6 {',
  `            let time_frac = if last_score < -80 { 0.92 } else { 0.85 };
            if t0.elapsed().as_millis() as f64 > time_ms as f64 * time_frac {`,
);
fs.writeFileSync(path.join(root, 'engine/src/ace/search.rs'), search, 'utf8');

writeFromGit('engine/src/main.rs', 'acev7-port');
let main = fs.readFileSync(path.join(root, 'engine/src/main.rs'), 'utf8');
main = main.replace(
  `fn ace_engine_mode(args: &[String]) -> Option<&'static str> {
    args.windows(2).find_map(|w| {
        if w[0] != "--engine" {
            return None;
        }
        match w[1].as_str() {
            "ace" => Some("ace"),
            "ace-cat" => Some("ace-cat"),
            "ace-ti" => Some("ace-ti"),
            _ => None,
        }
    })
}

fn is_ace_engine(args: &[String]) -> bool {
    ace_engine_mode(args).is_some()
}

fn run_genmove_ace(args: &[String]) {
    let mode = ace_engine_mode(args).unwrap_or("ace");
    let mut params = titanium::ace::AceParams {
        cat: mode == "ace-cat",
        ti_movegen: mode == "ace-ti",`,
  `fn ace_engine_flag(args: &[String]) -> Option<&str> {
    args.windows(2).find_map(|w| {
        if w[0] != "--engine" {
            return None;
        }
        match w[1].as_str() {
            "ace" | "ace-v8" | "ace-cat" | "ace-ti" | "ace-v8-ti" => Some(w[1].as_str()),
            _ => None,
        }
    })
}

fn ace_engine_mode(flag: &str) -> &'static str {
    match flag {
        "ace-cat" => "ace-cat",
        "ace-ti" | "ace-v8-ti" => "ace-ti",
        _ => "ace",
    }
}

fn is_ace_engine(args: &[String]) -> bool {
    ace_engine_flag(args).is_some()
}

fn run_genmove_ace(args: &[String]) {
    let label = ace_engine_flag(args).unwrap_or("ace");
    let mode = ace_engine_mode(label);
    let mut params = titanium::ace::AceParams {
        cat: mode == "ace-cat",
        ti_movegen: mode == "ace-ti",`,
);
main = main.replace(
  `            let name = mode;
            eprintln!(
                "info json {{\\"engine\\":\\"{}\\",\\"rootScore\\":{},\\"searchDepth\\":{},\\"nodes\\":{},\\"elapsedMs\\":{}}}",
                name, info.score, info.depth, info.nodes, info.ms
            );`,
  `            eprintln!(
                "info json {{\\"engine\\":\\"{}\\",\\"rootScore\\":{},\\"searchDepth\\":{},\\"nodes\\":{},\\"elapsedMs\\":{}}}",
                label, info.score, info.depth, info.nodes, info.ms
            );`,
);
fs.writeFileSync(path.join(root, 'engine/src/main.rs'), main, 'utf8');

console.log('acev8-port files restored');
