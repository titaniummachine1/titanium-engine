//! ACE v10 HalfPW net — weights from `engine/src/weights/net_weights.bin`.
//!
//! Philosophy: NN = geometric prior, search = tactical proof. See `field_planes.rs`.
//!
//! Four embedded blobs (always under `engine/src/weights/`, not `titanium/`):
//!   `net_weights.bin`        — live production (deploy / training updates this) → v18
//!   `net_weights_v17.bin`    — frozen website-era v17 snapshot (compare target)
//!   `net_weights_frozen.bin` — pinned v13 baseline (ti-pure anchor + v15-frozen)
//!   `net_weights_medium.bin` — browser Medium tier, also used by native proxy
//!
//! Blob layout (little-endian):
//!   NetH[1 x u64]  Wskip[20] B1[NetH] W2[NetH] W1C[9*128*NetH] PO[81*NetH] PX[81*NetH]
//!   goal_inv_p0, goal_inv_p1, pawn_fwd_p0, pawn_fwd_p1,
//!   corridor_delta_p0, corridor_delta_p1, path_cross_p0, path_cross_p1,
//!   choke_p0, choke_p1, contested  (each 81, NetH-independent)
//!
//! `NetH` is an explicit 8-byte header read ONCE at cold start (see `net()` /
//! `net_frozen()` / `net_medium()`, all `OnceLock`-backed) -- NOT inferred from
//! blob length, NOT re-checked per eval call. This lets a differently-sized
//! HalfPW (produced by e.g. `training/tools/net2net_widen.py`) load and run
//! with zero source edits or rebuilds, as long as its width fits `MAX_NET_H`.
//! Hot-path arrays (`b1`, `w2`, and the per-search accumulators in
//! `search.rs`) are fixed-size `[f64; MAX_NET_H]` for stack allocation and
//! predictable codegen; only the first `h` slots are ever populated/read.
use sha2::{Digest, Sha256};
use std::sync::OnceLock;

/// Upper bound on hidden width any loaded net may declare. Bump this (and
/// rebuild) only if an experiment needs a wider net than this allows --
/// everything else adapts automatically from the blob's own header.
pub const MAX_NET_H: usize = 256;
pub const WSKIP_LEN: usize = 20;
const FIELD_PLANE_LEN: usize = 81;
const FIELD_PLANE_SETS: usize = 5;
const H_HEADER_LEN: usize = 8;

/// Distinct (walls_me, walls_opp) pairs: each hand holds 0..=10, so 11 x 11.
/// Every pair is reachable -- walls are conserved at `board + both hands == 20`,
/// so any hand pair just implies a board count, never an illegal state.
pub const WH_PAIRS: usize = 121;

/// Row index of a hand pair in the walls-in-hand embedding.
///
/// Ordered, not symmetric: the mover's own hand is the major term, so `(10, 0)`
/// and `(0, 10)` are different rows. A symmetric encoding (sum, or difference)
/// would collapse exactly the distinction the input was added to make.
///
/// Hands are clamped rather than asserted. A hand above ten means the caller
/// already has a corrupt position, and bending the eval beats panicking in
/// search; the invariant is enforced where walls are placed, not here.
#[inline]
pub fn wh_index(walls_me: usize, walls_opp: usize) -> usize {
    walls_me.min(10) * 11 + walls_opp.min(10)
}

static NET_BYTES: &[u8] = include_bytes!("../../weights/net_weights.bin");
static NET_FROZEN_BYTES: &[u8] = include_bytes!("../../weights/net_weights_frozen.bin");
static NET_MEDIUM_BYTES: &[u8] = include_bytes!("../../weights/net_weights_medium.bin");
/// Website-era Titanium v17 snapshot — never overwritten by deploy_accepted_to_website.
static NET_V17_BYTES: &[u8] = include_bytes!("../../weights/net_weights_v17.bin");

pub struct Net {
    /// Active hidden width for THIS loaded net (<= MAX_NET_H). Everything
    /// downstream (search.rs eval) loops `0..h`, never a compile-time NET_H.
    pub h: usize,
    pub ws: [f64; WSKIP_LEN],
    pub b1: [f64; MAX_NET_H],
    pub w2: [f64; MAX_NET_H],
    pub w1c: Vec<f64>,
    pub po: Vec<f64>,
    pub px: Vec<f64>,
    /// Walls-in-hand embedding, `[pair][hidden]` flattened, `WH_PAIRS * h`.
    ///
    /// Indexed `(walls_me * 11 + walls_opp) * h`, side-to-move canonical. Until
    /// this existed the net could not tell `(10,0)` from `(0,10)`: the two are
    /// byte-identical on every other input, yet one is a won endgame and the
    /// other a lost one. A joint embedding rather than two scalars because the
    /// value of holding a wall depends entirely on how many the opponent holds.
    ///
    /// Zero-filled when the blob omits the section, and adding 0.0 is exact, so
    /// an old net evaluates bit-identically through this path.
    pub wh: Vec<f64>,
    pub wh_active: bool,
    /// Combined CAT impact heatmap as a direct input plane (81, side-to-move
    /// canonical). Zero in legacy blobs (loader zero-pads) → `cat_active` false →
    /// not even computed, so the live net is unaffected. A retrained blob carries
    /// learned weights → `cat_active` true → contributes.
    pub cat_raw_me: Vec<f64>,
    pub cat_raw_opp: Vec<f64>,
    pub cat_propagated_me: Vec<f64>,
    pub cat_propagated_opp: Vec<f64>,
    pub cat_propagated_combined: Vec<f64>,
    pub cat_active: bool,
    /// Distance-field plane weights (81 each, h-independent, side-to-move
    /// canonical). Added as scalar features to the eval output, like the route
    /// planes. These give the net exact per-cell BFS distances instead of the
    /// lossy binarized route planes.
    ///   dist_me:    own per-cell distance to goal / 20.0
    ///   dist_opp:   opponent per-cell distance to goal / 20.0
    ///   dist_diff:  (d_opp - d_own) / 20.0, clamped [-1,1], broadcast
    pub dist_me: Vec<f64>,
    pub dist_opp: Vec<f64>,
    pub dist_diff: Vec<f64>,
    pub dist_field_active: bool,
    /// Side-to-move canonicalized weight tables: `[turn][cell]` so the hot
    /// path indexes by raw cell without re-applying NET_MIRC each time.
    pub dist_me_canon: Box<[[f64; 81]; 2]>,
    pub dist_opp_canon: Box<[[f64; 81]; 2]>,
    pub dist_diff_canon: Box<[[f64; 81]; 2]>,
}

fn read_f64s(bytes: &[u8], offset: &mut usize, count: usize) -> Vec<f64> {
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        let chunk: [u8; 8] = bytes[*offset..*offset + 8].try_into().unwrap();
        out.push(f64::from_le_bytes(chunk));
        *offset += 8;
    }
    out
}

fn read_h_header(bytes: &[u8]) -> usize {
    assert!(
        bytes.len() >= H_HEADER_LEN,
        "net_weights blob too short to hold NET_H header"
    );
    let chunk: [u8; 8] = bytes[0..H_HEADER_LEN].try_into().unwrap();
    let h = u64::from_le_bytes(chunk) as usize;
    assert!(
        h > 0 && h <= MAX_NET_H,
        "net_weights NET_H header = {h}, out of range (1..={MAX_NET_H}); \
         bump titanium::net::MAX_NET_H and rebuild if this is intentional"
    );
    h
}

/// Marker for the section-table blob format, at bytes 4..8.
///
/// Safe as a discriminator because the legacy header is a u64 LE holding `h`,
/// and `h <= MAX_NET_H = 256`, so bytes 1..8 are zero in every legacy blob.
const TLV_MAGIC: &[u8; 4] = b"TNW1";

/// Section tags. Four ASCII bytes, matched exactly.
///
/// `b"ROUT"` is retired and deliberately absent: the five route planes were
/// learned in v13/v17 but zeroed in every net shipped since, and the eval that
/// consumed them is gone. A blob that still carries the section loads fine --
/// an unrecognised tag takes the ignore path, which is exactly the intent.
mod tag {
    pub const WSKP: u32 = u32::from_le_bytes(*b"WSKP");
    pub const B1__: u32 = u32::from_le_bytes(*b"B1__");
    pub const W2__: u32 = u32::from_le_bytes(*b"W2__");
    pub const W1C_: u32 = u32::from_le_bytes(*b"W1C_");
    pub const PO__: u32 = u32::from_le_bytes(*b"PO__");
    pub const PX__: u32 = u32::from_le_bytes(*b"PX__");
    pub const WH__: u32 = u32::from_le_bytes(*b"WH__");
    pub const CATV: u32 = u32::from_le_bytes(*b"CATV");
    pub const DIST: u32 = u32::from_le_bytes(*b"DIST");
}

/// Build the hot-path derived tables and assemble a `Net`.
///
/// Shared by both blob formats on purpose: the section-table reader and the
/// legacy length-matcher must produce byte-identical nets, and the only way to
/// guarantee that is for them to converge before anything is derived.
#[allow(clippy::too_many_arguments)]
fn assemble(
    h: usize,
    ws_v: Vec<f64>,
    b1_v: Vec<f64>,
    w2_v: Vec<f64>,
    w1c: Vec<f64>,
    po: Vec<f64>,
    px: Vec<f64>,
    wh: Vec<f64>,
    cat: [Vec<f64>; 5],
    dist: [Vec<f64>; 3],
) -> Net {
    let [cat_raw_me, cat_raw_opp, cat_propagated_me, cat_propagated_opp, cat_propagated_combined] =
        cat;
    let [dist_me, dist_opp, dist_diff] = dist;

    // Presence is derived from the weights, not from the blob length: a section
    // that is present but all-zero is as inert as an absent one, and the
    // consumers skip the whole (expensive) feature computation on a false flag.
    let wh_active = wh.iter().any(|&w| w != 0.0);
    let cat_active = cat_raw_me
        .iter()
        .chain(&cat_raw_opp)
        .chain(&cat_propagated_me)
        .chain(&cat_propagated_opp)
        .chain(&cat_propagated_combined)
        .any(|&w| w != 0.0);
    let dist_field_active = dist_me
        .iter()
        .chain(&dist_opp)
        .chain(&dist_diff)
        .any(|&w| w != 0.0);

    // Pre-compute side-to-move canonicalized weight tables for the hot path.
    let mut dist_me_canon = Box::new([[0.0f64; 81]; 2]);
    let mut dist_opp_canon = Box::new([[0.0f64; 81]; 2]);
    let mut dist_diff_canon = Box::new([[0.0f64; 81]; 2]);
    for turn in 0..2usize {
        for sq in 0..FIELD_PLANE_LEN {
            let canon = if turn == 0 { sq } else { NET_MIRC[sq] };
            dist_me_canon[turn][sq] = dist_me[canon];
            dist_opp_canon[turn][sq] = dist_opp[canon];
            dist_diff_canon[turn][sq] = dist_diff[canon];
        }
    }
    let mut b1 = [0.0f64; MAX_NET_H];
    let mut w2 = [0.0f64; MAX_NET_H];
    b1[..h].copy_from_slice(&b1_v);
    w2[..h].copy_from_slice(&w2_v);
    Net {
        h,
        ws: ws_v.try_into().unwrap(),
        b1,
        w2,
        w1c,
        po,
        px,
        wh,
        wh_active,
        cat_raw_me,
        cat_raw_opp,
        cat_propagated_me,
        cat_propagated_opp,
        cat_propagated_combined,
        cat_active,
        dist_me,
        dist_opp,
        dist_diff,
        dist_field_active,
        dist_me_canon,
        dist_opp_canon,
        dist_diff_canon,
    }
}

fn load_net_from_bytes(bytes: &[u8]) -> Net {
    if bytes.len() >= H_HEADER_LEN && &bytes[4..8] == TLV_MAGIC {
        return load_net_tlv(bytes);
    }
    load_net_legacy(bytes)
}

/// Section-table format.
///
///   0..4    u32 LE  NET_H
///   4..8    magic   b"TNW1"
///   8..12   u32 LE  section count
///   12..    directory of {u32 tag, u64 byte_len}, payloads follow in order
///
/// Missing sections are zero-filled and unknown tags are ignored, so a blob
/// written by a newer trainer still loads and an older blob still works. That
/// replaces a length enumeration that had grown to eight accepted sizes -- of
/// which only two were ever shipped -- duplicated across three files and
/// already out of sync between the Rust and Python sides.
fn load_net_tlv(bytes: &[u8]) -> Net {
    let bad = |m: String| -> ! { panic!("net_weights TLV blob: {m}") };
    if bytes.len() < 12 {
        bad("shorter than its header".into());
    }
    let h = u32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    if h == 0 || h > MAX_NET_H {
        bad(format!("NET_H = {h}, out of range (1..={MAX_NET_H})"));
    }
    let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let dir_len = count
        .checked_mul(12)
        .unwrap_or_else(|| bad("section count overflows".into()));
    let body = 12usize
        .checked_add(dir_len)
        .unwrap_or_else(|| bad("directory overflows".into()));
    if bytes.len() < body {
        bad(format!("truncated directory ({count} sections)"));
    }

    // tag -> payload slice
    let mut found: Vec<(u32, &[u8])> = Vec::with_capacity(count);
    let mut off = body;
    for i in 0..count {
        let e = 12 + i * 12;
        let t = u32::from_le_bytes(bytes[e..e + 4].try_into().unwrap());
        let len = u64::from_le_bytes(bytes[e + 4..e + 12].try_into().unwrap()) as usize;
        let end = off
            .checked_add(len)
            .unwrap_or_else(|| bad(format!("section {i} length overflows")));
        if end > bytes.len() {
            bad(format!("section {i} runs past end of blob"));
        }
        found.push((t, &bytes[off..end]));
        off = end;
    }

    // Required sections error rather than zero-fill: a net with no `w1c` is not
    // a net with an inert feature, it is a corrupt file.
    let need = |t: u32, n: usize, name: &str| -> Vec<f64> {
        let Some(&(_, sl)) = found.iter().find(|(tt, _)| *tt == t) else {
            bad(format!("missing required section {name}"));
        };
        if sl.len() != n * 8 {
            bad(format!(
                "section {name}: {} bytes, expected {} for NET_H={h}",
                sl.len(),
                n * 8
            ));
        }
        let mut v = Vec::with_capacity(n);
        for c in sl.chunks_exact(8) {
            v.push(f64::from_le_bytes(c.try_into().unwrap()));
        }
        v
    };
    // Optional: absent -> zeros, which the `*_active` flags then read as inert.
    let opt = |t: u32, planes: usize| -> Vec<Vec<f64>> {
        let n = FIELD_PLANE_LEN;
        match found.iter().find(|(tt, _)| *tt == t) {
            Some(&(_, sl)) if sl.len() == planes * n * 8 => sl
                .chunks_exact(n * 8)
                .map(|p| {
                    p.chunks_exact(8)
                        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
                        .collect()
                })
                .collect(),
            Some(&(_, sl)) => bad(format!(
                "optional section has {} bytes, expected {}",
                sl.len(),
                planes * n * 8
            )),
            None => (0..planes).map(|_| vec![0.0; n]).collect(),
        }
    };

    let ws_v = need(tag::WSKP, WSKIP_LEN, "WSKP");
    let b1_v = need(tag::B1__, h, "B1__");
    let w2_v = need(tag::W2__, h, "W2__");
    let w1c = need(tag::W1C_, 9 * 128 * h, "W1C_");
    let po = need(tag::PO__, 81 * h, "PO__");
    let px = need(tag::PX__, 81 * h, "PX__");
    // Optional and h-dependent, so it cannot go through `opt` (which is fixed
    // at one 81-cell plane per entry). Absent -> zeros -> `wh_active` false.
    let wh = match found.iter().find(|(tt, _)| *tt == tag::WH__) {
        Some(&(_, sl)) if sl.len() == WH_PAIRS * h * 8 => sl
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect(),
        Some(&(_, sl)) => bad(format!(
            "section WH__: {} bytes, expected {} for NET_H={h}",
            sl.len(),
            WH_PAIRS * h * 8
        )),
        None => vec![0.0; WH_PAIRS * h],
    };
    let to5 = |v: Vec<Vec<f64>>| -> [Vec<f64>; 5] { v.try_into().unwrap() };
    let to3 = |v: Vec<Vec<f64>>| -> [Vec<f64>; 3] { v.try_into().unwrap() };
    // CAT is stored already-rescaled in this format. The legacy loader applies
    // historic x4 / x400/256 factors while reading; carrying those into a new
    // format would preserve an accident of the old one.
    let cat = to5(opt(tag::CATV, 5));
    let dist = to3(opt(tag::DIST, 3));

    assemble(h, ws_v, b1_v, w2_v, w1c, po, px, wh, cat, dist)
}

/// Serialize a net in the section-table format.
pub fn net_to_tlv(n: &Net) -> Vec<u8> {
    let mut sections: Vec<(u32, Vec<f64>)> = Vec::new();
    let flat = |ps: [&Vec<f64>; 5]| -> Vec<f64> { ps.iter().flat_map(|p| p.iter().copied()).collect() };
    sections.push((tag::WSKP, n.ws.to_vec()));
    sections.push((tag::B1__, n.b1[..n.h].to_vec()));
    sections.push((tag::W2__, n.w2[..n.h].to_vec()));
    sections.push((tag::W1C_, n.w1c.clone()));
    sections.push((tag::PO__, n.po.clone()));
    sections.push((tag::PX__, n.px.clone()));
    sections.push((tag::WH__, n.wh.clone()));
    sections.push((
        tag::CATV,
        flat([
            &n.cat_raw_me,
            &n.cat_raw_opp,
            &n.cat_propagated_me,
            &n.cat_propagated_opp,
            &n.cat_propagated_combined,
        ]),
    ));
    sections.push((
        tag::DIST,
        n.dist_me
            .iter()
            .chain(&n.dist_opp)
            .chain(&n.dist_diff)
            .copied()
            .collect(),
    ));

    let mut out = Vec::new();
    out.extend_from_slice(&(n.h as u32).to_le_bytes());
    out.extend_from_slice(TLV_MAGIC);
    out.extend_from_slice(&(sections.len() as u32).to_le_bytes());
    for (t, v) in &sections {
        out.extend_from_slice(&t.to_le_bytes());
        out.extend_from_slice(&((v.len() * 8) as u64).to_le_bytes());
    }
    for (_, v) in &sections {
        for x in v {
            out.extend_from_slice(&x.to_le_bytes());
        }
    }
    out
}

fn load_net_legacy(bytes: &[u8]) -> Net {
    let h = read_h_header(bytes);
    let mut offset = H_HEADER_LEN;

    // Accept the legacy blob (5 route planes) OR the retraining-ready blob that
    // additionally carries the `cat_heat` plane. Legacy → cat_heat zero-padded.
    let payload_f64s_no_cat =
        WSKIP_LEN + h + h + 9 * 128 * h + 81 * h + 81 * h + FIELD_PLANE_LEN * FIELD_PLANE_SETS;
    let expected_no_cat = H_HEADER_LEN + payload_f64s_no_cat * 8;
    let expected_cat_v5 = expected_no_cat + FIELD_PLANE_LEN * 8;
    let expected_cat_v5_witness = expected_no_cat + FIELD_PLANE_LEN * 3 * 8;
    let expected_cat_v5_normalized = expected_no_cat + FIELD_PLANE_LEN * 5 * 8;
    // Distance-field extension: 3 additional field planes (dist_me, dist_opp,
    // dist_diff), each 81 f64s. Can be combined with any of the above variants.
    let dist_field_extra = FIELD_PLANE_LEN * 3 * 8;
    let expected_no_cat_dist = expected_no_cat + dist_field_extra;
    let expected_cat_v5_dist = expected_cat_v5 + dist_field_extra;
    let expected_cat_v5_witness_dist = expected_cat_v5_witness + dist_field_extra;
    let expected_cat_v5_normalized_dist = expected_cat_v5_normalized + dist_field_extra;

    let has_cat_v5 = bytes.len() == expected_cat_v5;
    let has_cat_v5_witness = bytes.len() == expected_cat_v5_witness;
    let has_cat_v5_normalized = bytes.len() == expected_cat_v5_normalized;
    let has_dist_only = bytes.len() == expected_no_cat_dist;
    let has_cat_v5_dist = bytes.len() == expected_cat_v5_dist;
    let has_cat_v5_witness_dist = bytes.len() == expected_cat_v5_witness_dist;
    let has_cat_v5_normalized_dist = bytes.len() == expected_cat_v5_normalized_dist;
    let has_dist = has_dist_only || has_cat_v5_dist || has_cat_v5_witness_dist || has_cat_v5_normalized_dist;

    assert!(
        bytes.len() == expected_no_cat || has_cat_v5 || has_cat_v5_witness || has_cat_v5_normalized
        || has_dist_only || has_cat_v5_dist || has_cat_v5_witness_dist || has_cat_v5_normalized_dist,
        "net_weights blob size mismatch for declared NET_H={h} \
         (got {} bytes, expected {expected_no_cat}, {expected_cat_v5}, or {expected_cat_v5_witness}) — \
         run training/freeze_baseline_weights.py",
        bytes.len()
    );

    let ws_v = read_f64s(bytes, &mut offset, WSKIP_LEN);
    let b1_v = read_f64s(bytes, &mut offset, h);
    let w2_v = read_f64s(bytes, &mut offset, h);
    let w1c = read_f64s(bytes, &mut offset, 9 * 128 * h);
    let po = read_f64s(bytes, &mut offset, 81 * h);
    let px = read_f64s(bytes, &mut offset, 81 * h);
    // The five retired route planes sit at a FIXED offset here, between `px` and
    // the CAT tail. They are no longer read into the net, but the bytes must
    // still be stepped over or every section after them decodes from the wrong
    // place. Skipping is not the same as deleting in a positional format.
    offset += FIELD_PLANE_LEN * FIELD_PLANE_SETS * 8;
    let (cat_raw_me, cat_raw_opp, cat_propagated_me, cat_propagated_opp, cat_propagated_combined) =
        if has_cat_v5_normalized || has_cat_v5_normalized_dist {
            (
                read_f64s(bytes, &mut offset, FIELD_PLANE_LEN),
                read_f64s(bytes, &mut offset, FIELD_PLANE_LEN),
                read_f64s(bytes, &mut offset, FIELD_PLANE_LEN),
                read_f64s(bytes, &mut offset, FIELD_PLANE_LEN),
                read_f64s(bytes, &mut offset, FIELD_PLANE_LEN),
            )
        } else if has_cat_v5_witness || has_cat_v5_witness_dist {
            let mut raw_me = read_f64s(bytes, &mut offset, FIELD_PLANE_LEN);
            let mut raw_opp = read_f64s(bytes, &mut offset, FIELD_PLANE_LEN);
            let mut combined = read_f64s(bytes, &mut offset, FIELD_PLANE_LEN);
            for w in &mut raw_me {
                *w *= 4.0;
            }
            for w in &mut raw_opp {
                *w *= 4.0;
            }
            for w in &mut combined {
                *w *= 400.0 / 256.0;
            }
            (
                raw_me,
                raw_opp,
                vec![0.0; FIELD_PLANE_LEN],
                vec![0.0; FIELD_PLANE_LEN],
                combined,
            )
        } else if has_cat_v5 || has_cat_v5_dist {
            let mut combined = read_f64s(bytes, &mut offset, FIELD_PLANE_LEN);
            for w in &mut combined {
                *w *= 400.0 / 256.0;
            }
            (
                vec![0.0; FIELD_PLANE_LEN],
                vec![0.0; FIELD_PLANE_LEN],
                vec![0.0; FIELD_PLANE_LEN],
                vec![0.0; FIELD_PLANE_LEN],
                combined,
            )
        } else {
            (
                vec![0.0; FIELD_PLANE_LEN],
                vec![0.0; FIELD_PLANE_LEN],
                vec![0.0; FIELD_PLANE_LEN],
                vec![0.0; FIELD_PLANE_LEN],
                vec![0.0; FIELD_PLANE_LEN],
            )
        };
    // Distance-field extension weights (zero-padded if blob doesn't carry them)
    let (dist_me, dist_opp, dist_diff) = if has_dist {
        (
            read_f64s(bytes, &mut offset, FIELD_PLANE_LEN),
            read_f64s(bytes, &mut offset, FIELD_PLANE_LEN),
            read_f64s(bytes, &mut offset, FIELD_PLANE_LEN),
        )
    } else {
        (
            vec![0.0; FIELD_PLANE_LEN],
            vec![0.0; FIELD_PLANE_LEN],
            vec![0.0; FIELD_PLANE_LEN],
        )
    };
    assemble(
        h,
        ws_v,
        b1_v,
        w2_v,
        w1c,
        po,
        px,
        // The positional format predates this input and has nowhere to put it.
        // Zeros keep every shipped blob evaluating exactly as before.
        vec![0.0; WH_PAIRS * h],
        [
            cat_raw_me,
            cat_raw_opp,
            cat_propagated_me,
            cat_propagated_opp,
            cat_propagated_combined,
        ],
        [dist_me, dist_opp, dist_diff],
    )
}

/// Training / deployed weights (`net_weights.bin`, overridable via `TITANIUM_NET_WEIGHTS_PATH`).
pub fn net() -> &'static Net {
    static NET: OnceLock<Net> = OnceLock::new();
    NET.get_or_init(|| {
        if let Ok(path) = std::env::var("TITANIUM_NET_WEIGHTS_PATH") {
            let bytes = std::fs::read(&path)
                .unwrap_or_else(|e| panic!("TITANIUM_NET_WEIGHTS_PATH read failed ({path}): {e}"));
            load_net_from_bytes(&bytes)
        } else {
            load_net_from_bytes(NET_BYTES)
        }
    })
}
/// Original v13 baseline — same search as v15, frozen HalfPW (`net_weights_frozen.bin`).
pub fn net_frozen() -> &'static Net {
    static NET: OnceLock<Net> = OnceLock::new();
    NET.get_or_init(|| load_net_from_bytes(NET_FROZEN_BYTES))
}

/// Legacy website Titanium v17 weights — frozen for v17-vs-v18 comparison.
pub fn net_v17() -> &'static Net {
    static NET: OnceLock<Net> = OnceLock::new();
    NET.get_or_init(|| load_net_from_bytes(NET_V17_BYTES))
}

pub fn v17_weights_sha256() -> [u8; 32] {
    Sha256::digest(NET_V17_BYTES).into()
}

pub fn live_weights_sha256() -> [u8; 32] {
    Sha256::digest(NET_BYTES).into()
}

pub fn frozen_weights_sha256() -> [u8; 32] {
    Sha256::digest(NET_FROZEN_BYTES).into()
}

static NET_MEDIUM: OnceLock<Net> = OnceLock::new();

/// Runtime medium-tier weights (fetched by the browser worker).
pub fn install_medium_weights(bytes: &[u8]) -> Result<(), &'static str> {
    if bytes.len() < H_HEADER_LEN {
        return Err("medium weights too short for NET_H header");
    }
    let h = u64::from_le_bytes(bytes[0..H_HEADER_LEN].try_into().unwrap()) as usize;
    if h == 0 || h > MAX_NET_H {
        return Err("medium weights NET_H header out of range");
    }
    let payload_f64s_no_cat =
        WSKIP_LEN + h + h + 9 * 128 * h + 81 * h + 81 * h + FIELD_PLANE_LEN * FIELD_PLANE_SETS;
    let expected_no_cat = H_HEADER_LEN + payload_f64s_no_cat * 8;
    let expected_cat_v5 = expected_no_cat + FIELD_PLANE_LEN * 8;
    let expected_cat_v5_witness = expected_no_cat + FIELD_PLANE_LEN * 3 * 8;
    let expected_cat_v5_normalized = expected_no_cat + FIELD_PLANE_LEN * 5 * 8;
    let dist_field_extra = FIELD_PLANE_LEN * 3 * 8;
    if bytes.len() != expected_no_cat
        && bytes.len() != expected_cat_v5
        && bytes.len() != expected_cat_v5_witness
        && bytes.len() != expected_cat_v5_normalized
        && bytes.len() != expected_no_cat + dist_field_extra
        && bytes.len() != expected_cat_v5 + dist_field_extra
        && bytes.len() != expected_cat_v5_witness + dist_field_extra
        && bytes.len() != expected_cat_v5_normalized + dist_field_extra
    {
        return Err("medium weights size mismatch");
    }
    let net = load_net_from_bytes(bytes);
    NET_MEDIUM
        .set(net)
        .map_err(|_| "medium weights already installed")
}

pub fn net_medium() -> Option<&'static Net> {
    if let Some(net) = NET_MEDIUM.get() {
        return Some(net);
    }
    static NET_BUILTIN_MEDIUM: OnceLock<Net> = OnceLock::new();
    Some(NET_BUILTIN_MEDIUM.get_or_init(|| load_net_from_bytes(NET_MEDIUM_BYTES)))
}
// ── Side-to-move canonicalization tables ─────────────────────────────────────
// P2 positions are rotated 180 degrees so the mover always advances toward
// canonical row 8. Both row and column must reverse; a row-only reflection
// makes role-swapped positions encode differently.
const fn build_mirc() -> [usize; 81] {
    let mut arr = [0usize; 81];
    let mut i = 0;
    while i < 81 {
        arr[i] = (8 - i / 9) * 9 + (8 - i % 9);
        i += 1;
    }
    arr
}
const fn build_mirs() -> [usize; 64] {
    let mut arr = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        arr[i] = (7 - i / 8) * 8 + (7 - i % 8);
        i += 1;
    }
    arr
}
const fn build_bkt() -> [usize; 81] {
    let mut arr = [0usize; 81];
    let mut i = 0;
    while i < 81 {
        arr[i] = (i / 9 / 3) * 3 + (i % 9) / 3;
        i += 1;
    }
    arr
}
pub static NET_MIRC: [usize; 81] = build_mirc();
pub static NET_MIRS: [usize; 64] = build_mirs();
pub static NET_BKT: [usize; 81] = build_bkt();

#[cfg(test)]
mod tlv_tests {
    use super::*;

    pub(super) fn shipped() -> Vec<(&'static str, &'static [u8])> {
        vec![
            ("net_weights.bin", NET_BYTES),
            ("net_weights_frozen.bin", NET_FROZEN_BYTES),
            ("net_weights_medium.bin", NET_MEDIUM_BYTES),
            ("net_weights_v17.bin", NET_V17_BYTES),
        ]
    }

    fn assert_same(name: &str, a: &Net, b: &Net) {
        assert_eq!(a.h, b.h, "{name}: h");
        assert_eq!(a.ws, b.ws, "{name}: ws");
        assert_eq!(a.b1, b.b1, "{name}: b1");
        assert_eq!(a.w2, b.w2, "{name}: w2");
        assert_eq!(a.w1c, b.w1c, "{name}: w1c");
        assert_eq!(a.po, b.po, "{name}: po");
        assert_eq!(a.px, b.px, "{name}: px");
        assert_eq!(a.wh, b.wh, "{name}: wh");
        assert_eq!(a.wh_active, b.wh_active, "{name}: wh_active");
        assert_eq!(a.cat_raw_me, b.cat_raw_me, "{name}: cat_raw_me");
        assert_eq!(a.cat_raw_opp, b.cat_raw_opp, "{name}: cat_raw_opp");
        assert_eq!(a.cat_propagated_me, b.cat_propagated_me, "{name}: cat_prop_me");
        assert_eq!(a.cat_propagated_opp, b.cat_propagated_opp, "{name}: cat_prop_opp");
        assert_eq!(
            a.cat_propagated_combined, b.cat_propagated_combined,
            "{name}: cat_prop_combined"
        );
        assert_eq!(a.dist_me, b.dist_me, "{name}: dist_me");
        assert_eq!(a.dist_opp, b.dist_opp, "{name}: dist_opp");
        assert_eq!(a.dist_diff, b.dist_diff, "{name}: dist_diff");
        // Derived tables and presence flags must agree too, or the hot path
        // would differ while the raw weights matched.
        assert_eq!(a.cat_active, b.cat_active, "{name}: cat_active");
        assert_eq!(a.dist_field_active, b.dist_field_active, "{name}: dist_active");
        assert_eq!(a.dist_me_canon, b.dist_me_canon, "{name}: dist_me_canon");
        assert_eq!(a.dist_opp_canon, b.dist_opp_canon, "{name}: dist_opp_canon");
        assert_eq!(a.dist_diff_canon, b.dist_diff_canon, "{name}: dist_diff_canon");
    }

    /// Every shipped blob must survive legacy -> TLV -> legacy-equivalent with
    /// bit-identical f64s, including the derived hot-path tables.
    ///
    /// This is the whole justification for the format change being callable a
    /// no-op. Comparing raw weights alone would not do it: the `*_canon` tables
    /// are what the hot path actually reads.
    #[test]
    fn every_shipped_blob_roundtrips_through_tlv() {
        for (name, bytes) in shipped() {
            let legacy = load_net_from_bytes(bytes);
            let tlv_bytes = net_to_tlv(&legacy);
            assert_eq!(
                &tlv_bytes[4..8],
                TLV_MAGIC,
                "{name}: written blob must carry the TLV magic"
            );
            let back = load_net_from_bytes(&tlv_bytes);
            assert_same(name, &legacy, &back);
        }
    }

    /// The dispatcher must still route legacy blobs to the legacy reader. A
    /// magic check that accidentally matched would silently reinterpret every
    /// shipped net.
    #[test]
    fn shipped_blobs_are_not_mistaken_for_tlv() {
        for (name, bytes) in shipped() {
            assert_ne!(&bytes[4..8], TLV_MAGIC, "{name} must not look like TLV");
        }
    }

    /// Unknown sections are ignored and absent optional sections zero-fill.
    /// Together these are what let a newer trainer add a tail without a Rust
    /// change, and an older blob keep loading after one.
    #[test]
    fn tlv_tolerates_unknown_and_missing_sections() {
        let base = load_net_from_bytes(NET_BYTES);
        let mut bytes = net_to_tlv(&base);

        // Append an unknown section: bump the count, add a directory entry, and
        // put its payload at the very end.
        let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let payload_start = 12 + count as usize * 12;
        let junk: Vec<u8> = (0..64u8).collect();
        let mut out = Vec::new();
        out.extend_from_slice(&bytes[0..8]);
        out.extend_from_slice(&(count + 1).to_le_bytes());
        out.extend_from_slice(&bytes[12..payload_start]);
        out.extend_from_slice(&u32::from_le_bytes(*b"ZZZZ").to_le_bytes());
        out.extend_from_slice(&(junk.len() as u64).to_le_bytes());
        out.extend_from_slice(&bytes[payload_start..]);
        out.extend_from_slice(&junk);
        let with_junk = load_net_from_bytes(&out);
        assert_same("unknown-section", &base, &with_junk);

        // Drop the optional DIST section entirely -> zero-filled, inert.
        bytes = net_to_tlv(&base);
        let count = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let mut keep: Vec<(u32, Vec<u8>)> = Vec::new();
        let mut off = 12 + count * 12;
        for i in 0..count {
            let e = 12 + i * 12;
            let t = u32::from_le_bytes(bytes[e..e + 4].try_into().unwrap());
            let len = u64::from_le_bytes(bytes[e + 4..e + 12].try_into().unwrap()) as usize;
            if t != tag::DIST {
                keep.push((t, bytes[off..off + len].to_vec()));
            }
            off += len;
        }
        let mut out = Vec::new();
        out.extend_from_slice(&bytes[0..8]);
        out.extend_from_slice(&(keep.len() as u32).to_le_bytes());
        for (t, p) in &keep {
            out.extend_from_slice(&t.to_le_bytes());
            out.extend_from_slice(&((p.len()) as u64).to_le_bytes());
        }
        for (_, p) in &keep {
            out.extend_from_slice(p);
        }
        let without_dist = load_net_from_bytes(&out);
        assert!(
            !without_dist.dist_field_active,
            "absent DIST must be inert, not merely zero"
        );
        assert_eq!(without_dist.w1c, base.w1c, "dropping DIST disturbed w1c");
    }
}

#[cfg(test)]
mod walls_in_hand_tests {
    use super::tlv_tests::shipped;
    use super::*;

    /// The input exists to separate hand pairs that every other input encodes
    /// identically. If the encoding is not injective it has bought nothing.
    #[test]
    fn every_hand_pair_gets_its_own_row() {
        let mut seen = vec![usize::MAX; WH_PAIRS];
        for me in 0..=10 {
            for opp in 0..=10 {
                let i = wh_index(me, opp);
                assert!(i < WH_PAIRS, "({me},{opp}) -> {i} out of range");
                assert_eq!(
                    seen[i],
                    usize::MAX,
                    "({me},{opp}) collides with pair #{}",
                    seen[i]
                );
                seen[i] = me * 11 + opp;
            }
        }
        assert!(seen.iter().all(|&v| v != usize::MAX), "left rows unused");
    }

    /// The specific failure that motivated the input: ten walls to nil is a very
    /// different position from nil to ten, and the old net saw one thing.
    #[test]
    fn ten_to_nil_differs_from_nil_to_ten() {
        assert_ne!(
            wh_index(10, 0),
            wh_index(0, 10),
            "swapped hands must not share a row"
        );
    }

    /// Out-of-range hands clamp instead of indexing past the embedding.
    #[test]
    fn oversized_hands_stay_in_bounds() {
        assert!(wh_index(99, 99) < WH_PAIRS);
        assert_eq!(wh_index(99, 99), wh_index(10, 10));
    }

    /// Shipped blobs predate the section, so they must load with it inert --
    /// otherwise this change is not the no-op the node-identity run claims.
    #[test]
    fn shipped_blobs_carry_no_walls_in_hand_weights() {
        for (name, bytes) in shipped() {
            let n = load_net_from_bytes(bytes);
            assert_eq!(n.wh.len(), WH_PAIRS * n.h, "{name}: wh wrong length");
            assert!(!n.wh_active, "{name}: legacy blob must be inert here");
            assert!(n.wh.iter().all(|&w| w == 0.0), "{name}: wh not zeroed");
        }
    }

    /// A blob that does carry the section must survive the round trip with the
    /// weights intact and the presence flag flipped.
    #[test]
    fn walls_in_hand_survives_the_tlv_round_trip() {
        let mut n = load_net_from_bytes(NET_BYTES);
        for (i, w) in n.wh.iter_mut().enumerate() {
            *w = (i as f64) * 1e-4 - 0.5;
        }
        n.wh_active = true;
        let round = load_net_tlv(&net_to_tlv(&n));
        assert_eq!(round.wh, n.wh, "wh weights did not round-trip");
        assert!(round.wh_active, "wh_active lost in the round trip");
    }
}
