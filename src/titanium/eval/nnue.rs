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
    /// Sparse route embeddings, canonicalized to side-to-move coordinates.
    pub route_me: Vec<f64>,
    pub route_opp: Vec<f64>,
    pub route_near_me: Vec<f64>,
    pub route_near_opp: Vec<f64>,
    pub route_contested: Vec<f64>,
    pub route_active: bool,
    /// Route plane weights re-indexed by centered flood bit, per side to move
    /// (`[turn][plane][flood_bit]`, planes: me/opp/near_me/near_opp/contested;
    /// turn 1 pre-applies NET_MIRC). Leaf route scoring then reads `tbl[bit]`
    /// per set bit instead of bit→square→canonical translation each time.
    pub route_bybit: Box<[[[f64; 128]; 5]; 2]>,
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

fn load_net_from_bytes(bytes: &[u8]) -> Net {
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
    let has_cat_v5 = bytes.len() == expected_cat_v5;
    let has_cat_v5_witness = bytes.len() == expected_cat_v5_witness;
    let has_cat_v5_normalized = bytes.len() == expected_cat_v5_normalized;
    assert!(
        bytes.len() == expected_no_cat || has_cat_v5 || has_cat_v5_witness || has_cat_v5_normalized,
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
    let route_me = read_f64s(bytes, &mut offset, FIELD_PLANE_LEN);
    let route_opp = read_f64s(bytes, &mut offset, FIELD_PLANE_LEN);
    let route_near_me = read_f64s(bytes, &mut offset, FIELD_PLANE_LEN);
    let route_near_opp = read_f64s(bytes, &mut offset, FIELD_PLANE_LEN);
    let route_contested = read_f64s(bytes, &mut offset, FIELD_PLANE_LEN);
    let route_active = route_me
        .iter()
        .chain(&route_opp)
        .chain(&route_near_me)
        .chain(&route_near_opp)
        .chain(&route_contested)
        .any(|&w| w != 0.0);
    let (cat_raw_me, cat_raw_opp, cat_propagated_me, cat_propagated_opp, cat_propagated_combined) =
        if has_cat_v5_normalized {
            (
                read_f64s(bytes, &mut offset, FIELD_PLANE_LEN),
                read_f64s(bytes, &mut offset, FIELD_PLANE_LEN),
                read_f64s(bytes, &mut offset, FIELD_PLANE_LEN),
                read_f64s(bytes, &mut offset, FIELD_PLANE_LEN),
                read_f64s(bytes, &mut offset, FIELD_PLANE_LEN),
            )
        } else if has_cat_v5_witness {
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
        } else if has_cat_v5 {
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
    let cat_active = cat_raw_me
        .iter()
        .chain(&cat_raw_opp)
        .chain(&cat_propagated_me)
        .chain(&cat_propagated_opp)
        .chain(&cat_propagated_combined)
        .any(|&w| w != 0.0);
    let mut route_bybit = Box::new([[[0.0f64; 128]; 5]; 2]);
    for turn in 0..2usize {
        for sq in 0..FIELD_PLANE_LEN {
            let canon = if turn == 0 { sq } else { NET_MIRC[sq] };
            let bit = crate::util::grid::FLOOD_BIT_BY_SQ[sq].trailing_zeros() as usize;
            route_bybit[turn][0][bit] = route_me[canon];
            route_bybit[turn][1][bit] = route_opp[canon];
            route_bybit[turn][2][bit] = route_near_me[canon];
            route_bybit[turn][3][bit] = route_near_opp[canon];
            route_bybit[turn][4][bit] = route_contested[canon];
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
        route_me,
        route_opp,
        route_near_me,
        route_near_opp,
        route_contested,
        route_active,
        route_bybit,
        cat_raw_me,
        cat_raw_opp,
        cat_propagated_me,
        cat_propagated_opp,
        cat_propagated_combined,
        cat_active,
    }
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
    if bytes.len() != expected_no_cat
        && bytes.len() != expected_cat_v5
        && bytes.len() != expected_cat_v5_witness
        && bytes.len() != expected_cat_v5_normalized
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
