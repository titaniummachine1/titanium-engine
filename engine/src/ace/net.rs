//! ACE v7 HalfPW net — weights from `net_weights.bin` (H=32 NET_DATA blob).
//!
//! Blob layout (little-endian f64): Wskip[13] B1[32] W2[32] W1C[36864] PO[2592] PX[2592].

use std::sync::OnceLock;

pub const NET_H: usize = 32;
const WSKIP_LEN: usize = 13;
const W1C_LEN: usize = 9 * 128 * NET_H; // 9 pawn buckets × 128 wall inputs × hidden
const PO_LEN: usize = 81 * NET_H;
const PX_LEN: usize = 81 * NET_H;

static NET_BYTES: &[u8] = include_bytes!("net_weights.bin");

pub struct Net {
    pub ws: [f64; WSKIP_LEN],
    pub b1: [f64; NET_H],
    pub w2: [f64; NET_H],
    pub w1c: Vec<f64>,
    pub po: Vec<f64>,
    pub px: Vec<f64>,
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

pub fn net() -> &'static Net {
    static NET: OnceLock<Net> = OnceLock::new();
    NET.get_or_init(|| {
        let total = WSKIP_LEN + NET_H + NET_H + W1C_LEN + PO_LEN + PX_LEN;
        assert_eq!(NET_BYTES.len(), total * 8, "net_weights.bin size mismatch");
        let mut offset = 0;
        let ws_v = read_f64s(NET_BYTES, &mut offset, WSKIP_LEN);
        let b1_v = read_f64s(NET_BYTES, &mut offset, NET_H);
        let w2_v = read_f64s(NET_BYTES, &mut offset, NET_H);
        let w1c = read_f64s(NET_BYTES, &mut offset, W1C_LEN);
        let po = read_f64s(NET_BYTES, &mut offset, PO_LEN);
        let px = read_f64s(NET_BYTES, &mut offset, PX_LEN);
        Net {
            ws: ws_v.try_into().unwrap(),
            b1: b1_v.try_into().unwrap(),
            w2: w2_v.try_into().unwrap(),
            w1c,
            po,
            px,
        }
    })
}

// ── Symmetry tables (match the JS NET_MIRC / NET_MIRS / NET_BKT loops) ────────

const fn build_mirc() -> [usize; 81] {
    let mut arr = [0usize; 81];
    let mut i = 0;
    while i < 81 {
        arr[i] = (8 - i / 9) * 9 + i % 9;
        i += 1;
    }
    arr
}

const fn build_mirs() -> [usize; 64] {
    let mut arr = [0usize; 64];
    let mut i = 0;
    while i < 64 {
        arr[i] = (7 - i / 8) * 8 + i % 8;
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
