//! Wall O(1) tables — per-slot L2 collision gate (`WALL_PHYSICAL_TABLE`).
//!
//! Topology flood-skip needs no table: `wall_needs_flood_*_mask` in
//! `movegen::o1::lookup` computes it exactly via bitboard shifts.

use super::geometry::{set_wall, wall_physically_legal, MiniBoard};

#[derive(Debug)]
pub struct WallSlotMeta {
    pub row: u8,
    pub col: u8,
    pub horizontal: bool,
    pub collision_mask: u8,
    pub table: [bool; 256],
}

pub fn discover_all_wall_tables(bar: &super::progress::PhaseBar) -> [WallSlotMeta; 128] {
    bar.begin(128, "wall physical [128 slots]");
    let mut out = Vec::with_capacity(128);
    for hr in 0..8u8 {
        for hc in 0..8u8 {
            out.push(discover_wall_slot(hr, hc, true));
            bar.tick(&format!("H {hr},{hc}"));
        }
    }
    for vr in 0..8u8 {
        for vc in 0..8u8 {
            out.push(discover_wall_slot(vr, vc, false));
            bar.tick(&format!("V {vr},{vc}"));
        }
    }
    bar.finish("wall physical done");
    out.try_into().unwrap()
}

fn board_from_probes(probes: &[(u8, u8, bool)], local: usize) -> MiniBoard {
    let mut b = MiniBoard::default();
    for (i, &(r, c, h)) in probes.iter().enumerate() {
        if (local >> i) & 1 != 0 {
            set_wall(&mut b, r, c, h);
        }
    }
    b
}

fn discover_wall_slot(row: u8, col: u8, horizontal: bool) -> WallSlotMeta {
    let probes = collision_probe_bits(row, col, horizontal);
    let mask = (1u8 << probes.len()) - 1;
    let mut table = [false; 256];

    for local in 0..=mask {
        let b = board_from_probes(&probes, local as usize);
        table[local as usize] = wall_physically_legal(&b, row, col, horizontal);
    }

    WallSlotMeta {
        row,
        col,
        horizontal,
        collision_mask: mask,
        table,
    }
}

/// Bits that can block placement: self H/V, neighbor along wall.
pub fn collision_probe_bits(row: u8, col: u8, horizontal: bool) -> Vec<(u8, u8, bool)> {
    let mut v = vec![(row, col, true), (row, col, false)];
    if horizontal {
        if col > 0 {
            v.push((row, col - 1, true));
        }
        if col < 7 {
            v.push((row, col + 1, true));
        }
    } else {
        if row > 0 {
            v.push((row - 1, col, false));
        }
        if row < 7 {
            v.push((row + 1, col, false));
        }
    }
    v
}
