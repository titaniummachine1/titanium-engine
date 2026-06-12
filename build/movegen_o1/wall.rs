//! Wall O(1) tables — per-slot L2 collision gate (`WALL_PHYSICAL_TABLE`).

use std::collections::HashMap;

use super::geometry::{can_wall_block_topology, set_wall, wall_physically_legal, MiniBoard};

pub const WALL_PSEUDO_KEYS: usize = 1024;
pub const MAX_PSEUDO_PROBES: usize = 16;
pub const PHYS_PSEUDO_COMBOS: usize = 1 << MAX_PSEUDO_PROBES;

pub const WALL_TOPO_KEYS: usize = 1024;
pub const MAX_TOPO_PROBES: usize = 10;
pub const PHYS_TOPO_COMBOS: usize = 1 << MAX_TOPO_PROBES;

#[derive(Debug)]
pub struct WallSlotMeta {
    pub row: u8,
    pub col: u8,
    pub horizontal: bool,
    pub collision_mask: u8,
    pub table: [bool; 256],
}

#[derive(Debug)]
pub struct WallPseudoMeta {
    pub h_probes: Vec<(u8, u8, bool)>,
    pub v_probes: Vec<(u8, u8, bool)>,
    pub h_key_count: u16,
    pub v_key_count: u16,
    pub h_table: [u64; WALL_PSEUDO_KEYS],
    pub v_table: [u64; WALL_PSEUDO_KEYS],
    pub h_remap: Vec<u8>,
    pub v_remap: Vec<u8>,
}

#[derive(Debug)]
pub struct WallTopoMeta {
    pub h_probes: Vec<(u8, u8, bool)>,
    pub v_probes: Vec<(u8, u8, bool)>,
    pub h_key_count: u16,
    pub v_key_count: u16,
    /// Bit set ⇒ `can_wall_block_topology` (needs parallel flood before accepting).
    pub h_table: [u64; WALL_TOPO_KEYS],
    pub v_table: [u64; WALL_TOPO_KEYS],
    pub h_remap: Vec<u8>,
    pub v_remap: Vec<u8>,
}

pub fn discover_all_wall_tables(
    bar: &super::progress::PhaseBar,
) -> ([WallSlotMeta; 128], WallPseudoMeta, WallTopoMeta) {
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
    let slots: [WallSlotMeta; 128] = out.try_into().unwrap();
    let pseudo = stub_wall_pseudo_global();
    let topo = discover_wall_topo_global(bar);
    (slots, pseudo, topo)
}

fn stub_wall_pseudo_global() -> WallPseudoMeta {
    let mut h_table = [0u64; WALL_PSEUDO_KEYS];
    h_table[0] = u64::MAX;
    let mut v_table = [0u64; WALL_PSEUDO_KEYS];
    v_table[0] = u64::MAX;
    WallPseudoMeta {
        h_probes: Vec::new(),
        v_probes: Vec::new(),
        h_key_count: 1,
        v_key_count: 1,
        h_table,
        v_table,
        h_remap: vec![0],
        v_remap: vec![0],
    }
}

#[allow(dead_code)]
fn discover_wall_pseudo_global(bar: &super::progress::PhaseBar) -> WallPseudoMeta {
    bar.begin(1, "wall pseudo-global [H/V probes]");
    let candidates = all_wall_positions();
    let h_probes = discover_probes_forward_u64(
        &candidates,
        pseudo_h_mask_for_part,
        MAX_PSEUDO_PROBES,
        WALL_PSEUDO_KEYS,
        "pseudo-h",
    );
    let v_probes = discover_probes_forward_u64(
        &candidates,
        pseudo_v_mask_for_part,
        MAX_PSEUDO_PROBES,
        WALL_PSEUDO_KEYS,
        "pseudo-v",
    );
    let h = build_u64_mask_meta(
        &h_probes,
        pseudo_h_mask_for_part,
        WALL_PSEUDO_KEYS,
        "pseudo-h",
    );
    let v = build_u64_mask_meta(
        &v_probes,
        pseudo_v_mask_for_part,
        WALL_PSEUDO_KEYS,
        "pseudo-v",
    );
    bar.finish(&format!(
        "wall pseudo-global done (H {} probes/{} keys, V {} probes/{} keys)",
        h.probes.len(),
        h.key_count,
        v.probes.len(),
        v.key_count
    ));
    WallPseudoMeta {
        h_probes: h.probes,
        v_probes: v.probes,
        h_key_count: h.key_count,
        v_key_count: v.key_count,
        h_table: h.table,
        v_table: v.table,
        h_remap: h.remap,
        v_remap: v.remap,
    }
}

fn discover_wall_topo_global(bar: &super::progress::PhaseBar) -> WallTopoMeta {
    bar.begin(1, "wall topo-global [H/V probes]");
    let candidates = all_wall_positions();
    let h_probes = discover_probes_forward_u64(
        &candidates,
        topo_h_mask_for_part,
        MAX_TOPO_PROBES,
        WALL_TOPO_KEYS,
        "topo-h",
    );
    let v_probes = discover_probes_forward_u64(
        &candidates,
        topo_v_mask_for_part,
        MAX_TOPO_PROBES,
        WALL_TOPO_KEYS,
        "topo-v",
    );
    let h = build_u64_mask_meta(&h_probes, topo_h_mask_for_part, WALL_TOPO_KEYS, "topo-h");
    let v = build_u64_mask_meta(&v_probes, topo_v_mask_for_part, WALL_TOPO_KEYS, "topo-v");
    bar.finish(&format!(
        "wall topo-global done (H {} probes/{} keys, V {} probes/{} keys)",
        h.probes.len(),
        h.key_count,
        v.probes.len(),
        v.key_count
    ));
    WallTopoMeta {
        h_probes: h.probes,
        v_probes: v.probes,
        h_key_count: h.key_count,
        v_key_count: v.key_count,
        h_table: h.table,
        v_table: v.table,
        h_remap: h.remap,
        v_remap: v.remap,
    }
}

struct U64MaskMetaBuilt {
    probes: Vec<(u8, u8, bool)>,
    key_count: u16,
    table: [u64; 1024],
    remap: Vec<u8>,
}

fn build_u64_mask_meta(
    probes: &[(u8, u8, bool)],
    mask_fn: fn(&[(u8, u8, bool)], usize) -> u64,
    max_keys: usize,
    label: &str,
) -> U64MaskMetaBuilt {
    let nw = probes.len();
    if nw > MAX_TOPO_PROBES {
        panic!("{label}: {nw} probes > {}", MAX_TOPO_PROBES);
    }
    if probes.is_empty() {
        let mask = mask_fn(probes, 0);
        let mut table = [0u64; 1024];
        table[0] = mask;
        return U64MaskMetaBuilt {
            probes: Vec::new(),
            key_count: 1,
            table,
            remap: vec![0],
        };
    }
    let phys_combos = 1usize << nw;
    let mut table = [0u64; 1024];
    let mut remap = vec![0u8; phys_combos];
    let mut mask_to_key: HashMap<u64, u16> = HashMap::new();
    let mut next_key = 0u16;

    for phys in 0..phys_combos {
        let mask = mask_fn(probes, phys);
        let key = *mask_to_key.entry(mask).or_insert_with(|| {
            let k = next_key as usize;
            if k >= max_keys {
                panic!("{label}: >{max_keys} distinct masks (nw={nw})");
            }
            table[k] = mask;
            next_key += 1;
            next_key - 1
        });
        remap[phys] = key as u8;
    }

    U64MaskMetaBuilt {
        probes: probes.to_vec(),
        key_count: next_key as u16,
        table,
        remap,
    }
}

/// Greedy forward probe selection — avoids 2^n peel on the full-board union.
fn discover_probes_forward_u64(
    candidates: &[(u8, u8, bool)],
    mask_fn: fn(&[(u8, u8, bool)], usize) -> u64,
    max_probes: usize,
    max_keys: usize,
    label: &str,
) -> Vec<(u8, u8, bool)> {
    let mut probes = Vec::new();
    loop {
        let mut added = false;
        for &cand in candidates {
            if probes.contains(&cand) {
                continue;
            }
            if probes.len() >= max_probes {
                break;
            }
            let mut trial = probes.clone();
            trial.push(cand);
            if mask_tables_equivalent_u64(&probes, &trial, mask_fn) {
                continue;
            }
            let built = build_u64_mask_meta(&trial, mask_fn, max_keys, label);
            if built.key_count as usize >= max_keys {
                continue;
            }
            probes.push(cand);
            added = true;
            break;
        }
        if !added {
            break;
        }
    }
    while !probes.is_empty() {
        let built = build_u64_mask_meta(&probes, mask_fn, max_keys, label);
        if built.key_count as usize <= max_keys {
            break;
        }
        probes.pop();
    }
    probes
}

fn mask_tables_equivalent_u64(
    a: &[(u8, u8, bool)],
    b: &[(u8, u8, bool)],
    mask_fn: fn(&[(u8, u8, bool)], usize) -> u64,
) -> bool {
    debug_assert_eq!(b.len(), a.len() + 1);
    let new_idx = a.len();
    for part_a in 0..(1usize << a.len()) {
        let ma = mask_fn(a, part_a);
        if mask_fn(b, part_a) != ma || mask_fn(b, part_a | (1 << new_idx)) != ma {
            return false;
        }
    }
    true
}

fn all_wall_positions() -> Vec<(u8, u8, bool)> {
    let mut priority = Vec::new();
    let mut rest = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for row in 0..8u8 {
        for col in 0..8u8 {
            for &horizontal in &[true, false] {
                for probe in collision_probe_bits(row, col, horizontal) {
                    if seen.insert(probe) {
                        priority.push(probe);
                    }
                }
            }
        }
    }
    for row in 0..8u8 {
        for col in 0..8u8 {
            for &h in &[true, false] {
                let p = (row, col, h);
                if seen.insert(p) {
                    rest.push(p);
                }
            }
        }
    }
    priority.extend(rest);
    priority
}

fn topo_masks_for_part(probes: &[(u8, u8, bool)], part: usize) -> (u64, u64) {
    let b = board_from_probes(probes, part);
    topo_masks_for_board(&b)
}

fn topo_h_mask_for_part(probes: &[(u8, u8, bool)], part: usize) -> u64 {
    topo_masks_for_part(probes, part).0
}

fn topo_v_mask_for_part(probes: &[(u8, u8, bool)], part: usize) -> u64 {
    topo_masks_for_part(probes, part).1
}

fn topo_masks_for_board(b: &MiniBoard) -> (u64, u64) {
    let mut h = 0u64;
    let mut v = 0u64;
    for row in 0..8u8 {
        for col in 0..8u8 {
            let bit = (row as u64) * 8 + col as u64;
            if can_wall_block_topology(b, row, col, true) {
                h |= 1 << bit;
            }
            if can_wall_block_topology(b, row, col, false) {
                v |= 1 << bit;
            }
        }
    }
    (h, v)
}

fn pseudo_masks_for_part(probes: &[(u8, u8, bool)], part: usize) -> (u64, u64) {
    let b = board_from_probes(probes, part);
    pseudo_masks_for_board(&b)
}

fn pseudo_h_mask_for_part(probes: &[(u8, u8, bool)], part: usize) -> u64 {
    pseudo_masks_for_part(probes, part).0
}

fn pseudo_v_mask_for_part(probes: &[(u8, u8, bool)], part: usize) -> u64 {
    pseudo_masks_for_part(probes, part).1
}

fn pseudo_masks_for_board(b: &MiniBoard) -> (u64, u64) {
    let mut h = 0u64;
    let mut v = 0u64;
    for row in 0..8u8 {
        for col in 0..8u8 {
            let bit = (row as u64) * 8 + col as u64;
            if wall_physically_legal(b, row, col, true) {
                h |= 1 << bit;
            }
            if wall_physically_legal(b, row, col, false) {
                v |= 1 << bit;
            }
        }
    }
    (h, v)
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
        let b = board_from_collision_bits(row, col, horizontal, &probes, local);
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

fn board_from_collision_bits(
    _row: u8,
    _col: u8,
    _horizontal: bool,
    probes: &[(u8, u8, bool)],
    local: u8,
) -> MiniBoard {
    board_from_probes(probes, local as usize)
}
