include!("generated_tables_data.rs");

const PAWN_WALL_REMAP_BYTES: &[u8] = include_bytes!("generated_remap.bin");
const WALL_PSEUDO_H_REMAP_BYTES: &[u8] = include_bytes!("generated_wall_pseudo_h_remap.bin");
const WALL_PSEUDO_V_REMAP_BYTES: &[u8] = include_bytes!("generated_wall_pseudo_v_remap.bin");
const WALL_TOPO_H_REMAP_BYTES: &[u8] = include_bytes!("generated_wall_topo_h_remap.bin");
const WALL_TOPO_V_REMAP_BYTES: &[u8] = include_bytes!("generated_wall_topo_v_remap.bin");

#[inline]
pub fn wall_remap_byte(sq: u8, enemy_key: u8, phys_combo: usize) -> u8 {
    let idx = (sq as usize * 5 + enemy_key as usize) * PHYS_WALL_COMBOS + phys_combo;
    PAWN_WALL_REMAP_BYTES[idx]
}

#[inline]
pub fn wall_pseudo_h_remap_byte(phys_combo: usize) -> u8 {
    WALL_PSEUDO_H_REMAP_BYTES[phys_combo]
}

#[inline]
pub fn wall_pseudo_v_remap_byte(phys_combo: usize) -> u8 {
    WALL_PSEUDO_V_REMAP_BYTES[phys_combo]
}

#[inline]
pub fn wall_topo_h_remap_byte(phys_combo: usize) -> u8 {
    WALL_TOPO_H_REMAP_BYTES[phys_combo]
}

#[inline]
pub fn wall_topo_v_remap_byte(phys_combo: usize) -> u8 {
    WALL_TOPO_V_REMAP_BYTES[phys_combo]
}
