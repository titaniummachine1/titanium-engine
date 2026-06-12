include!("generated_tables_data.rs");

const PAWN_WALL_REMAP_BYTES: &[u8] = include_bytes!("generated_remap.bin");

#[inline]
pub fn wall_remap_byte(sq: u8, enemy_key: u8, phys_combo: usize) -> u8 {
    let idx = (sq as usize * 5 + enemy_key as usize) * PHYS_WALL_COMBOS + phys_combo;
    PAWN_WALL_REMAP_BYTES[idx]
}
