/** Rust Titanium player id — engine/ binary via titanium_ai.mjs only. */

export const RUST_TITANIUM_ID = 'rust-titanium';
export const GORISANSON_ID = 'gorisanson';
export const QUORIDOR_V3_ID = 'quoridor-v3';

export function assertRustTitaniumId(id) {
  if (id !== RUST_TITANIUM_ID) {
    throw new Error(`Expected "${RUST_TITANIUM_ID}" (Rust CLI). Got "${id}".`);
  }
}
