//! Legal move generation (no pruning — see `cat::prune`).

pub mod legal;

pub use legal::{
    generate_legal_moves, generate_legal_moves_into, generate_legal_moves_slice, MAX_LEGAL_MOVES,
};
