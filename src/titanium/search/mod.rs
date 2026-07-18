//! Search — Layer 3. The only play layer that makes search decisions.
//!
//! LMR helpers (`v16_lmr`, `cat_index_lmr`) and TT cache-tier sizing live here
//! next to the play engine. Historical αβ/CLI lives in `engine/legacy/search/`.

mod search_impl;
pub mod v16_lmr;
pub mod cat_index_lmr;
pub mod tt_sizing;

pub use search_impl::*;
