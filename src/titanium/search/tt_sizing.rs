//! Cache-tier TT index-bit sizing from CPUID L1/L2/L3 sizes.
//!
//! Shared by the Titanium play TT (and historically the perft TT) so overflow-driven
//! growth targets the same hardware cache tiers.

/// Fallback tier bits when CPUID cache detection is unavailable (non-x86 etc.).
const FALLBACK_START_BITS: usize = 9;
const FALLBACK_L2_BITS: usize = 11;
const FALLBACK_L3_BITS: usize = 16;

/// Detect (L1_data_per_core, L2_per_core, L3_total) in bytes via CPUID leaf 4.
/// Returns `None` on non-x86 or if the leaf reports no caches.
fn detect_cache_bytes() -> Option<(usize, usize, usize)> {
    #[cfg(target_arch = "x86_64")]
    {
        let mut l1d = 0usize;
        let mut l2 = 0usize;
        let mut l3 = 0usize;
        for sub in 0u32..64 {
            let r = std::arch::x86_64::__cpuid_count(4, sub);
            let cache_type = r.eax & 0x1f;
            if cache_type == 0 {
                break;
            } // no more caches
            let level = ((r.eax >> 5) & 0x7) as usize;
            let is_data = (cache_type & 1) != 0; // 1=data, 3=unified
            if !is_data {
                continue;
            }
            let line_size = ((r.ebx & 0xfff) + 1) as usize;
            let partitions = (((r.ebx >> 12) & 0x3ff) + 1) as usize;
            let ways = (((r.ebx >> 22) & 0x3ff) + 1) as usize;
            let sets = (r.ecx as usize) + 1;
            let size = line_size * partitions * ways * sets;
            match level {
                1 if l1d == 0 => l1d = size,
                2 if l2 == 0 => l2 = size,
                3 if l3 == 0 => l3 = size,
                _ => {}
            }
        }
        if l1d > 0 && l2 > 0 && l3 > 0 {
            return Some((l1d, l2, l3));
        }
    }
    None
}

/// Cache-tier index bits for ANY transposition table whose logical entry is
/// `entry_bytes` wide. Returns `(l1_start, l2, l3)` index bits — the largest
/// power-of-two entry count that fits each tier. Used by the Titanium search TT
/// (7 parallel arrays totaling ~25 B/entry) for overflow-driven cache-tier growth.
/// Falls back to 9/11/16 when CPUID cache detection is unavailable.
pub fn cache_tier_bits(entry_bytes: usize) -> (usize, usize, usize) {
    let to_bits = |cache: usize| -> usize {
        let n = cache / entry_bytes.max(1);
        if n < 2 {
            return 8;
        }
        ((usize::BITS - n.leading_zeros() - 1) as usize).clamp(8, 27)
    };
    if let Some((l1d, l2, l3)) = detect_cache_bytes() {
        let s = to_bits(l1d);
        let l2b = to_bits(l2).max(s + 1);
        let l3b = to_bits(l3).max(l2b + 1);
        (s, l2b, l3b)
    } else {
        (FALLBACK_START_BITS, FALLBACK_L2_BITS, FALLBACK_L3_BITS)
    }
}
