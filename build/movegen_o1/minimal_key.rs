//! Shrink wide local bitboards to minimal injective keys.

/// Greedy bit removal: drop any bit that does not change the outcome map.
pub fn find_minimal_runtime_mask(states: &[(u64, u16)]) -> u64 {
    if states.is_empty() {
        return 0;
    }
    let mut mask = states[0].0;
    for &(s, _) in &states[1..] {
        mask |= s;
    }
    if mask == 0 {
        return 0;
    }

    let mut bits: Vec<u32> = Vec::new();
    let mut m = mask;
    while m != 0 {
        let b = m.trailing_zeros();
        bits.push(b);
        m &= m - 1;
    }

    for &bit in &bits {
        let trial = mask & !(1u64 << bit);
        if trial == 0 {
            continue;
        }
        if is_injective(states, trial) {
            mask = trial;
        }
    }
    mask
}

fn is_injective(states: &[(u64, u16)], runtime_mask: u64) -> bool {
    let mut seen = std::collections::HashMap::new();
    for &(wide, moves) in states {
        let key = pext64(wide, runtime_mask);
        if let Some(&prev) = seen.get(&key) {
            if prev != moves {
                return false;
            }
        } else {
            seen.insert(key, moves);
        }
    }
    true
}

#[inline]
pub fn pext64(src: u64, mask: u64) -> u64 {
    let mut dst = 0u64;
    let mut k = 0u32;
    let mut m = mask;
    while m != 0 {
        let bit = m.trailing_zeros();
        if (src >> bit) & 1 != 0 {
            dst |= 1u64 << k;
        }
        k += 1;
        m &= m - 1;
    }
    dst
}

pub fn key_bits_needed(unique_outcomes: usize) -> u8 {
    if unique_outcomes <= 1 {
        0
    } else {
        (usize::BITS - (unique_outcomes - 1).leading_zeros()) as u8
    }
}

pub fn table_len_for_mask(runtime_mask: u64) -> usize {
    let n = runtime_mask.count_ones();
    if n == 0 {
        1
    } else if n >= 32 {
        usize::MAX // sentinel — should not happen
    } else {
        1usize << n
    }
}
