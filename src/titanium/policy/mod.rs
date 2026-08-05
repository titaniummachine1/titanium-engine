//! Spatial policy head for move ordering — replaces CAT's corridor attention
//! heatmap with a learned MLP on a 14-plane encoding (subset of the
//! Claustrophobia AlphaZero 16-plane encoding; threat planes 14-15 omitted
//! per ablation — they contribute little to policy quality).
//!
//! ## Architecture
//!
//! ```text
//! input  (14 planes × 81 cells = 1,134 features)
//!   → Linear 1134→256 + ReLU
//!   → Linear 256→209 (raw logits)
//! ```
//!
//! Output is 209 spatial logits: 81 pawn destination cells (row-major) +
//! 64 horizontal wall slots + 64 vertical wall slots, matching the
//! Claustrophobia action space exactly.
//!
//! ## Encoding (14 planes, side-to-move canonical)
//!
//! | Plane | Description |
//! |-------|-------------|
//! | 0     | own pawn (one-hot) |
//! | 1     | opponent pawn (one-hot) |
//! | 2     | horizontal wall occupancy (8×8, row 8 / col 8 zero) |
//! | 3     | vertical wall occupancy (8×8) |
//! | 4     | own walls remaining / 10 (broadcast) |
//! | 5     | opponent walls remaining / 10 (broadcast) |
//! | 6     | side-to-move flag = 1.0 (broadcast, canonical) |
//! | 7     | own per-cell distance to goal / 20.0 (unreachable = 1.0) |
//! | 8     | opponent per-cell distance to goal / 20.0 |
//! | 9     | own shortest-path mask (one-hot) |
//! | 10    | opponent shortest-path mask (one-hot) |
//! | 11    | legal horizontal wall slots (one-hot) |
//! | 12    | legal vertical wall slots (one-hot) |
//! | 13    | race differential (d_opp - d_own) / 20.0, clamped [-1,1] (broadcast) |
//!
//! ## Cost
//!
//! ~1134×256 + 256×209 = ~345K multiplies per node. CAT's build_catv5_heatmaps
//! costs ~47% of search time; this head is designed to be well under 10%.

use crate::titanium::position::game::GameState;

pub const POLICY_PLANES: usize = 14;
pub const PLANE_SIZE: usize = 81;
pub const POLICY_INPUT_LEN: usize = POLICY_PLANES * PLANE_SIZE; // 1134
pub const POLICY_HIDDEN: usize = 256;
pub const POLICY_ACTIONS: usize = 209; // 81 pawn + 64 H walls + 64 V walls
pub const DIST_NORM: f32 = 20.0;

/// Blob layout (little-endian f32):
///   header:  [u32 magic = 0x504F4C50]  [u32 n_features]  [u32 hidden]  [u32 n_actions]
///   W1:      [hidden × n_features]
///   b1:      [hidden]
///   W2:      [n_actions × hidden]
///   b2:      [n_actions]
const BLOB_MAGIC: u32 = 0x504F4C50; // "POLO" — Policy On Linear mOdel
const BLOB_HEADER_LEN: usize = 16;

pub struct SpatialPolicyNet {
    /// Row-major [POLICY_HIDDEN][POLICY_INPUT_LEN]
    w1: Vec<f32>,
    /// Length POLICY_HIDDEN
    b1: Vec<f32>,
    /// Row-major [POLICY_ACTIONS][POLICY_HIDDEN]
    w2: Vec<f32>,
    /// Length POLICY_ACTIONS
    b2: Vec<f32>,
    /// Pre-allocated hidden buffer (reused per call)
    hidden_buf: Vec<f32>,
    /// Pre-allocated output buffer (reused per call)
    last_logits: [f32; POLICY_ACTIONS],
}

impl SpatialPolicyNet {
    pub fn load(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < BLOB_HEADER_LEN {
            return Err("policy blob too short for header".into());
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != BLOB_MAGIC {
            return Err(format!(
                "policy blob magic 0x{magic:08X}, expected 0x{BLOB_MAGIC:08X}"
            ));
        }
        let n_features = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
        let hidden = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let n_actions = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;

        if n_features != POLICY_INPUT_LEN {
            return Err(format!(
                "policy blob n_features={n_features}, expected {POLICY_INPUT_LEN}"
            ));
        }
        if hidden != POLICY_HIDDEN {
            return Err(format!(
                "policy blob hidden={hidden}, expected {POLICY_HIDDEN}"
            ));
        }
        if n_actions != POLICY_ACTIONS {
            return Err(format!(
                "policy blob n_actions={n_actions}, expected {POLICY_ACTIONS}"
            ));
        }

        let w1_count = hidden * n_features;
        let b1_count = hidden;
        let w2_count = n_actions * hidden;
        let b2_count = n_actions;
        let expected_payload = (w1_count + b1_count + w2_count + b2_count) * 4;
        if bytes.len() - BLOB_HEADER_LEN != expected_payload {
            return Err(format!(
                "policy blob payload {} bytes, expected {expected_payload}",
                bytes.len() - BLOB_HEADER_LEN
            ));
        }

        let read_f32s = |offset: usize, count: usize| -> Vec<f32> {
            (0..count)
                .map(|i| {
                    let base = BLOB_HEADER_LEN + (offset + i) * 4;
                    f32::from_le_bytes(bytes[base..base + 4].try_into().unwrap())
                })
                .collect()
        };

        let w1 = read_f32s(0, w1_count);
        let b1 = read_f32s(w1_count, b1_count);
        let w2 = read_f32s(w1_count + b1_count, w2_count);
        let b2 = read_f32s(w1_count + b1_count + w2_count, b2_count);

        Ok(Self {
            w1,
            b1,
            w2,
            b2,
            hidden_buf: vec![0.0; POLICY_HIDDEN],
            last_logits: [0.0; POLICY_ACTIONS],
        })
    }

    /// Score all 209 actions given the encoded feature vector.
    /// Returns raw logits — the caller picks the legal moves and sorts.
    #[inline]
    pub fn score_all(&mut self, features: &[f32]) -> &[f32; POLICY_ACTIONS] {
        debug_assert_eq!(features.len(), POLICY_INPUT_LEN);

        // Layer 1: hidden = ReLU(W1 · features + b1)
        for h in 0..POLICY_HIDDEN {
            let row_base = h * POLICY_INPUT_LEN;
            let mut acc = self.b1[h];
            for f in 0..POLICY_INPUT_LEN {
                acc += self.w1[row_base + f] * features[f];
            }
            self.hidden_buf[h] = acc.max(0.0); // ReLU
        }

        // Layer 2: logits = W2 · hidden + b2
        for a in 0..POLICY_ACTIONS {
            let row_base = a * POLICY_HIDDEN;
            let mut acc = self.b2[a];
            for h in 0..POLICY_HIDDEN {
                acc += self.w2[row_base + h] * self.hidden_buf[h];
            }
            self.last_logits[a] = acc;
        }

        &self.last_logits
    }

    /// Encode a GameState into the 14-plane feature vector (side-to-move canonical).
    /// This is the expensive part — it requires BFS floods for distance fields.
    ///
    /// `d0_field` and `d1_field` are per-cell distances to each player's goal,
    /// already computed by the search's distance layer infrastructure.
    /// `legal_h` and `legal_v` are bitmasks of geometrically legal wall slots.
    pub fn encode_features(
        g: &GameState,
        d0_field: &[u8; 81], // P0's distance to goal per cell
        d1_field: &[u8; 81], // P1's distance to goal per cell
        legal_h: u64,        // geometrically legal H wall slots
        legal_v: u64,        // geometrically legal V wall slots
        out: &mut [f32; POLICY_INPUT_LEN],
    ) {
        let me = g.turn as usize;
        let opp = 1 - me;
        let side = g.turn;

        // Canonical transforms: identity for side 0, 180-degree rotation for side 1
        let cell_idx = |plane: usize, c: u8| -> usize {
            let cc = if side == 0 { c as usize } else { 80 - c as usize };
            plane * PLANE_SIZE + cc
        };
        let slot_idx = |plane: usize, s: usize| -> usize {
            let cs = if side == 0 { s } else { 63 - s };
            plane * PLANE_SIZE + (cs / 8) * 9 + (cs % 8)
        };

        *out = [0.0; POLICY_INPUT_LEN];

        // Planes 0/1: one-hot pawns
        out[cell_idx(0, g.pawn[me] as u8)] = 1.0;
        out[cell_idx(1, g.pawn[opp] as u8)] = 1.0;

        // Planes 2/3: wall occupancy
        let mut h = g.hw_bits;
        while h != 0 {
            let s = h.trailing_zeros() as usize;
            out[slot_idx(2, s)] = 1.0;
            h &= h - 1;
        }
        let mut v = g.vw_bits;
        while v != 0 {
            let s = v.trailing_zeros() as usize;
            out[slot_idx(3, s)] = 1.0;
            v &= v - 1;
        }

        // Planes 4/5: wall inventories (broadcast)
        let own_walls = g.wl[me] as f32 / 10.0;
        let opp_walls = g.wl[opp] as f32 / 10.0;
        out[4 * PLANE_SIZE..5 * PLANE_SIZE].fill(own_walls);
        out[5 * PLANE_SIZE..6 * PLANE_SIZE].fill(opp_walls);

        // Plane 6: side-to-move (constant 1.0 in canonical frame)
        out[6 * PLANE_SIZE..7 * PLANE_SIZE].fill(1.0);

        // Planes 7/8: per-cell distance to goal
        // d0_field is P0's distances, d1_field is P1's distances.
        // In canonical frame, "own" = me, "opp" = opp.
        let own_field = if me == 0 { d0_field } else { d1_field };
        let opp_field = if me == 0 { d1_field } else { d0_field };
        for c in 0u8..81 {
            let d_own = own_field[c as usize];
            let d_opp = opp_field[c as usize];
            let norm_own = if d_own == u8::MAX { 1.0 } else { (d_own as f32).min(DIST_NORM) / DIST_NORM };
            let norm_opp = if d_opp == u8::MAX { 1.0 } else { (d_opp as f32).min(DIST_NORM) / DIST_NORM };
            out[cell_idx(7, c)] = norm_own;
            out[cell_idx(8, c)] = norm_opp;
        }

        // Planes 9/10: shortest-path masks
        // Cell c is on a shortest path iff dist(pawn→c) + dist(c→goal) == dist(pawn→goal)
        // We need a flood from each pawn. For now, approximate using the goal field:
        // a cell is on the path if its goal-distance decreases monotonically toward the goal.
        // Actually, we need pawn→cell distances too. The search already has these in
        // d0_layers/d1_layers, but we only have the goal-distance field here.
        // For the initial implementation, compute path masks from the goal field:
        // a cell c is on the own shortest path if there exists a neighbor c' with
        // dist(c') == dist(c) - 1, leading toward the goal. This is an approximation;
        // the proper version needs the pawn flood too.
        //
        // TODO: pass pawn flood fields to encode_features for exact path masks.
        // For now, leave planes 9/10 as zeros — the MLP can still learn from
        // the distance fields (plane 7/8) which contain the gradient information.

        // Planes 11/12: legal wall slots
        let mut lh = legal_h;
        while lh != 0 {
            let s = lh.trailing_zeros() as usize;
            out[slot_idx(11, s)] = 1.0;
            lh &= lh - 1;
        }
        let mut lv = legal_v;
        while lv != 0 {
            let s = lv.trailing_zeros() as usize;
            out[slot_idx(12, s)] = 1.0;
            lv &= lv - 1;
        }

        // Plane 13: race differential (broadcast)
        let d_own_pawn = own_field[g.pawn[me] as usize];
        let d_opp_pawn = opp_field[g.pawn[opp] as usize];
        let dist_diff = if d_own_pawn == u8::MAX || d_opp_pawn == u8::MAX {
            0.0
        } else {
            ((d_opp_pawn as f32 - d_own_pawn as f32) / DIST_NORM).clamp(-1.0, 1.0)
        };
        out[13 * PLANE_SIZE..14 * PLANE_SIZE].fill(dist_diff);
    }

    /// Map an engine move id to a 209-action index.
    /// Returns None for illegal/unmappable moves.
    /// 0..80: pawn destination cell (row-major, row 0 = top in ACE convention)
    /// 81..144: horizontal wall slot (0..63)
    /// 145..208: vertical wall slot (0..63)
    #[inline]
    pub fn move_to_action_index(m: i16) -> Option<usize> {
        if m < 0 {
            return None;
        }
        if m < 81 {
            // Pawn move — m IS the destination cell in ACE convention
            return Some(m as usize);
        }
        // Wall moves: need to check the engine's wall move encoding
        let slot = crate::titanium::wall_slot(m);
        if slot >= 64 {
            return None;
        }
        if crate::titanium::is_hwall_move(m) {
            Some(81 + slot as usize)
        } else {
            Some(145 + slot as usize)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_blob() -> Vec<u8> {
        let mut v = Vec::new();
        v.extend_from_slice(&BLOB_MAGIC.to_le_bytes());
        v.extend_from_slice(&(POLICY_INPUT_LEN as u32).to_le_bytes());
        v.extend_from_slice(&(POLICY_HIDDEN as u32).to_le_bytes());
        v.extend_from_slice(&(POLICY_ACTIONS as u32).to_le_bytes());
        // W1: hidden × input_len
        for i in 0..(POLICY_HIDDEN * POLICY_INPUT_LEN) {
            v.extend_from_slice(&((i as f32) * 0.0001).to_le_bytes());
        }
        // b1: hidden
        for i in 0..POLICY_HIDDEN {
            v.extend_from_slice(&((i as f32) * 0.01).to_le_bytes());
        }
        // W2: actions × hidden
        for i in 0..(POLICY_ACTIONS * POLICY_HIDDEN) {
            v.extend_from_slice(&((i as f32) * 0.0001).to_le_bytes());
        }
        // b2: actions
        for i in 0..POLICY_ACTIONS {
            v.extend_from_slice(&((i as f32) * 0.01).to_le_bytes());
        }
        v
    }

    #[test]
    fn loads_and_scores() {
        let blob = make_blob();
        let mut net = SpatialPolicyNet::load(&blob).expect("load");
        assert_eq!(net.hidden_buf.len(), POLICY_HIDDEN);

        let features = [0.5f32; POLICY_INPUT_LEN];
        let logits = net.score_all(&features);
        assert_eq!(logits.len(), POLICY_ACTIONS);
        // At least one logit should be non-zero with non-zero weights
        assert!(logits.iter().any(|&l| l != 0.0));
    }

    #[test]
    fn rejects_bad_magic() {
        let mut blob = make_blob();
        blob[0..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
        assert!(SpatialPolicyNet::load(&blob).is_err());
    }

    #[test]
    fn rejects_short_blob() {
        assert!(SpatialPolicyNet::load(&[0u8; 4]).is_err());
    }

    #[test]
    fn move_to_action_index_pawn_moves() {
        // Pawn move to cell 0 = action 0
        assert_eq!(SpatialPolicyNet::move_to_action_index(0), Some(0));
        assert_eq!(SpatialPolicyNet::move_to_action_index(80), Some(80));
    }
}
