//! Fail-closed loader for the detached +1 LMR shadow sidecar.

use sha2::{Digest, Sha256};
use std::path::Path;

use super::net::live_weights_sha256;

pub const INPUTS: usize = 37;
const MAGIC_V1: &[u8; 8] = b"TISRDX1\0";
const MAGIC_V3: &[u8; 8] = b"TILMR3\0\0";
const BYTES_V1: usize = 8 + 12 + 32 + 8 + INPUTS * 8 + 4 * 8 + 32;

#[derive(Debug, Clone)]
enum ReductionArch {
    Linear5 {
        weights: [f64; 5],
        bias: f64,
    },
    Linear32 {
        weights: [f64; 32],
        bias: f64,
    },
    Linear37 {
        weights: [f64; 37],
        bias: f64,
    },
    Mlp37x8 {
        w1: [[f64; 37]; 8],
        b1: [f64; 8],
        w2: [f64; 8],
        b2: f64,
    },
}

#[derive(Debug, Clone)]
pub struct ReductionSidecar {
    arch: ReductionArch,
    calibration_scale: f64,
    calibration_shift: f64,
    threshold: f64,
}

impl ReductionSidecar {
    #[cfg(not(target_arch = "wasm32"))]
    pub fn load(path: &Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("sidecar read failed: {e}"))?;
        Self::from_bytes(&bytes)
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 8 {
            return Err("sidecar too short".into());
        }
        if &bytes[..8] == MAGIC_V1.as_slice() {
            Self::from_v1(bytes)
        } else if &bytes[..8] == MAGIC_V3.as_slice() {
            Self::from_v3(bytes)
        } else {
            Err("sidecar magic mismatch".into())
        }
    }

    fn from_v1(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() != BYTES_V1 {
            return Err(format!(
                "sidecar size mismatch: {} != {BYTES_V1}",
                bytes.len()
            ));
        }
        let read_u32 = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().unwrap());
        if read_u32(8) != 1 || read_u32(12) != 1 || read_u32(16) as usize != INPUTS {
            return Err("sidecar schema mismatch".into());
        }
        if bytes[20..52] != live_weights_sha256() {
            return Err("sidecar trunk hash mismatch".into());
        }
        if read_u32(52) != 1 || read_u32(56) != 1 {
            return Err("sidecar calibration/data version mismatch".into());
        }
        let payload_end = BYTES_V1 - 32;
        let digest: [u8; 32] = Sha256::digest(&bytes[..payload_end]).into();
        if bytes[payload_end..] != digest {
            return Err("sidecar payload hash mismatch".into());
        }
        let mut offset = 60;
        let mut weights = [0.0; 37];
        for weight in &mut weights {
            *weight = f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            offset += 8;
        }
        let mut next_f64 = || {
            let value = f64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            offset += 8;
            value
        };
        let bias = next_f64();
        let calibration_scale = next_f64();
        let calibration_shift = next_f64();
        let threshold = next_f64();
        if !weights.iter().all(|v| v.is_finite())
            || ![bias, calibration_scale, calibration_shift, threshold]
                .iter()
                .all(|v| v.is_finite())
            || !(0.0..=1.0).contains(&threshold)
        {
            return Err("sidecar contains invalid numeric values".into());
        }
        Ok(Self {
            arch: ReductionArch::Linear37 { weights, bias },
            calibration_scale,
            calibration_shift,
            threshold,
        })
    }

    fn from_v3(bytes: &[u8]) -> Result<Self, String> {
        let payload_end = bytes
            .len()
            .checked_sub(32)
            .ok_or_else(|| "sidecar too short".to_string())?;
        let digest: [u8; 32] = Sha256::digest(&bytes[..payload_end]).into();
        if bytes[payload_end..] != digest {
            return Err("sidecar payload hash mismatch".into());
        }
        let read_u32 = |at: usize| -> Result<u32, String> {
            let slice = bytes
                .get(at..at + 4)
                .ok_or_else(|| "sidecar truncated".to_string())?;
            Ok(u32::from_le_bytes(slice.try_into().unwrap()))
        };
        if read_u32(8)? != 1 {
            return Err("sidecar schema mismatch".into());
        }
        let layer_count = read_u32(12)?;
        let model_tag = read_u32(16)?;
        if bytes
            .get(20..52)
            .ok_or_else(|| "sidecar truncated".to_string())?
            != live_weights_sha256()
        {
            return Err("sidecar trunk hash mismatch".into());
        }
        let mut offset = 52usize;
        fn read_u32_at(bytes: &[u8], offset: &mut usize) -> Result<u32, String> {
            let slice = bytes
                .get(*offset..*offset + 4)
                .ok_or_else(|| "sidecar truncated".to_string())?;
            *offset += 4;
            Ok(u32::from_le_bytes(slice.try_into().unwrap()))
        }
        fn read_f64_at(bytes: &[u8], offset: &mut usize) -> Result<f64, String> {
            let slice = bytes
                .get(*offset..*offset + 8)
                .ok_or_else(|| "sidecar truncated".to_string())?;
            *offset += 8;
            Ok(f64::from_le_bytes(slice.try_into().unwrap()))
        }

        let arch = match (layer_count, model_tag) {
            (1, 3) => {
                if read_u32_at(bytes, &mut offset)? != 5 {
                    return Err("sidecar input count mismatch".into());
                }
                let mut weights = [0.0; 5];
                for weight in &mut weights {
                    *weight = read_f64_at(bytes, &mut offset)?;
                }
                let bias = read_f64_at(bytes, &mut offset)?;
                ReductionArch::Linear5 { weights, bias }
            }
            (1, 0) => {
                if read_u32_at(bytes, &mut offset)? != 32 {
                    return Err("sidecar input count mismatch".into());
                }
                let mut weights = [0.0; 32];
                for weight in &mut weights {
                    *weight = read_f64_at(bytes, &mut offset)?;
                }
                let bias = read_f64_at(bytes, &mut offset)?;
                ReductionArch::Linear32 { weights, bias }
            }
            (1, 1) => {
                if read_u32_at(bytes, &mut offset)? != 37 {
                    return Err("sidecar input count mismatch".into());
                }
                let mut weights = [0.0; 37];
                for weight in &mut weights {
                    *weight = read_f64_at(bytes, &mut offset)?;
                }
                let bias = read_f64_at(bytes, &mut offset)?;
                ReductionArch::Linear37 { weights, bias }
            }
            (2, 2) => {
                let in1 = read_u32_at(bytes, &mut offset)? as usize;
                let out1 = read_u32_at(bytes, &mut offset)? as usize;
                let act1 = read_u32_at(bytes, &mut offset)?;
                if in1 != 37 || out1 != 8 || act1 != 1 {
                    return Err("sidecar layer-1 mismatch".into());
                }
                let mut w1 = [[0.0; 37]; 8];
                for row in &mut w1 {
                    for weight in row {
                        *weight = read_f64_at(bytes, &mut offset)?;
                    }
                }
                let mut b1 = [0.0; 8];
                for bias in &mut b1 {
                    *bias = read_f64_at(bytes, &mut offset)?;
                }
                let in2 = read_u32_at(bytes, &mut offset)? as usize;
                let out2 = read_u32_at(bytes, &mut offset)? as usize;
                let act2 = read_u32_at(bytes, &mut offset)?;
                if in2 != 8 || out2 != 1 || act2 != 0 {
                    return Err("sidecar layer-2 mismatch".into());
                }
                let mut w2 = [0.0; 8];
                for weight in &mut w2 {
                    *weight = read_f64_at(bytes, &mut offset)?;
                }
                let b2 = read_f64_at(bytes, &mut offset)?;
                ReductionArch::Mlp37x8 { w1, b1, w2, b2 }
            }
            _ => return Err("unsupported sidecar architecture".into()),
        };
        let calibration_scale = read_f64_at(bytes, &mut offset)?;
        let calibration_shift = read_f64_at(bytes, &mut offset)?;
        let threshold = read_f64_at(bytes, &mut offset)?;
        if offset != payload_end {
            return Err("sidecar payload size mismatch".into());
        }
        let all_finite = match &arch {
            ReductionArch::Linear5 { weights, bias } => {
                weights.iter().all(|v| v.is_finite()) && bias.is_finite()
            }
            ReductionArch::Linear32 { weights, bias } => {
                weights.iter().all(|v| v.is_finite()) && bias.is_finite()
            }
            ReductionArch::Linear37 { weights, bias } => {
                weights.iter().all(|v| v.is_finite()) && bias.is_finite()
            }
            ReductionArch::Mlp37x8 { w1, b1, w2, b2 } => {
                w1.iter().flatten().all(|v| v.is_finite())
                    && b1.iter().all(|v| v.is_finite())
                    && w2.iter().all(|v| v.is_finite())
                    && b2.is_finite()
            }
        };
        if !all_finite
            || ![calibration_scale, calibration_shift, threshold]
                .iter()
                .all(|v| v.is_finite())
            || !(0.0..=1.0).contains(&threshold)
        {
            return Err("sidecar contains invalid numeric values".into());
        }
        Ok(Self {
            arch,
            calibration_scale,
            calibration_shift,
            threshold,
        })
    }

    pub fn predict(&self, hidden: &[f64; 32], context: &[f64; 5]) -> f64 {
        let logit = match &self.arch {
            ReductionArch::Linear5 { weights, bias } => {
                let mut value = *bias;
                for (weight, feature) in weights.iter().zip(context) {
                    value += weight * feature;
                }
                value
            }
            ReductionArch::Linear32 { weights, bias } => {
                let mut value = *bias;
                for (weight, feature) in weights.iter().zip(hidden) {
                    value += weight * feature;
                }
                value
            }
            ReductionArch::Linear37 { weights, bias } => {
                let mut value = *bias;
                for (weight, feature) in weights[..32].iter().zip(hidden) {
                    value += weight * feature;
                }
                for (weight, feature) in weights[32..].iter().zip(context) {
                    value += weight * feature;
                }
                value
            }
            ReductionArch::Mlp37x8 { w1, b1, w2, b2 } => {
                let mut hidden8 = [0.0; 8];
                for row in 0..8 {
                    let mut value = b1[row];
                    for (weight, feature) in w1[row][..32].iter().zip(hidden) {
                        value += weight * feature;
                    }
                    for (weight, feature) in w1[row][32..].iter().zip(context) {
                        value += weight * feature;
                    }
                    hidden8[row] = value.max(0.0);
                }
                let mut value = *b2;
                for (weight, feature) in w2.iter().zip(hidden8) {
                    value += weight * feature;
                }
                value
            }
        };
        let calibrated = self.calibration_scale * logit + self.calibration_shift;
        if !calibrated.is_finite() {
            return 0.0;
        }
        1.0 / (1.0 + (-calibrated.clamp(-60.0, 60.0)).exp())
    }

    pub fn would_activate(&self, probability: f64) -> bool {
        probability.is_finite() && probability >= self.threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn blob_v1() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC_V1);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(INPUTS as u32).to_le_bytes());
        bytes.extend_from_slice(&live_weights_sha256());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        for _ in 0..INPUTS {
            bytes.extend_from_slice(&0.0f64.to_le_bytes());
        }
        bytes.extend_from_slice(&(-6.0f64).to_le_bytes());
        bytes.extend_from_slice(&1.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.99f64.to_le_bytes());
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        bytes.extend_from_slice(&digest);
        bytes
    }

    #[test]
    fn neutral_sidecar_favors_no_activation() {
        let sidecar = ReductionSidecar::from_bytes(&blob_v1()).unwrap();
        let p = sidecar.predict(&[0.5; 32], &[0.5; 5]);
        assert!(p < 0.01);
        assert!(!sidecar.would_activate(p));
    }

    #[test]
    fn malformed_hash_and_trunk_fail_closed() {
        let mut bad_payload = blob_v1();
        bad_payload[100] ^= 1;
        assert!(ReductionSidecar::from_bytes(&bad_payload).is_err());
        let mut bad_trunk = blob_v1();
        bad_trunk[20] ^= 1;
        let payload_end = bad_trunk.len() - 32;
        let digest: [u8; 32] = Sha256::digest(&bad_trunk[..payload_end]).into();
        bad_trunk[payload_end..].copy_from_slice(&digest);
        assert!(ReductionSidecar::from_bytes(&bad_trunk).is_err());
    }

    #[test]
    fn nan_fails_closed() {
        let mut bytes = blob_v1();
        bytes[60..68].copy_from_slice(&f64::NAN.to_le_bytes());
        let payload_end = bytes.len() - 32;
        let digest: [u8; 32] = Sha256::digest(&bytes[..payload_end]).into();
        bytes[payload_end..].copy_from_slice(&digest);
        assert!(ReductionSidecar::from_bytes(&bytes).is_err());
    }

    fn blob_v3_linear32() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC_V3);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        bytes.extend_from_slice(&live_weights_sha256());
        bytes.extend_from_slice(&32u32.to_le_bytes());
        for _ in 0..32 {
            bytes.extend_from_slice(&0.0f64.to_le_bytes());
        }
        bytes.extend_from_slice(&(-6.0f64).to_le_bytes());
        bytes.extend_from_slice(&1.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.99f64.to_le_bytes());
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        bytes.extend_from_slice(&digest);
        bytes
    }

    fn blob_v3_linear5() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC_V3);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        bytes.extend_from_slice(&live_weights_sha256());
        bytes.extend_from_slice(&5u32.to_le_bytes());
        for _ in 0..5 {
            bytes.extend_from_slice(&0.0f64.to_le_bytes());
        }
        bytes.extend_from_slice(&(-6.0f64).to_le_bytes());
        bytes.extend_from_slice(&1.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.99f64.to_le_bytes());
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        bytes.extend_from_slice(&digest);
        bytes
    }

    fn blob_v3_linear37() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC_V3);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&live_weights_sha256());
        bytes.extend_from_slice(&37u32.to_le_bytes());
        for _ in 0..37 {
            bytes.extend_from_slice(&0.0f64.to_le_bytes());
        }
        bytes.extend_from_slice(&(-6.0f64).to_le_bytes());
        bytes.extend_from_slice(&1.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.99f64.to_le_bytes());
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        bytes.extend_from_slice(&digest);
        bytes
    }

    fn blob_v3_mlp() -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC_V3);
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&live_weights_sha256());
        bytes.extend_from_slice(&37u32.to_le_bytes());
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        for _ in 0..(37 * 8) {
            bytes.extend_from_slice(&0.0f64.to_le_bytes());
        }
        for _ in 0..8 {
            bytes.extend_from_slice(&0.0f64.to_le_bytes());
        }
        bytes.extend_from_slice(&8u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());
        for _ in 0..8 {
            bytes.extend_from_slice(&0.0f64.to_le_bytes());
        }
        bytes.extend_from_slice(&(-6.0f64).to_le_bytes());
        bytes.extend_from_slice(&1.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
        bytes.extend_from_slice(&0.99f64.to_le_bytes());
        let digest: [u8; 32] = Sha256::digest(&bytes).into();
        bytes.extend_from_slice(&digest);
        bytes
    }

    #[test]
    fn v3_linear32_is_supported() {
        let sidecar = ReductionSidecar::from_bytes(&blob_v3_linear32()).unwrap();
        let p = sidecar.predict(&[0.5; 32], &[0.5; 5]);
        assert!(p < 0.01);
    }

    #[test]
    fn v3_linear5_is_supported() {
        let sidecar = ReductionSidecar::from_bytes(&blob_v3_linear5()).unwrap();
        let p = sidecar.predict(&[0.5; 32], &[0.5; 5]);
        assert!(p < 0.01);
    }

    #[test]
    fn v3_linear37_is_supported() {
        let sidecar = ReductionSidecar::from_bytes(&blob_v3_linear37()).unwrap();
        let p = sidecar.predict(&[0.5; 32], &[0.5; 5]);
        assert!(p < 0.01);
    }

    #[test]
    fn v3_mlp_is_supported() {
        let sidecar = ReductionSidecar::from_bytes(&blob_v3_mlp()).unwrap();
        let p = sidecar.predict(&[0.5; 32], &[0.5; 5]);
        assert!(p < 0.01);
    }
}

#[cfg(test)]
pub(crate) fn neutral_test_blob() -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MAGIC_V3);
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&1u32.to_le_bytes());
    bytes.extend_from_slice(&live_weights_sha256());
    bytes.extend_from_slice(&37u32.to_le_bytes());
    for _ in 0..37 {
        bytes.extend_from_slice(&0.0f64.to_le_bytes());
    }
    bytes.extend_from_slice(&(-6.0f64).to_le_bytes());
    bytes.extend_from_slice(&1.0f64.to_le_bytes());
    bytes.extend_from_slice(&0.0f64.to_le_bytes());
    bytes.extend_from_slice(&0.99f64.to_le_bytes());
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    bytes.extend_from_slice(&digest);
    bytes
}
