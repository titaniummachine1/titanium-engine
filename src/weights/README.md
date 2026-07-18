# Weights

Titanium NNUE / HalfPW blobs. Always live here (not under `titanium/`).

| File | Role |
|------|------|
| net_weights.bin | live production (deploy overwrites) |
| net_weights_v17.bin | frozen v17 website snapshot |
| net_weights_frozen.bin | pinned v13 baseline |
| net_weights_medium.bin | medium tier |

Loaded via `include_bytes!` from `titanium/eval/nnue.rs`. Override at runtime with `TITANIUM_NET_WEIGHTS_PATH`.
