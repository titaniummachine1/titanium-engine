# Manual iteration (no auto-loop)

Each chunk (~10–15 min):
- 6 workers × 2 games = **12 probe games**
- Agent reads aggregate + reports, tweaks LMR/pierce, then next chunk

```bash
node benchmark/overnight_iterate.mjs --resume --steps 1 --workers 6 --probe-games 12 --no-confirm
```

Confirm only after manual review when probe is unclear.
