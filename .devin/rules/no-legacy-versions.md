# NO LEGACY VERSIONS IN THE LIVE ENGINE — EVER

## The rule (non-negotiable)

The engine repo (`C:/gitProjects/Quoridor best AI/engine`) is **always and only** the latest strongest build. No exceptions.

### Forbidden

- Adding `new_v16`, `new_v17`, `new_v18`, or any old version constructor to `wasm.rs` or any other file in the live engine
- Keeping old version code paths "for comparison" or "for the website"
- Building old engine versions inside the live engine repo
- Leaving old WASM binaries, old weights, or old build artifacts in the live tree
- Any `git checkout <old-commit>` in the live engine repo that leaves uncommitted state

### Allowed

- The live engine has exactly ONE version: the current strongest. Old constructors that already exist (`new_v17`, `new_v18`) are frozen reference points — do not delete them, but do NOT add more. When a new version ships, it gets ONE new constructor (`new_v19`, `new_v20`, etc.) and the previous one stays as-is.
- To build or test an older version: use a **git worktree** at the tagged commit, OR download the release artifact. Never touch the live engine.
- To reference old behavior: use `git show <tag>:path/to/file.rs`. Never checkout old code into the working tree.

## Release discipline — every version gets a release

When a new version is promoted to the website:

1. **Tag it**: `git tag v<NN> <commit-sha>` — the exact commit that ships
2. **Push the tag**: `git push origin v<NN>`
3. **Build native exe**: `RUSTFLAGS=-C target-cpu=native cargo build --release -p titanium` → copy to `artifacts/releases/v<NN>/titanium_v<NN>.exe`
4. **Build WASM**: `npm run build:wasm` from `site/web/` → output goes to `artifacts/releases/v<NN>/wasm/`
5. **Create GitHub release**: `gh release create v<NN> artifacts/releases/v<NN>/* --title "Titanium v<NN>" --notes "<changelog>"`

This means every past version is recoverable from its release artifact or git tag — no need to keep old code in the live engine.

## If you need to test old vs new

1. Create a worktree: `git worktree add ../v<NN>-build v<NN>`
2. Build in the worktree
3. Run the match
4. Remove the worktree: `git worktree remove ../v<NN>-build`

The live engine repo is never touched.

## Violations

If you find old version code polluting the live engine:
1. Do NOT delete it without checking — `new_v17`/`new_v18` are intentional frozen references
2. But if someone added `new_v16` back, or left uncommitted checkout state, that's a violation — report it and clean it up
