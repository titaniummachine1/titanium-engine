export function renderLmrHint(container, state, controller) {
  const existing = container.querySelector('.lmr-hint');
  if (!state.showLmrHint) {
    existing?.remove();
    return;
  }
  if (existing) {
    return;
  }

  const shallow = state.settings.lmrVisionShallow;
  const hint = document.createElement('div');
  hint.className = 'lmr-hint';
  hint.innerHTML = `
    <div class="lmr-hint__card">
      <p class="lmr-hint__title">LMR vision</p>
      <div class="lmr-hint__bar" aria-hidden="true">
        <span class="lmr-hint__swatch lmr-hint__swatch--deep"></span>
        <span class="lmr-hint__swatch lmr-hint__swatch--mid"></span>
        <span class="lmr-hint__swatch lmr-hint__swatch--shallow"></span>
      </div>
      <p class="lmr-hint__labels"><span>full depth</span><span>reduced</span><span>deep cut</span></p>
      <p class="lmr-hint__text">
        ${shallow
    ? '<strong>Shallow</strong> — static LMR plan for this position before any search runs (pierce profile, move window, planned cuts). This is what speeds the tree up.'
    : '<strong>Search</strong> — actual root LMR after search (updates each depth, stays until position changes). <code>%</code> = node share per move.'}
        Numbers: <code>cm</code> = corridor heat, <code>−N</code> = ply reduction, <code>dN</code> = child depth, <code>%</code> = node share (search).
        Color = cut depth (green full → red deep cut). Dim = not searched yet. <code>↺</code> = re-search.
      </p>
      <button type="button" class="btn btn--primary btn--small" data-action="dismiss-lmr-hint">Got it</button>
    </div>
  `;
  hint.querySelector('[data-action="dismiss-lmr-hint"]')?.addEventListener('click', () => {
    controller.dismissLmrHint();
  });
  container.appendChild(hint);
}
