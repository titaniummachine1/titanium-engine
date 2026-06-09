export function renderCatHint(container, state, controller) {
  const existing = container.querySelector('.cat-hint');
  if (!state.showCatHint) {
    existing?.remove();
    return;
  }

  if (existing) {
    return;
  }

  const hint = document.createElement('div');
  hint.className = 'cat-hint';
  hint.innerHTML = `
    <div class="cat-hint__card">
      <p class="cat-hint__title">CAT vision</p>
      <div class="cat-hint__bar" aria-hidden="true"></div>
      <p class="cat-hint__labels"><span>cold</span><span>warm</span><span>hot</span></p>
      <p class="cat-hint__text">Square tint = attention heat. Dark squares are unreachable. Hover wall slots for searchable outlines.</p>
      <button type="button" class="btn btn--primary btn--small" data-action="dismiss-cat-hint">Got it</button>
    </div>
  `;
  hint.querySelector('[data-action="dismiss-cat-hint"]')?.addEventListener('click', () => {
    controller.dismissCatHint();
  });
  container.appendChild(hint);
}
