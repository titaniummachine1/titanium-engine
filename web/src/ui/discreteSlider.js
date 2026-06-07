/**
 * Labeled stepped slider — scraped quoridor-ai AI Strength / AI Time pattern.
 */

export function renderDiscreteSlider({
  label,
  settingName,
  playerNum,
  value,
  presets,
  disabled = false,
}) {
  const min = presets[0].id;
  const max = presets[presets.length - 1].id;
  const ticks = presets
    .map((preset) => `<span class="discrete-slider__tick">${escapeHtml(preset.label)}</span>`)
    .join('');

  return `
    <div class="discrete-slider${disabled ? ' discrete-slider--disabled' : ''}">
      <label class="control-label control-label--sub">${escapeHtml(label)}</label>
      <input
        type="range"
        class="discrete-slider__input scraped-slider"
        data-setting="${settingName}-${playerNum}"
        min="${min}"
        max="${max}"
        step="1"
        value="${value}"
        ${disabled ? 'disabled' : ''}
      />
      <div class="discrete-slider__ticks" aria-hidden="true">${ticks}</div>
    </div>`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;');
}
