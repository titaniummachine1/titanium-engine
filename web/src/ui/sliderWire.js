/**
 * Wire range sliders without re-rendering the panel mid-drag.
 */

export function syncSliderFill(input) {
  const min = Number(input.min);
  const max = Number(input.max);
  const value = Number(input.value);
  const span = max - min;
  const pct = span === 0 ? 0 : ((value - min) / span) * 100;
  input.style.setProperty('--fill-pct', `${pct}%`);
}

/**
 * @param {HTMLElement} container
 * @param {string} selector
 * @param {(value: string, event: Event) => void} onValue
 * @param {() => void} [onCommit] — full UI refresh after drag ends
 */
export function wireRangeSlider(container, selector, onValue, onCommit) {
  const input = container.querySelector(selector);
  if (!input) {
    return;
  }

  const apply = (event) => {
    syncSliderFill(input);
    onValue(input.value, event);
  };

  syncSliderFill(input);
  input.addEventListener('input', apply);
  input.addEventListener('change', () => {
    apply({ target: input });
    onCommit?.();
  });
}
