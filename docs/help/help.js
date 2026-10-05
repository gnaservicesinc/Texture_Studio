'use strict';
// Local explanatory controls. No model code, network requests or stored settings.
const $ = id => document.getElementById(id);
const number = (id, allowZero = false) => {
  const control = $(id);
  const value = Number(control.value);
  if (!control.value.trim() || !Number.isSafeInteger(value) || value < (allowZero ? 0 : 1) || value > Number(control.max)) return null;
  return value;
};
const format = value => value.toLocaleString();
const goals = {
  extractor: 'Open Extractor and choose only the original products you need. You can do this without training.',
  photo: 'Open Photo Studio. Review its person layers, then export the cutout or chosen map.',
  datasets: 'Open Dataset Studio. Teachers use the full display photo. Review their full display maps, then train the experimental display student with consistent units. Stock RAFT is a separate left-grid experiment.',
  trainer: 'Open Trainer. Check the data and baseline, then start with a short run.',
  raw: 'Open Raw Studio. Inspect the stored samples and calibration before making derived depth.'
};
function updateGoal() { const goal = $('goal').value; $('goal-answer').textContent = goals[goal]; $('goal-link').href = '#' + goal; }
$('goal').addEventListener('change', updateGoal); updateGoal();
function updateMap() {
  const width = number('map-width'), height = number('map-height');
  $('map-size').textContent = width === null || height === null ? 'Enter a whole width and height within the shown limits.' : `${format(width * height)} samples · ${(width * height * 4 / 1048576).toFixed(1)} MiB before compression. Lossless ZIP size depends on the map.`;
}
['map-width', 'map-height'].forEach(id => $(id).addEventListener('input', updateMap)); updateMap();
const qualities = [
  ['Low', '256-pixel output tiles, 16-channel added features and 8 RAFT iterations. Every usable display target pixel is still trained.'],
  ['Medium', '512-pixel output tiles, 24-channel added features and 16 RAFT iterations. Native stereo inputs stay whole. More work and memory.'],
  ['High', '768-pixel output tiles, 32-channel added features and 24 RAFT iterations. Highest cost here, without guaranteed accuracy or memory fit. Stock mode uses these tile sizes as native crops.']
];
function updateQuality() { const quality = qualities[Number($('quality').value)]; $('quality-name').textContent = quality[0]; $('quality-answer').textContent = quality[1]; }
$('quality').addEventListener('input', updateQuality); updateQuality();
const modes = {
  distillation: 'The display student learns from original full-display teacher estimates. It can learn blur, wrong scale and mistakes too. Each run keeps one unit convention; no warp or stereo-teacher fallback is used.',
  supervised: 'The student learns from declared measured references on its own output grid. Display references belong on the full display grid; stock references belong on the native left grid. A teacher prediction is not a measurement.',
  mixed: 'The selected student uses eligible reference targets where present and compatible teacher targets for other entries. Check which kind each sample uses.'
};
function updateMode() { $('mode-answer').textContent = modes[$('training-mode').value]; }
$('training-mode').addEventListener('change', updateMode); updateMode();
function updateTraining() {
  const images = number('train-images'), accumulation = number('accumulation'), limit = number('train-limit');
  const epochMode = $('limit-mode').value === 'epochs';
  $('limit-label').textContent = epochMode ? 'Training epochs' : 'Total Steps';
  if (images === null || accumulation === null || limit === null) { $('training-plan').textContent = 'Enter positive whole numbers within the shown limits.'; return; }
  const perEpoch = Math.ceil(images / accumulation);
  const total = epochMode ? perEpoch * limit : limit;
  if (!Number.isSafeInteger(total)) { $('training-plan').textContent = 'This plan is too large to show exactly.'; return; }
  const complete = Math.floor(total / perEpoch), remainder = total % perEpoch;
  const visits = epochMode ? images * limit : complete * images + Math.min(remainder * accumulation, images);
  $('training-plan').textContent = `${format(perEpoch)} updates per epoch · ${format(total)} Total Steps · ${format(visits)} image visits. ${epochMode ? 'Ends after ' + format(limit) + ' complete epochs.' : 'Spans ' + format(Math.ceil(total / perEpoch)) + ' epochs and stops at the requested update.'}`;
}
['train-images', 'accumulation', 'train-limit'].forEach(id => $(id).addEventListener('input', updateTraining));
$('limit-mode').addEventListener('change', updateTraining); updateTraining();
function updateValidation() {
  const count = number('validation-count'), sample = number('validation-sample', true);
  if (count === null || sample === null) { $('validation-plan').textContent = 'Enter a positive set size and a sample size of zero or more.'; return; }
  const used = sample === 0 ? count : Math.min(count, sample);
  $('validation-plan').textContent = used === count ? `All ${format(count)} entries are checked each time.` : `${format(used)} of ${format(count)} entries (${(used / count * 100).toFixed(1)}%) are chosen at random each time. Scores can vary. An early-end goal is confirmed with the whole set before stopping.`;
}
['validation-count', 'validation-sample'].forEach(id => $(id).addEventListener('input', updateValidation)); updateValidation();
const topics = [...document.querySelectorAll('.topic')];
const links = [...document.querySelectorAll('aside nav a')];
$('search').addEventListener('input', () => {
  const query = $('search').value.toLocaleLowerCase().trim();
  let shown = 0;
  topics.forEach(topic => { const match = !query || topic.textContent.toLocaleLowerCase().includes(query); topic.hidden = !match; if (match) shown++; });
  links.forEach(link => { const topic = document.querySelector(link.getAttribute('href')); link.hidden = topic.hidden; });
  $('no-results').hidden = shown > 0;
});
function markCurrent() {
  const hash = location.hash || '#start';
  const topic = document.getElementById(hash.slice(1));
  if (topic && topic.hidden) { $('search').value = ''; $('search').dispatchEvent(new Event('input')); }
  links.forEach(link => { if (link.getAttribute('href') === hash) link.setAttribute('aria-current', 'location'); else link.removeAttribute('aria-current'); });
}
window.addEventListener('hashchange', markCurrent); markCurrent();
