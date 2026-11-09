/* ==========================================================================
   BBB-TransAI interface

   Motion is spring-driven and starts from whatever is on screen, so an
   animation that is interrupted picks up from its current value instead of
   jumping to the new one. No dependencies.
   ========================================================================== */

'use strict';

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)');
const COARSE = window.matchMedia('(hover: none)');

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/* --------------------------------------------------------------------------
   spring
   -------------------------------------------------------------------------- */

/**
 * Animate a scalar with a spring described the way Apple describes one:
 * `response` is how quickly it reaches the target in seconds, `damping` is
 * 1.0 for a critically damped settle and below 1.0 to overshoot.
 * Returns a handle exposing the live value so an interrupt can read it.
 */
function spring({ from, to, velocity = 0, response = 0.4, damping = 1, onUpdate, onDone }) {
  const omega = (2 * Math.PI) / response;
  const k = omega * omega;
  const c = 2 * damping * omega;

  let x = from;
  let v = velocity;
  let last = null;
  let started = null;
  let frame = 0;
  let stopped = false;

  // A spring approaches its target asymptotically, so a fixed epsilon either
  // never trips or trips immediately depending on the units. Scale it to the
  // distance being travelled, and cap the total run so nothing can spin on
  // forever holding a will-change layer open.
  const magnitude = Math.max(Math.abs(to - from), Math.abs(to), 1);
  const epsilonPosition = magnitude * 2e-3;
  const epsilonVelocity = magnitude * 2e-2;
  const deadline = 4000;

  if (REDUCED.matches) {
    onUpdate(to, 0);
    if (onDone) onDone();
    return { stop() {}, get value() { return to; }, get velocity() { return 0; } };
  }

  const step = (now) => {
    if (stopped) return;
    if (last === null) { last = now; started = now; }
    const dt = Math.min((now - last) / 1000, 1 / 30);
    last = now;

    const substeps = Math.max(1, Math.ceil(dt / (1 / 300)));
    const h = dt / substeps;
    for (let i = 0; i < substeps; i += 1) {
      v += (-k * (x - to) - c * v) * h;
      x += v * h;
    }

    onUpdate(x, v);

    const settled = Math.abs(to - x) < epsilonPosition && Math.abs(v) < epsilonVelocity;
    if (settled || now - started > deadline) {
      x = to;
      v = 0;
      onUpdate(to, 0);
      if (onDone) onDone();
      return;
    }
    frame = requestAnimationFrame(step);
  };

  frame = requestAnimationFrame(step);
  return {
    stop() { stopped = true; cancelAnimationFrame(frame); },
    get value() { return x; },
    get velocity() { return v; },
  };
}

/** A sliding selection pill. X and width get their own spring so they never desync. */
function slidingPill(container, pill) {
  let live = { x: 0, w: 0 };
  let sx = null;
  let sw = null;
  let settled = false;

  const paint = () => {
    pill.style.transform = `translate3d(${live.x}px,0,0)`;
    pill.style.width = `${live.w}px`;
  };

  function moveTo(target, animate = true) {
    if (!target) return;
    const base = container.getBoundingClientRect();
    const rect = target.getBoundingClientRect();
    const x = rect.left - base.left;
    const w = rect.width;
    if (w === 0) return;

    if (!animate || !settled) {
      if (sx) sx.stop();
      if (sw) sw.stop();
      live = { x, w };
      paint();
      settled = true;
      return;
    }

    const vx = sx ? sx.velocity : 0;
    const vw = sw ? sw.velocity : 0;
    if (sx) sx.stop();
    if (sw) sw.stop();

    // Both springs start from the live on-screen value and inherit velocity.
    sx = spring({
      from: live.x, to: x, velocity: vx, response: 0.34,
      onUpdate: (value) => { live.x = value; paint(); },
    });
    sw = spring({
      from: live.w, to: w, velocity: vw, response: 0.34,
      onUpdate: (value) => { live.w = value; paint(); },
    });
  }

  return { moveTo, reset: (el) => moveTo(el, false) };
}

/** Materialise a card: blur, scale and opacity arrive together. */
function materialise(el, delay = 0) {
  el.classList.add('enter');
  const start = () => {
    spring({
      from: 0, to: 1, response: 0.52,
      onUpdate: (p) => {
        const e = Math.max(0, Math.min(1, p));
        el.style.opacity = String(e);
        el.style.transform = `translate3d(0,${(1 - e) * 12}px,0) scale(${0.985 + 0.015 * e})`;
        el.style.filter = `blur(${(1 - e) * 10}px)`;
      },
      onDone: () => {
        el.classList.remove('enter');
        el.style.cssText = '';
      },
    });
  };
  if (delay > 0) setTimeout(start, delay); else start();
}

/* --------------------------------------------------------------------------
   formatting
   -------------------------------------------------------------------------- */

const pct = (x, digits = 1) => `${(x * 100).toFixed(digits)}`;
const signed = (x, digits = 3) => `${x >= 0 ? '+' : '−'}${Math.abs(x).toFixed(digits)}`;
const esc = (s) => String(s).replace(/[&<>"']/g, (ch) => (
  { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
));

/* --------------------------------------------------------------------------
   appearance
   -------------------------------------------------------------------------- */

const THEMES = ['auto', 'light', 'dark'];

function readTheme() {
  try {
    const stored = localStorage.getItem('bbb-theme');
    return THEMES.includes(stored) ? stored : 'auto';
  } catch (err) {
    return 'auto';
  }
}

function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  try { localStorage.setItem('bbb-theme', theme); } catch (err) { /* private mode */ }
}

function setupTheme() {
  applyTheme(readTheme());
  $('#theme-toggle').addEventListener('click', () => {
    const next = THEMES[(THEMES.indexOf(readTheme()) + 1) % THEMES.length];
    applyTheme(next);
  });
}

/* --------------------------------------------------------------------------
   gradients for the gauges
   -------------------------------------------------------------------------- */

/**
 * Probability bands. Each one owns a hue that the gauge, the verdict, the
 * badges and that peptide's row in a batch all share, so a result reads at a
 * glance and still carries its exact number.
 */
const BANDS = [
  { key: 'green',  min: 0.75, label: 'High',     tone: 'var(--c-green)',  from: 'var(--c-mint)',   to: 'var(--c-green)' },
  { key: 'blue',   min: 0.50, label: 'Moderate', tone: 'var(--c-blue)',   from: 'var(--c-cyan)',   to: 'var(--c-blue)' },
  { key: 'orange', min: 0.25, label: 'Low',      tone: 'var(--c-orange)', from: 'var(--c-yellow)', to: 'var(--c-orange)' },
  { key: 'pink',   min: 0.00, label: 'Very low', tone: 'var(--c-pink)',   from: 'var(--c-purple)', to: 'var(--c-pink)' },
];

const bandFor = (probability) => BANDS.find((b) => probability >= b.min) || BANDS[BANDS.length - 1];

/** Hues cycled across lists so neighbouring rows stay distinguishable. */
const WHEEL = [
  'var(--c-blue)', 'var(--c-indigo)', 'var(--c-purple)', 'var(--c-pink)',
  'var(--c-orange)', 'var(--c-yellow)', 'var(--c-green)', 'var(--c-mint)',
  'var(--c-teal)', 'var(--c-cyan)',
];

function installGradients() {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('width', '0');
  svg.setAttribute('height', '0');
  svg.setAttribute('aria-hidden', 'true');
  svg.style.position = 'absolute';
  // stop-color is written through `style`, not as a presentation attribute:
  // var() is only resolved in CSS declarations, and an attribute holding
  // var(...) is invalid and silently paints black.
  svg.innerHTML = `<defs>${BANDS.map((b) => `
      <linearGradient id="grad-${b.key}" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0%"   style="stop-color:${b.from}"/>
        <stop offset="100%" style="stop-color:${b.to}"/>
      </linearGradient>`).join('')}</defs>`;
  document.body.appendChild(svg);
}

/* --------------------------------------------------------------------------
   sequence maths, mirrored from bbb_core so the readout is instant
   -------------------------------------------------------------------------- */

const ALLOWED = new Set('ACDEFGHIKLMNPQRSTVWY');
const POSITIVE = new Set('KRH');
const NEGATIVE = new Set('DE');
const HYDROPHOBIC = new Set('AVILMFWYP');

function cleanSeq(text) {
  return String(text).toUpperCase().replace(/\s+/g, '')
    .split('').filter((a) => ALLOWED.has(a)).join('');
}

function localComposition(seq) {
  const n = seq.length;
  if (!n) return null;
  let pos = 0;
  let neg = 0;
  let hyd = 0;
  for (const a of seq) {
    if (POSITIVE.has(a)) pos += 1;
    if (NEGATIVE.has(a)) neg += 1;
    if (HYDROPHOBIC.has(a)) hyd += 1;
  }
  return { length: n, netCharge: pos - neg, fracPositive: pos / n, fracHydrophobic: hyd / n };
}

function countPeptides(text) {
  let count = 0;
  let inRecord = false;
  for (const raw of String(text).split('\n')) {
    const line = raw.trim();
    if (!line) continue;
    if (line.startsWith('>')) { count += 1; inRecord = true; continue; }
    if (inRecord && /^[ACDEFGHIKLMNPQRSTVWY\-*]+$/i.test(line)) continue;
    inRecord = false;
    count += 1;
  }
  return count;
}

/* --------------------------------------------------------------------------
   state
   -------------------------------------------------------------------------- */

const state = {
  mode: 'single',
  structure: null,      // { name, radius_gyration, residues }
  status: null,
  job: null,
  polling: null,
  lastResult: null,
};

/* --------------------------------------------------------------------------
   navigation
   -------------------------------------------------------------------------- */

function setupTabs() {
  const list = $('.tabs');
  const pill = slidingPill(list, $('.tab-pill'));
  const tabs = $$('.tab');

  const select = (tab, animate = true) => {
    tabs.forEach((t) => {
      const on = t === tab;
      t.classList.toggle('is-selected', on);
      t.setAttribute('aria-selected', String(on));
      $(`#${t.dataset.panel}`).classList.toggle('is-active', on);
    });
    pill.moveTo(tab, animate);
  };

  tabs.forEach((tab, index) => {
    tab.addEventListener('click', () => select(tab));
    tab.addEventListener('keydown', (event) => {
      const delta = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
      if (!delta) return;
      event.preventDefault();
      const next = tabs[(index + delta + tabs.length) % tabs.length];
      next.focus();
      select(next);
    });
  });

  requestAnimationFrame(() => select($('.tab.is-selected'), false));
  window.addEventListener('resize', () => pill.reset($('.tab.is-selected')));
}

function setupSegmented() {
  const group = $('.segmented');
  const pill = slidingPill(group, $('.segment-pill'));
  const segments = $$('.segment');

  const select = (segment, animate = true) => {
    state.mode = segment.dataset.mode;
    segments.forEach((s) => {
      const on = s === segment;
      s.classList.toggle('is-selected', on);
      s.setAttribute('aria-checked', String(on));
    });
    $$('.mode').forEach((m) => m.classList.toggle('is-active', m.dataset.mode === state.mode));
    pill.moveTo(segment, animate);
    updateStructureAvailability();
  };

  segments.forEach((segment, index) => {
    segment.addEventListener('click', () => select(segment));
    segment.addEventListener('keydown', (event) => {
      const delta = event.key === 'ArrowRight' ? 1 : event.key === 'ArrowLeft' ? -1 : 0;
      if (!delta) return;
      event.preventDefault();
      const next = segments[(index + delta + segments.length) % segments.length];
      next.focus();
      select(next);
    });
  });

  requestAnimationFrame(() => select($('.segment.is-selected'), false));
  window.addEventListener('resize', () => pill.reset($('.segment.is-selected')));
}

/* --------------------------------------------------------------------------
   composer
   -------------------------------------------------------------------------- */

function updateSingleReadout() {
  const raw = $('#single-seq').value;
  const seq = cleanSeq(raw);
  const counter = $('#single-counter');
  const readout = $('#single-readout');
  const comp = localComposition(seq);

  const dropped = raw.replace(/\s+/g, '').length - seq.length;
  counter.textContent = dropped > 0
    ? `${seq.length} residues · ${dropped} ignored`
    : `${seq.length} residues`;
  counter.classList.toggle('is-warn', seq.length > 200);

  if (!comp) { readout.hidden = true; return; }
  readout.hidden = false;
  $('#ro-length').textContent = comp.length;
  $('#ro-charge').textContent = comp.netCharge > 0 ? `+${comp.netCharge}` : comp.netCharge;
  $('#ro-positive').textContent = `${pct(comp.fracPositive, 0)}%`;
  $('#ro-hydro').textContent = `${pct(comp.fracHydrophobic, 0)}%`;
}

function updateBatchCounter() {
  const n = countPeptides($('#batch-seq').value);
  const counter = $('#batch-counter');
  counter.textContent = `${n} peptide${n === 1 ? '' : 's'}`;
  counter.classList.toggle('is-warn', n > (state.status?.max_peptides ?? 200));
}

function updateStructureAvailability() {
  const disclosure = $('#structure-disclosure');
  const note = $('#structure-summary');
  if (state.mode === 'batch') {
    disclosure.open = false;
    disclosure.setAttribute('hidden', '');
    note.textContent = '';
  } else {
    disclosure.removeAttribute('hidden');
    note.textContent = state.structure
      ? `measured · Rg ${state.structure.radius_gyration.toFixed(2)} Å`
      : 'optional · improves two descriptors';
  }
}

function setupComposer() {
  $('#single-seq').addEventListener('input', updateSingleReadout);
  $('#batch-seq').addEventListener('input', updateBatchCounter);

  $$('.mode-single .chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      $('#single-id').value = chip.dataset.id;
      $('#single-seq').value = chip.dataset.seq;
      clearStructure();
      updateSingleReadout();
      $('#single-seq').focus();
    });
  });

  $('#batch-example').addEventListener('click', () => {
    $('#batch-seq').value = [
      'TAT YGRKKRRQRRR',
      'Angiopep-2 TFFYGGSRGKRNNFKTEEY',
      'Penetratin RQIKIWFQNRRMKWKK',
      'MAP KLALKLALKALKAALKLA',
    ].join('\n');
    updateBatchCounter();
  });

  $('#clear').addEventListener('click', () => {
    $('#single-seq').value = '';
    $('#single-id').value = 'PEP_1';
    $('#batch-seq').value = '';
    clearStructure();
    updateSingleReadout();
    updateBatchCounter();
    $('#results').replaceChildren();
    hideError();
    state.lastResult = null;
  });

  // Enter in the single sequence box runs the prediction; Shift+Enter inserts a newline.
  $('#single-seq').addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  });

  $('#run').addEventListener('click', submit);
  setupDropzone();
  updateSingleReadout();
  updateBatchCounter();
}

/* --------------------------------------------------------------------------
   structure upload
   -------------------------------------------------------------------------- */

function clearStructure() {
  state.structure = null;
  const box = $('#structure-result');
  box.hidden = true;
  box.replaceChildren();
  $('#structure-file').value = '';
  updateStructureAvailability();
}

async function ingestStructure(file) {
  if (!file) return;
  if (file.size > 3 * 1024 * 1024) {
    showError('That structure file is larger than 3 MB.');
    return;
  }
  const text = await file.text();
  try {
    const response = await fetch('/api/structure', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ content: text, name: file.name }),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The structure could not be read.');

    state.structure = {
      name: file.name,
      radius_gyration: data.radius_gyration,
      residues: data.residues,
    };
    const box = $('#structure-result');
    box.hidden = false;
    box.innerHTML = `
      <span>Measured from <b>${esc(file.name)}</b></span>
      <span class="badge badge-accent">Rg ${data.radius_gyration.toFixed(2)} Å</span>
      <span class="badge">${data.residues} residues</span>
      <button class="button button-quiet" type="button" id="structure-remove">Remove</button>`;
    $('#structure-remove').addEventListener('click', clearStructure);
    updateStructureAvailability();
    hideError();
  } catch (err) {
    showError(err.message);
    clearStructure();
  }
}

function setupDropzone() {
  const zone = $('#dropzone');
  const input = $('#structure-file');

  zone.addEventListener('click', (event) => {
    if (event.target.tagName !== 'INPUT') input.click();
  });
  zone.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); input.click(); }
  });
  input.addEventListener('change', () => ingestStructure(input.files[0]));

  ['dragenter', 'dragover'].forEach((type) => zone.addEventListener(type, (event) => {
    event.preventDefault();
    zone.classList.add('is-over');
  }));
  ['dragleave', 'dragend'].forEach((type) => zone.addEventListener(type, () => {
    zone.classList.remove('is-over');
  }));
  zone.addEventListener('drop', (event) => {
    event.preventDefault();
    zone.classList.remove('is-over');
    ingestStructure(event.dataTransfer?.files?.[0]);
  });
}

/* --------------------------------------------------------------------------
   errors and progress
   -------------------------------------------------------------------------- */

function showError(message) {
  const box = $('#error');
  box.textContent = message;
  box.hidden = false;
}

function hideError() {
  const box = $('#error');
  box.hidden = true;
  box.textContent = '';
}

function showProgress(on) {
  const card = $('#progress');
  card.hidden = !on;
  if (on) {
    $('#progress-fill').style.width = '0%';
    $('#progress-stage').textContent = 'Preparing';
    $('#progress-count').textContent = '';
  }
}

function setProgress(snapshot) {
  $('#progress-stage').textContent = snapshot.stage || 'Working';
  if (snapshot.total) {
    $('#progress-count').textContent = `${snapshot.step} / ${snapshot.total}`;
    $('#progress-fill').style.width = `${Math.round((snapshot.step / snapshot.total) * 100)}%`;
  }
}

/* --------------------------------------------------------------------------
   submit and poll
   -------------------------------------------------------------------------- */

function collectInput() {
  if (state.mode === 'single') {
    const id = ($('#single-id').value || 'PEP_1').trim().split(/\s+/)[0] || 'PEP_1';
    const seq = cleanSeq($('#single-seq').value);
    if (!seq) throw new Error('Enter a peptide sequence.');
    const structures = {};
    if (state.structure) structures[id] = state.structure.radius_gyration;
    return { peptides: `${id} ${seq}`, structures };
  }
  const text = $('#batch-seq').value.trim();
  if (!text) throw new Error('Paste at least one peptide.');
  return { peptides: text, structures: {} };
}

async function submit() {
  hideError();
  const button = $('#run');
  if (button.disabled) return;

  let payload;
  try {
    payload = collectInput();
  } catch (err) {
    showError(err.message);
    return;
  }

  button.disabled = true;
  $('.button-label', button).textContent = 'Working…';
  $('#results').replaceChildren();
  showProgress(true);

  try {
    const response = await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'The server refused the request.');
    state.job = data.job;
    setProgress(data);
    poll();
  } catch (err) {
    finishRun();
    showError(err.message);
  }
}

function finishRun() {
  const button = $('#run');
  button.disabled = false;
  $('.button-label', button).textContent = 'Predict permeability';
  showProgress(false);
  if (state.polling) { clearTimeout(state.polling); state.polling = null; }
}

function poll() {
  state.polling = setTimeout(async () => {
    try {
      const response = await fetch(`/api/jobs/${state.job}`, { cache: 'no-store' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || 'Lost track of that prediction.');

      if (data.state === 'running') { setProgress(data); poll(); return; }
      if (data.state === 'error') { finishRun(); showError(data.error); return; }

      setProgress(data);
      finishRun();
      state.lastResult = data.result;
      render(data.result);
      if (COARSE.matches && navigator.vibrate) navigator.vibrate(8);
    } catch (err) {
      finishRun();
      showError(err.message);
    }
  }, 280);
}

/* --------------------------------------------------------------------------
   rendering
   -------------------------------------------------------------------------- */

function gaugeMarkup(probability) {
  const band = bandFor(probability);
  const radius = 54;
  const circumference = 2 * Math.PI * radius;
  return `
    <div class="gauge">
      <svg viewBox="0 0 128 128" role="img" aria-label="Permeability probability ${pct(probability)} percent">
        <circle class="gauge-track" cx="64" cy="64" r="${radius}"></circle>
        <circle class="gauge-value" cx="64" cy="64" r="${radius}"
                stroke="url(#grad-${band.key})"
                stroke-dasharray="${circumference.toFixed(2)}"
                stroke-dashoffset="${circumference.toFixed(2)}"></circle>
      </svg>
      <div class="gauge-label">
        <div class="gauge-number" data-target="${probability}">0.0<span>%</span></div>
        <div><span class="gauge-status">${band.label}</span></div>
      </div>
    </div>`;
}

function animateGauge(root, probability) {
  const circle = $('.gauge-value', root);
  const number = $('.gauge-number', root);
  if (!circle || !number) return;
  const radius = Number(circle.getAttribute('r'));
  const circumference = 2 * Math.PI * radius;

  spring({
    from: 0, to: probability, response: 0.85,
    onUpdate: (value) => {
      const clamped = Math.max(0, Math.min(1, value));
      circle.setAttribute('stroke-dashoffset', (circumference * (1 - clamped)).toFixed(2));
      number.innerHTML = `${pct(clamped)}<span>%</span>`;
    },
    onDone: () => { number.innerHTML = `${pct(probability)}<span>%</span>`; },
  });
}

function metricsMarkup(row) {
  const comp = row.composition;
  const p = row.percentiles || {};
  const items = [
    { name: 'Length', value: comp.length, unit: 'aa', pctKey: 'length', tone: 'var(--c-blue)' },
    {
      name: 'Net charge',
      value: comp.net_charge > 0 ? `+${comp.net_charge}` : comp.net_charge,
      pctKey: 'net_charge',
      tone: 'var(--c-purple)',
    },
    { name: 'K / R / H', value: pct(comp.frac_positive, 0), unit: '%', pctKey: 'frac_positive', tone: 'var(--c-pink)' },
    { name: 'Hydrophobic', value: pct(comp.frac_hydrophobic, 0), unit: '%', pctKey: 'frac_hydrophobic', tone: 'var(--c-orange)' },
    { name: 'Radius of gyration', value: comp.radius_gyration.toFixed(2), unit: 'Å', tone: 'var(--c-teal)' },
    { name: 'Compactness', value: comp.compactness.toFixed(2), tone: 'var(--c-indigo)' },
  ];

  return `<div class="metrics">${items.map((item) => {
    const percentile = item.pctKey ? p[item.pctKey] : null;
    return `<div class="metric" style="--tone:${item.tone}">
      <div class="metric-name">${esc(item.name)}</div>
      <div class="metric-value">${esc(item.value)}${item.unit ? `<small>${item.unit}</small>` : ''}</div>
      ${percentile === null || percentile === undefined ? '' : `
        <div class="metric-bar"><span data-width="${percentile}"></span></div>
        <div class="metric-pct">${percentile.toFixed(0)}th percentile of training set</div>`}
    </div>`;
  }).join('')}</div>`;
}

function driversMarkup(row) {
  const drivers = row.drivers || [];
  if (!drivers.length) return '';
  const peak = Math.max(...drivers.map((d) => Math.abs(d.contribution)), 1e-6);

  const rows = drivers.map((d) => {
    const up = d.contribution >= 0;
    const width = (Math.abs(d.contribution) / peak) * 50;
    return `<div class="driver">
      <div class="driver-name">
        <b>${esc(d.label)}</b>
        <code>${esc(d.feature)} = ${d.value}</code>
      </div>
      <div class="driver-bar">
        <span class="driver-fill ${up ? 'up' : 'down'}" data-width="${width.toFixed(2)}"></span>
      </div>
      <div class="driver-amount ${up ? 'up' : ''}">${signed(d.contribution)}</div>
    </div>`;
  }).join('');

  return `
    <div class="section-head">
      <h3>What drove this</h3>
      <span class="section-note">exact contributions, they sum to the probability</span>
    </div>
    <div class="drivers">${rows}</div>
    <p class="driver-foot">
      The forest starts every peptide at its base rate of
      <b>${pct(row.base_rate)}%</b>; the descriptors above move it from there to
      <b>${pct(row.probability)}%</b>. Each number is the mean change in predicted
      probability across all 500 trees that split on that descriptor.
    </p>`;
}

function structureBadge(row) {
  return row.structure_mode === 'measured'
    ? '<span class="badge badge-tone" style="--tone:var(--c-green)">Structure measured</span>'
    : '<span class="badge badge-tone" style="--tone:var(--c-teal)">Structure estimated from length</span>';
}

function singleCard(row, threshold) {
  const call = row.permeable ? 'Likely to cross' : 'Unlikely to cross';
  const band = bandFor(row.probability);
  return `
    <div class="card" data-result style="--tone:${band.tone}">
      <div class="verdict">
        ${gaugeMarkup(row.probability)}
        <div class="verdict-text">
          <div class="verdict-call">${call}</div>
          <div class="verdict-id">${esc(row.id)} · ${esc(row.sequence)}</div>
          <div class="verdict-meta">
            <span class="badge badge-tone" style="--tone:var(--c-indigo)">${row.composition.length} residues</span>
            ${structureBadge(row)}
            <span class="badge">cut-off ${threshold.toFixed(2)}</span>
          </div>
        </div>
      </div>

      <div class="section-head">
        <h3>Composition</h3>
        <span class="section-note">percentiles against the 538 training peptides</span>
      </div>
      ${metricsMarkup(row)}
      ${driversMarkup(row)}
    </div>`;
}

function batchCard(result) {
  const rows = [...result.results].sort((a, b) => b.probability - a.probability);
  const permeable = rows.filter((r) => r.permeable).length;

  const list = rows.map((row, index) => `
    <button class="ranked-row" type="button" data-index="${index}" aria-expanded="false"
            style="--tone:${bandFor(row.probability).tone}">
      <span class="ranked-rank">${index + 1}</span>
      <span class="ranked-id">${esc(row.id)}<span class="ranked-seq">${esc(row.sequence)}</span></span>
      <span class="ranked-bar"><span data-width="${(row.probability * 100).toFixed(1)}"></span></span>
      <span class="ranked-prob">${row.probability.toFixed(3)}</span>
    </button>
    <div class="ranked-detail" data-detail="${index}" hidden></div>`).join('');

  return `
    <div class="card" data-result>
      <div class="section-head" style="margin-top:0;padding-top:0;border:0">
        <h2>${rows.length} peptide${rows.length === 1 ? '' : 's'}</h2>
        <span class="section-note">ranked by probability · tap a row for its descriptors</span>
      </div>
      <div class="ranked">${list}</div>
      <div class="summary-row">
        <span class="badge badge-tone" style="--tone:var(--c-green)">${permeable} above the ${result.threshold.toFixed(2)} cut-off</span>
        <span class="badge badge-tone" style="--tone:var(--c-orange)">${rows.length - permeable} below</span>
        <a class="button button-quiet" id="export" href="/api/export/${state.job}.csv" download>Download CSV</a>
      </div>
    </div>`;
}

function render(result) {
  const host = $('#results');
  const rows = result.results;
  host.innerHTML = rows.length === 1 ? singleCard(rows[0], result.threshold) : batchCard(result);

  const card = $('[data-result]', host);
  materialise(card);

  if (rows.length === 1) {
    animateGauge(card, rows[0].probability);
    growBars(card);
  } else {
    growBars(card);
    wireBatchRows(card, result);
  }

  card.scrollIntoView({ behavior: REDUCED.matches ? 'auto' : 'smooth', block: 'nearest' });
}

/** Bars grow from zero with a spring, slightly staggered down the list. */
function growBars(root) {
  $$('[data-width]', root).forEach((bar, index) => {
    const target = Number(bar.dataset.width);
    const start = () => spring({
      from: 0, to: target, response: 0.55,
      onUpdate: (value) => { bar.style.width = `${Math.max(0, value)}%`; },
    });
    if (REDUCED.matches) { bar.style.width = `${target}%`; return; }
    setTimeout(start, Math.min(index * 28, 420));
  });
}

function wireBatchRows(card, result) {
  const sorted = [...result.results].sort((a, b) => b.probability - a.probability);
  $$('.ranked-row', card).forEach((button) => {
    button.addEventListener('click', () => {
      const index = Number(button.dataset.index);
      const detail = $(`[data-detail="${index}"]`, card);
      const open = !detail.hidden;

      $$('.ranked-detail', card).forEach((d) => { d.hidden = true; d.replaceChildren(); });
      $$('.ranked-row', card).forEach((r) => {
        r.classList.remove('is-open');
        r.setAttribute('aria-expanded', 'false');
      });

      if (open) return;

      const row = sorted[index];
      detail.style.setProperty('--tone', bandFor(row.probability).tone);
      detail.innerHTML = `
        <div class="verdict" style="margin-top:0.5rem">
          ${gaugeMarkup(row.probability)}
          <div class="verdict-text">
            <div class="verdict-call">${row.permeable ? 'Likely to cross' : 'Unlikely to cross'}</div>
            <div class="verdict-id">${esc(row.id)} · ${esc(row.sequence)}</div>
            <div class="verdict-meta">
              <span class="badge badge-tone" style="--tone:var(--c-indigo)">${row.composition.length} residues</span>
              ${structureBadge(row)}
            </div>
          </div>
        </div>
        ${metricsMarkup(row)}
        ${driversMarkup(row)}`;
      detail.hidden = false;
      button.classList.add('is-open');
      button.setAttribute('aria-expanded', 'true');
      animateGauge(detail, row.probability);
      growBars(detail);
      materialise(detail);
    });
  });
}

/* --------------------------------------------------------------------------
   model panel
   -------------------------------------------------------------------------- */

function renderModel(status) {
  const host = $('#model-card');
  const card = status.model;

  if (!card) {
    host.innerHTML = `<div class="card"><div class="error">${esc(status.error || 'The model could not be loaded.')}</div></div>`;
    return;
  }

  const rg = card.rg_estimator || {};
  const specs = [
    { name: 'Estimator', value: 'Random forest', tone: 'var(--c-blue)' },
    { name: 'Trees', value: card.n_trees, tone: 'var(--c-indigo)' },
    { name: 'Descriptors', value: card.n_features, tone: 'var(--c-purple)' },
    { name: 'Training peptides', value: card.training_peptides ?? '–', tone: 'var(--c-pink)' },
    { name: 'BBB positive', value: card.training_positive ?? '–', tone: 'var(--c-green)' },
    { name: 'Descriptor families', value: card.descriptor_families.length, tone: 'var(--c-teal)' },
  ];

  const importance = card.top_features;
  const peak = Math.max(...importance.map((f) => f.importance), 1e-6);

  host.innerHTML = `
    <div class="card">
      <h2>${esc(card.name)}</h2>
      <p class="prose">
        A ${esc(card.kind.replace('Classifier', ''))} classifier of <b>${card.n_trees}</b> trees over
        <b>${card.n_features}</b> descriptors: <b>${card.n_sequence_features}</b> computed from the
        sequence by iFeature across ${card.descriptor_families.length} descriptor families, and
        <b>${card.structural_features.length}</b> derived from the peptide's three-dimensional shape.
      </p>
      <div class="model-grid">
        ${specs.map((s) => `<div class="spec" style="--tone:${s.tone}">
          <div class="spec-name">${esc(s.name)}</div>
          <div class="spec-value">${esc(String(s.value))}</div>
        </div>`).join('')}
      </div>
      <div class="family-list">
        ${card.descriptor_families.map((f, i) => `<span class="badge badge-tone"
          style="--tone:${WHEEL[i % WHEEL.length]}">${esc(f)}</span>`).join('')}
      </div>
    </div>

    <div class="card">
      <h2>The structural descriptors</h2>
      <p class="prose">
        Three of the thirty inputs were computed from AlphaFold models when the model was trained:
        <b>radius of gyration</b>, <b>compactness</b> and the <b>fraction of K, R and H</b>.
        Together they carry <b>${pct(card.structural_importance)}%</b> of the forest's total
        importance, so every prediction has to supply them.
      </p>
      <p class="prose">
        The residue fraction is exact from the sequence. Radius of gyration needs a structure, so
        when none is attached it is estimated as
        <b>Rg = ${rg.scale} &middot; L<sup>${rg.exponent}</sup></b>, fitted on the
        <b>${rg.n}</b> reference structures that ship with this repository
        ${rg.mean_abs_error_angstrom ? `(mean absolute error <b>${rg.mean_abs_error_angstrom} Å</b>,
        about ${pct(rg.median_rel_error ?? 0, 0)}% of the typical value)` : ''};
        compactness follows as length divided by Rg.
      </p>
      <div class="callout">
        Measured against the ${rg.n} peptides that have both a sequence and a structure, estimating the
        two shape descriptors this way shifts the predicted probability by <b>0.018 on average</b> and
        never changes a call at the 0.5 cut-off. Attach a PDB or mmCIF model on the Predict tab to
        measure them instead.
      </div>
    </div>

    <div class="card">
      <h2>Most influential descriptors</h2>
      <p class="prose">Global importance over the whole forest. A single prediction's own
        contributions are shown with its result.</p>
      <div class="importance">
        ${importance.map((f, i) => `<div class="importance-row" style="--tone:${WHEEL[i % WHEEL.length]}">
          <div class="importance-name" title="${esc(f.feature)}">${esc(f.label)}</div>
          <div class="importance-bar"><span data-width="${((f.importance / peak) * 100).toFixed(1)}"></span></div>
          <div class="importance-value">${f.importance.toFixed(3)}</div>
        </div>`).join('')}
      </div>
    </div>

    <div class="card">
      <h2>Scope</h2>
      <p class="prose">
        Sequences use the twenty standard residues, from 2 to ${status.max_length} of them, up to
        ${status.max_peptides} peptides at a time. Non-standard residues, modifications,
        D&#8209;amino acids, cyclisation and stapling are outside what the training set covered, and
        a prediction for such a peptide reflects only its linear sequence.
      </p>
      <p class="prose">
        Research use only. The output is a hypothesis to test in an assay, not a measurement.
      </p>
    </div>`;

  growBars(host);
}

/* --------------------------------------------------------------------------
   boot
   -------------------------------------------------------------------------- */

async function boot() {
  installGradients();
  setupTheme();
  setupTabs();
  setupSegmented();
  setupComposer();

  // The composer is usable before the server answers, so it is shown first
  // rather than waiting on a round trip.
  $$('.hero, .composer').forEach((el, i) => materialise(el, i * 60));

  try {
    const response = await fetch('/api/status', { cache: 'no-store' });
    const status = await response.json();
    state.status = status;
    renderModel(status);

    if (!status.ifeature) {
      showError('iFeature is not installed on the server, so predictions cannot run. '
        + 'The Model tab still describes the loaded model.');
      $('#run').disabled = true;
    }
  } catch (err) {
    $('#model-card').innerHTML =
      '<div class="card"><div class="error">Could not reach the server for model details.</div></div>';
  }
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
