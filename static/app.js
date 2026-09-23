'use strict';

const $ = (id) => document.getElementById(id);
const els = {
  status: $('status'), results: $('results'), spinner: $('spinner'),
  message: $('message'), meta: $('meta'), query: $('query'), file: $('file'),
  preview: $('preview'), strategy: $('strategy'), topk: $('topk'),
  weight: $('weight'), weightValue: $('weight-value'), weightField: $('weight-field'),
};

let mode = 'text';
let busy = false;

/* ---------- state ---------- */

function setBusy(on) {
  busy = on;
  els.spinner.classList.toggle('hidden', !on);
  document.querySelectorAll('button.primary').forEach((b) => { b.disabled = on; });
  if (on) {
    els.message.classList.add('hidden');
    els.results.innerHTML = '';
    els.meta.classList.add('hidden');
  }
}

function showMessage(text, isError) {
  els.message.textContent = text;
  els.message.classList.remove('hidden');
  els.message.classList.toggle('error', Boolean(isError));
  els.results.innerHTML = '';
}

/* ---------- rendering ---------- */

function fmt(value) {
  return value === null || value === undefined ? '—' : Number(value).toFixed(4);
}

function card(item) {
  const el = document.createElement('article');
  el.className = 'card';

  const rank = document.createElement('span');
  rank.className = 'rank';
  rank.textContent = '#' + item.rank;

  const img = document.createElement('img');
  img.src = item.image_url;
  img.alt = item.name || ('Product ' + item.product_id);
  img.loading = 'lazy';

  const body = document.createElement('div');
  body.className = 'body';

  const name = document.createElement('div');
  name.className = 'name';
  name.textContent = item.name || '(no title)';

  const score = document.createElement('div');
  score.className = 'score';
  score.textContent = 'score ' + fmt(item.final_score);

  const tags = document.createElement('div');
  tags.className = 'tags';
  const md = item.metadata || {};
  tags.textContent = [md.articleType, md.baseColour, md.gender].filter(Boolean).join(' · ');

  const details = document.createElement('details');
  details.className = 'breakdown';
  const summary = document.createElement('summary');
  summary.textContent = 'breakdown';
  const dl = document.createElement('dl');
  const b = item.breakdown || {};
  [['image', fmt(b.image)], ['title', fmt(b.title)],
   ['strategy', b.strategy || ''], ['id', item.product_id]].forEach(([k, v]) => {
    const dt = document.createElement('dt'); dt.textContent = k;
    const dd = document.createElement('dd'); dd.textContent = v;
    dl.append(dt, dd);
  });
  details.append(summary, dl);

  body.append(name, score, tags, details);
  el.append(rank, img, body);
  return el;
}

function render(payload) {
  if (!payload.results.length) {
    showMessage('No products matched. Try a broader phrase or clear the filters.', false);
    return;
  }
  els.results.innerHTML = '';
  payload.results.forEach((item) => els.results.appendChild(card(item)));
  els.meta.textContent =
    `${payload.count} results · ${payload.strategy} · ` +
    `pool of ${payload.candidate_pool} candidates · ${payload.took_ms} ms`;
  els.meta.classList.remove('hidden');
}

/* ---------- requests ---------- */

function collectFilters() {
  const filters = {};
  ['articleType', 'baseColour', 'gender'].forEach((key) => {
    const value = $('f-' + key).value;
    if (value) filters[key] = value;
  });
  return Object.keys(filters).length ? filters : null;
}

async function readError(response) {
  try {
    const data = await response.json();
    if (typeof data.detail === 'string') return data.detail;
    if (Array.isArray(data.detail) && data.detail.length) return data.detail[0].msg;
  } catch (_) { /* fall through */ }
  return 'Request failed (' + response.status + ').';
}

async function searchText() {
  const query = els.query.value.trim();
  if (!query) { showMessage('Type something to search for.', true); return; }
  setBusy(true);
  try {
    const response = await fetch('/search/text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query,
        top_k: Number(els.topk.value) || 10,
        strategy: els.strategy.value,
        image_weight: Number(els.weight.value),
        filters: collectFilters(),
      }),
    });
    if (!response.ok) { showMessage(await readError(response), true); return; }
    render(await response.json());
  } catch (error) {
    showMessage('Could not reach the server. Is it running?', true);
  } finally {
    setBusy(false);
  }
}

async function searchImage() {
  const file = els.file.files[0];
  if (!file) { showMessage('Choose an image file first.', true); return; }
  const form = new FormData();
  form.append('file', file);
  const params = new URLSearchParams({
    top_k: String(Number(els.topk.value) || 10),
    strategy: els.strategy.value,
    image_weight: String(Number(els.weight.value)),
  });
  setBusy(true);
  try {
    const response = await fetch('/search/image?' + params, { method: 'POST', body: form });
    if (!response.ok) { showMessage(await readError(response), true); return; }
    render(await response.json());
  } catch (error) {
    showMessage('Could not reach the server. Is it running?', true);
  } finally {
    setBusy(false);
  }
}

function run() {
  if (busy) return;
  (mode === 'text' ? searchText : searchImage)();
}

/* ---------- wiring ---------- */

document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    mode = tab.dataset.mode;
    document.querySelectorAll('.tab').forEach((t) => {
      const active = t === tab;
      t.classList.toggle('active', active);
      t.setAttribute('aria-selected', String(active));
    });
    $('text-pane').classList.toggle('hidden', mode !== 'text');
    $('image-pane').classList.toggle('hidden', mode !== 'image');
  });
});

document.querySelectorAll('.chip').forEach((chip) => {
  chip.addEventListener('click', () => { els.query.value = chip.textContent; run(); });
});

$('search-text').addEventListener('click', run);
$('search-image').addEventListener('click', run);
els.query.addEventListener('keydown', (e) => { if (e.key === 'Enter') run(); });

els.file.addEventListener('change', () => {
  const file = els.file.files[0];
  if (!file) { els.preview.classList.add('hidden'); return; }
  els.preview.src = URL.createObjectURL(file);
  els.preview.classList.remove('hidden');
});

els.weight.addEventListener('input', () => {
  els.weightValue.textContent = Number(els.weight.value).toFixed(2);
});

els.strategy.addEventListener('change', () => {
  const usesWeights = ['weighted_sum', 'raw_weighted_sum'].includes(els.strategy.value);
  els.weightField.classList.toggle('hidden', !usesWeights);
});

/* ---------- boot ---------- */

(async function boot() {
  try {
    const response = await fetch('/health');
    const info = await response.json();
    if (!response.ok || info.ready === false) {
      els.status.textContent = 'Not ready: ' + (info.reason || 'artifacts unavailable');
      els.status.classList.add('error');
      return;
    }
    const counts = info.row_counts || {};
    els.status.textContent =
      `${info.products.toLocaleString()} products · ` +
      `${(counts.image || 0).toLocaleString()} image + ${(counts.title || 0).toLocaleString()} title ` +
      `vectors in one table · ${info.device}`;
  } catch (error) {
    els.status.textContent = 'Could not reach the server.';
    els.status.classList.add('error');
    return;
  }

  try {
    const options = await (await fetch('/filters')).json();
    ['articleType', 'baseColour', 'gender'].forEach((key) => {
      const select = $('f-' + key);
      (options[key] || []).forEach((value) => {
        const option = document.createElement('option');
        option.value = value;
        option.textContent = value;
        select.appendChild(option);
      });
    });
  } catch (error) { /* filters are optional */ }
})();
