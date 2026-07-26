/* landing.js — loading sequence, scroll reveals, sakura fields */

import { SakuraField } from './petals.js';

const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

/* The intro plays from the hero, so a reload must not drop the visitor
 * halfway down the page with the loader covering it. */
if ('scrollRestoration' in history) history.scrollRestoration = 'manual';
scrollTo(0, 0);

/* ══════════════ loading screen ══════════════ */

const loader   = document.getElementById('loader');
const fill     = document.getElementById('loader-fill');
const pctEl    = document.getElementById('loader-pct');
const statusEl = document.getElementById('loader-status');
const ring     = document.querySelector('.enso__ring');

const STATUSES = [
  [0,  '建築を準備しています'],   // preparing
  [28, '間取り図を読み込み中'],   // reading the plan
  [52, '構成を検索しています'],   // retrieving compositions
  [76, '幾何形状を構築中'],       // building geometry
  [94, '完成'],                   // complete
];

const loaderField = new SakuraField(document.getElementById('loader-petals'), {
  mode: 'vortex',
  palette: 'glow',
  glow: true,
  density: 0.00022,
  minSize: 12,
  maxSize: 40,
  speed: 1.15,
});
loaderField.start();

/* Progress is real where it can be — fonts and the section images — and
 * eased elsewhere, so the bar never sits frozen at an arbitrary number.
 *
 * The <img> elements carry loading="lazy" and sit below the fold, so they
 * would never fire `load` while the loader covers the screen. Preloading
 * through `new Image()` bypasses the lazy attribute and gives real progress
 * without giving up lazy loading for the page itself. */
let progress = 0;
let assetsReady = 0;
const sources = [...document.querySelectorAll('.step__media img')]
  .map((img) => img.currentSrc || img.src);
const totalAssets = sources.length + 1;            // + fonts

const bumpAsset = () => { assetsReady++; };
sources.forEach((src) => {
  const pre = new Image();
  pre.onload = pre.onerror = bumpAsset;
  pre.src = src;
});
(document.fonts ? document.fonts.ready : Promise.resolve()).then(bumpAsset);

const MIN_MS = reduced ? 400 : 2600;               // let the animation breathe
const MAX_MS = reduced ? 800 : 7000;               // never hang on a slow asset
const started = performance.now();
let done = false;

function tick() {
  const elapsed = performance.now() - started;
  const timeShare = Math.min(elapsed / MIN_MS, 1);
  const assetShare = assetsReady / totalAssets;
  const timedOut = elapsed >= MAX_MS;
  // Never let the bar exceed what has actually loaded by much.
  const target = (timedOut ? 1 : Math.min(timeShare, 0.35 + assetShare * 0.65)) * 100;

  progress += (target - progress) * 0.08;
  const shown = Math.min(99, Math.floor(progress));
  pctEl.textContent = shown;
  fill.style.width = shown + '%';
  ring.style.strokeDashoffset = 490 - (490 * shown) / 100;

  for (const [at, text] of STATUSES) {
    if (shown >= at) statusEl.textContent = text;
  }

  if (!done && (timedOut || (timeShare >= 1 && assetShare >= 1 && shown >= 97))) {
    done = true;
    finish();
    return;
  }
  requestAnimationFrame(tick);
}

function finish() {
  pctEl.textContent = '100';
  fill.style.width = '100%';
  ring.style.strokeDashoffset = 0;

  if (reduced) { reveal(); return; }

  loaderField.explode();
  setTimeout(() => {
    loader.classList.add('is-washing');
    reveal();
  }, 420);
  setTimeout(() => {
    loader.classList.add('is-gone');
    loaderField.destroy();
  }, 1650);
}

function reveal() {
  document.body.classList.remove('is-loading');
  pageField.start();
  scan();
  updateBranch();
}

requestAnimationFrame(tick);

/* ══════════════ page petals ══════════════ */

const pageField = new SakuraField(document.getElementById('page-petals'), {
  mode: 'fall',
  palette: 'light',
  density: 0.00009,
  minSize: 8,
  maxSize: 24,
  speed: 0.55,
});

/* ══════════════ scroll reveals ══════════════ */

const revealables = [...document.querySelectorAll('.reveal')];
revealables.forEach((el) => {
  el.style.setProperty('--d', el.dataset.delay || 0);
});

const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    if (e.isIntersecting) {
      e.target.classList.add('is-in');
      io.unobserve(e.target);
    }
  }
}, { rootMargin: '0px 0px -12% 0px', threshold: 0.12 });

function scan() { revealables.forEach((el) => io.observe(el)); }

/* ══════════════ scroll-driven sakura + branch ══════════════ */

const branchPath = document.getElementById('branch-path');
const buds = [...document.querySelectorAll('.branch__bud')];

let lastY = scrollY;
let ticking = false;

function updateBranch() {
  const max = document.documentElement.scrollHeight - innerHeight;
  const p = max > 0 ? Math.min(scrollY / max, 1) : 0;

  branchPath.style.strokeDashoffset = 640 - 640 * p;
  buds.forEach((b) => {
    const t = parseFloat(b.style.getPropertyValue('--t'));
    b.classList.toggle('is-open', p >= t - 0.04);
  });
}

addEventListener('scroll', () => {
  const dy = scrollY - lastY;
  lastY = scrollY;
  pageField.setScroll(dy);
  if (!ticking) {
    ticking = true;
    requestAnimationFrame(() => { updateBranch(); ticking = false; });
  }
}, { passive: true });

/* Decay the scroll push so petals settle when scrolling stops. */
setInterval(() => pageField.setScroll(0), 260);

/* Smooth-scroll the in-page anchors even with scroll-behavior disabled. */
document.querySelectorAll('a[href^="#"]').forEach((a) => {
  a.addEventListener('click', (e) => {
    const t = document.querySelector(a.getAttribute('href'));
    if (!t) return;
    e.preventDefault();
    t.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
  });
});
