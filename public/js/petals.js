/* petals.js — canvas sakura field
 *
 * Petals are pre-rendered once per colour into offscreen canvases, then
 * blitted with a transform each frame. Drawing the bezier path per petal
 * per frame is what makes naive versions of this stutter at a few hundred
 * petals; blitting holds 60 fps at well over a thousand.
 *
 * Two modes:
 *   'fall'   gentle descent with sway — the page background
 *   'vortex' petals orbit and spiral inward — the loading screen
 */

const TAU = Math.PI * 2;

const PALETTES = {
  light: ['#F7C6D4', '#F2A8BC', '#E58AA5', '#FBDCE4', '#D98CA3'],
  glow:  ['#FFD7E4', '#FFB3CC', '#FF9BBD', '#FFC9DC', '#F58FB4'],
};

/** Draw one petal into an offscreen canvas. */
function renderPetal(color, size, glow) {
  const c = document.createElement('canvas');
  const pad = glow ? size * 0.5 : size * 0.15;
  c.width = c.height = Math.ceil(size + pad * 2);
  const x = c.getContext('2d');
  const w = size * 0.78, h = size;
  x.translate(c.width / 2, c.height / 2);

  if (glow) {
    x.shadowColor = color;
    x.shadowBlur = size * 0.55;
  }

  // Notched teardrop: narrow at the base, wide and cleft at the tip.
  x.beginPath();
  x.moveTo(0, h * 0.5);
  x.bezierCurveTo(w * 0.58, h * 0.16, w * 0.44, -h * 0.34, w * 0.14, -h * 0.42);
  x.quadraticCurveTo(0, -h * 0.24, -w * 0.14, -h * 0.42);
  x.bezierCurveTo(-w * 0.44, -h * 0.34, -w * 0.58, h * 0.16, 0, h * 0.5);
  x.closePath();

  const g = x.createLinearGradient(0, h * 0.5, 0, -h * 0.5);
  g.addColorStop(0, color);
  g.addColorStop(0.55, color);
  g.addColorStop(1, '#FFFFFF');
  x.fillStyle = g;
  x.globalAlpha = 0.95;
  x.fill();

  // Faint central vein — reads at large sizes, invisible at small ones.
  x.shadowBlur = 0;
  x.globalAlpha = 0.18;
  x.strokeStyle = '#B4536F';
  x.lineWidth = Math.max(0.6, size * 0.018);
  x.beginPath();
  x.moveTo(0, h * 0.42);
  x.lineTo(0, -h * 0.3);
  x.stroke();

  return c;
}

export class SakuraField {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: true });
    this.mode = opts.mode || 'fall';
    this.density = opts.density ?? 0.00016;   // petals per px² of viewport
    this.palette = PALETTES[opts.palette || 'light'];
    this.glow = !!opts.glow;
    this.minSize = opts.minSize ?? 10;
    this.maxSize = opts.maxSize ?? 30;
    this.speed = opts.speed ?? 1;
    this.reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;

    this.petals = [];
    this.wind = 0;
    this.windTarget = 0;
    this.scrollPush = 0;
    this.vortex = 0;          // 0..1 blend into the spiral
    this.burst = 0;           // >0 while petals fly outward
    this.time = 0;
    this.running = false;

    this.sprites = this.palette.map((c) =>
      renderPetal(c, 96, this.glow));

    this._resize = this._resize.bind(this);
    this._frame = this._frame.bind(this);
    this._resize();
    addEventListener('resize', this._resize, { passive: true });
  }

  _resize() {
    const dpr = Math.min(devicePixelRatio || 1, 2);
    this.w = this.canvas.clientWidth || innerWidth;
    this.h = this.canvas.clientHeight || innerHeight;
    this.canvas.width = Math.round(this.w * dpr);
    this.canvas.height = Math.round(this.h * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const target = Math.round(
      Math.min(this.w * this.h * this.density, this.reduced ? 40 : 420));
    while (this.petals.length < target) this.petals.push(this._spawn(true));
    if (this.petals.length > target) this.petals.length = target;
  }

  _spawn(anywhere = false) {
    const z = Math.random();                 // 0 far, 1 near
    return {
      x: Math.random() * this.w,
      y: anywhere ? Math.random() * this.h : -40 - Math.random() * this.h * 0.4,
      z,
      size: this.minSize + z * (this.maxSize - this.minSize),
      sprite: (Math.random() * this.sprites.length) | 0,
      rot: Math.random() * TAU,
      spin: (Math.random() - 0.5) * 0.02,
      flip: Math.random() * TAU,
      flipRate: 0.012 + Math.random() * 0.03,
      vy: (0.25 + z * 0.9) * this.speed,
      sway: 0.4 + Math.random() * 1.1,
      phase: Math.random() * TAU,
      // vortex state
      angle: Math.random() * TAU,
      radius: 0,
      vr: 0,
      vx: 0,
    };
  }

  /** Nudge the horizontal drift — call from a scroll handler. */
  setScroll(velocity) {
    this.scrollPush = Math.max(-6, Math.min(6, velocity * 0.05));
  }

  /** Blend toward the spiral (1) or free fall (0). */
  setVortex(v) { this.vortexTarget = v; }

  /** Fling every petal outward from the centre. */
  explode() {
    this.burst = 1;
    const cx = this.w / 2, cy = this.h / 2;
    for (const p of this.petals) {
      const a = Math.atan2(p.y - cy, p.x - cx) + (Math.random() - 0.5) * 0.8;
      const force = 9 + Math.random() * 22 + p.z * 10;
      p.vx = Math.cos(a) * force;
      p.vy = Math.sin(a) * force;
      p.spin = (Math.random() - 0.5) * 0.35;
    }
  }

  _step(dt) {
    this.time += dt;
    const cx = this.w / 2, cy = this.h / 2;

    // Wind wanders slowly, with occasional gusts.
    if (Math.random() < 0.004) this.windTarget = (Math.random() - 0.5) * 2.2;
    this.wind += (this.windTarget - this.wind) * 0.02;
    this.vortex += ((this.vortexTarget ?? (this.mode === 'vortex' ? 1 : 0))
                    - this.vortex) * 0.03;
    if (this.burst > 0) this.burst = Math.max(0, this.burst - dt * 0.35);

    for (const p of this.petals) {
      p.rot += p.spin * dt * 60;
      p.flip += p.flipRate * dt * 60;

      if (this.burst > 0) {
        p.x += p.vx * dt * 60;
        p.y += p.vy * dt * 60;
        p.vx *= 0.97;
        p.vy *= 0.97;
        continue;
      }

      // Free-fall component
      const swayX = Math.sin(this.time * p.sway + p.phase) * (0.7 + p.z);
      const fallX = p.x + (swayX + this.wind * (0.4 + p.z) + this.scrollPush * p.z) * dt * 60;
      const fallY = p.y + p.vy * dt * 60 * (1 + Math.abs(this.scrollPush) * 0.15);

      if (this.vortex > 0.001) {
        // Spiral component: orbit the centre while drifting inward.
        if (!p.radius) {
          p.radius = Math.hypot(p.x - cx, p.y - cy);
          p.angle = Math.atan2(p.y - cy, p.x - cx);
        }
        p.angle += (0.006 + p.z * 0.012) * dt * 60;
        p.radius += (Math.sin(this.time * 0.7 + p.phase) * 0.6 - 0.35) * dt * 60;
        const minR = Math.min(this.w, this.h) * 0.12;
        if (p.radius < minR) p.radius = Math.max(this.w, this.h) * 0.62;
        const vx = cx + Math.cos(p.angle) * p.radius;
        const vy = cy + Math.sin(p.angle) * p.radius;
        p.x = fallX + (vx - fallX) * this.vortex;
        p.y = fallY + (vy - fallY) * this.vortex;
      } else {
        p.x = fallX;
        p.y = fallY;
        p.radius = 0;
      }

      // Recycle off-screen petals (not while spiralling or bursting).
      if (this.vortex < 0.2) {
        if (p.y > this.h + 60) Object.assign(p, this._spawn(false));
        else if (p.x < -80) p.x = this.w + 60;
        else if (p.x > this.w + 80) p.x = -60;
      }
    }
  }

  _draw() {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.w, this.h);
    for (const p of this.petals) {
      const flip = Math.cos(p.flip);          // fake 3-D tumble
      const sx = Math.max(0.12, Math.abs(flip));
      ctx.save();
      ctx.translate(p.x, p.y);
      ctx.rotate(p.rot);
      ctx.scale(sx, 1);
      ctx.globalAlpha = (0.35 + p.z * 0.6) * (this.glow ? 0.95 : 0.85);
      const s = this.sprites[p.sprite];
      const d = p.size * 1.35;
      ctx.drawImage(s, -d / 2, -d / 2, d, d);
      ctx.restore();
    }
  }

  _frame(t) {
    if (!this.running) return;
    const dt = Math.min((t - (this.last || t)) / 1000, 0.05);
    this.last = t;
    this._step(dt);
    this._draw();
    requestAnimationFrame(this._frame);
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.last = 0;
    requestAnimationFrame(this._frame);
  }

  stop() { this.running = false; }

  destroy() {
    this.stop();
    removeEventListener('resize', this._resize);
  }
}
