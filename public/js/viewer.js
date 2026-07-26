/* viewer.js — upload a plan, POST it to /api/generate, render the GLB */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';

const $ = (id) => document.getElementById(id);

const drop = $('drop'), fileIn = $('file'), preview = $('preview'),
      dropEmpty = $('drop-empty'), clearBtn = $('clear'),
      go = $('go'), styleSel = $('style'), seedIn = $('seed'),
      out = $('out'), alertEl = $('alert'), statsEl = $('stats'),
      recipeEl = $('recipe'), dlBtn = $('dl'), againBtn = $('again'),
      stageLoad = $('stage-load');

let imageDataURL = null;
let lastGLB = null;

/* ══════════════ file input ══════════════ */

function setImage(dataURL) {
  imageDataURL = dataURL;
  preview.src = dataURL;
  preview.hidden = false;
  dropEmpty.hidden = true;
  clearBtn.hidden = false;
  drop.classList.add('has-image');
  go.disabled = false;
  hideAlert();
}

function clearImage() {
  imageDataURL = null;
  preview.removeAttribute('src');
  preview.hidden = true;
  dropEmpty.hidden = false;
  clearBtn.hidden = true;
  drop.classList.remove('has-image');
  go.disabled = true;
  fileIn.value = '';
}

function readFile(file) {
  if (!file) return;
  if (!/^image\/(png|jpeg|webp)$/.test(file.type)) {
    return showAlert('That file is not a PNG, JPG or WebP image.');
  }
  if (file.size > 8 * 1024 * 1024) {
    return showAlert('That image is larger than 8 MB. Try a smaller export.');
  }
  const fr = new FileReader();
  fr.onload = () => setImage(fr.result);
  fr.onerror = () => showAlert('Could not read that file.');
  fr.readAsDataURL(file);
}

drop.addEventListener('click', () => { if (!imageDataURL) fileIn.click(); });
drop.addEventListener('keydown', (e) => {
  if ((e.key === 'Enter' || e.key === ' ') && !imageDataURL) {
    e.preventDefault();
    fileIn.click();
  }
});
fileIn.addEventListener('change', () => readFile(fileIn.files[0]));
clearBtn.addEventListener('click', (e) => { e.stopPropagation(); clearImage(); });

['dragenter', 'dragover'].forEach((t) =>
  drop.addEventListener(t, (e) => {
    e.preventDefault();
    drop.classList.add('is-over');
  }));
['dragleave', 'drop'].forEach((t) =>
  drop.addEventListener(t, (e) => {
    e.preventDefault();
    drop.classList.remove('is-over');
  }));
drop.addEventListener('drop', (e) => readFile(e.dataTransfer.files[0]));

$('use-sample').addEventListener('click', async () => {
  try {
    const r = await fetch('/img/sample-plan.png');
    if (!r.ok) throw new Error();
    const blob = await r.blob();
    const fr = new FileReader();
    fr.onload = () => setImage(fr.result);
    fr.readAsDataURL(blob);
  } catch {
    showAlert('Could not load the sample plan.');
  }
});

/* ══════════════ alerts ══════════════ */

function showAlert(msg) { alertEl.textContent = msg; alertEl.hidden = false; }
function hideAlert() { alertEl.hidden = true; }

/* ══════════════ generate ══════════════ */

async function generate(newSeed = false) {
  if (!imageDataURL) return;
  hideAlert();
  go.classList.add('is-busy');
  go.disabled = true;
  againBtn.disabled = true;
  if (!out.hidden) stageLoad.hidden = false;

  const seedVal = newSeed ? '' : seedIn.value.trim();

  try {
    const res = await fetch('/api/generate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: imageDataURL,
        style: styleSel.value,
        seed: seedVal === '' ? null : Number(seedVal),
      }),
    });

    let data;
    try { data = await res.json(); }
    catch { throw new Error(`Server returned ${res.status}.`); }

    if (!res.ok) throw new Error(data.error || `Server returned ${res.status}.`);

    lastGLB = base64ToBytes(data.glb);
    showStats(data.stats);
    out.hidden = false;
    await renderGLB(lastGLB);
    if (newSeed) seedIn.value = data.stats.seed;
    out.scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (err) {
    showAlert(err.message || 'Generation failed.');
  } finally {
    go.classList.remove('is-busy');
    go.disabled = !imageDataURL;
    againBtn.disabled = false;
    stageLoad.hidden = true;
  }
}

go.addEventListener('click', () => generate(false));
againBtn.addEventListener('click', () => generate(true));

function base64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

/* ══════════════ stats ══════════════ */

const FIELDS = [
  ['Footprint', (s) => `${s.width_m} × ${s.depth_m} m`],
  ['Area',      (s) => `${s.area_m2} m²`],
  ['Walls',     (s) => s.walls],
  ['Doors',     (s) => s.doors],
  ['Windows',   (s) => s.windows],
  ['Roof',      (s) => s.style],
  ['Pitch',     (s) => `${s.pitch}°`],
  ['Seed',      (s) => s.seed],
];

function showStats(s) {
  recipeEl.textContent = `${s.recipe} · ${s.material}`;
  statsEl.innerHTML = '';
  for (const [label, fn] of FIELDS) {
    const wrap = document.createElement('div');
    const dt = document.createElement('dt');
    const dd = document.createElement('dd');
    dt.textContent = label;
    dd.textContent = fn(s);
    wrap.append(dt, dd);
    statsEl.append(wrap);
  }
}

dlBtn.addEventListener('click', () => {
  if (!lastGLB) return;
  const url = URL.createObjectURL(new Blob([lastGLB], { type: 'model/gltf-binary' }));
  const a = document.createElement('a');
  a.href = url;
  a.download = 'house.glb';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});

/* ══════════════ three.js stage ══════════════ */

const canvas = $('stage');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.toneMapping = THREE.ACESFilmicToneMapping;
// The GLB ships flat PBR base colours with no textures, so there is nothing
// to hide blown highlights. Keep the total light budget low or the whole
// model washes out to white — the lawn and roof lose their colour first.
renderer.toneMappingExposure = 0.92;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(42, 16 / 10, 0.1, 2000);

const pmrem = new THREE.PMREMGenerator(renderer);
scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environmentIntensity = 0.35;

const key = new THREE.DirectionalLight(0xfff4ec, 1.45);
key.position.set(6, 12, 8);
key.castShadow = true;
key.shadow.mapSize.set(2048, 2048);
key.shadow.bias = -0.0006;
key.shadow.normalBias = 0.02;
scene.add(key);
scene.add(key.target);

const fill = new THREE.DirectionalLight(0xd8e4ff, 0.3);
fill.position.set(-8, 5, -6);
scene.add(fill);
scene.add(new THREE.HemisphereLight(0xffffff, 0xd8cfc7, 0.28));

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.maxPolarAngle = Math.PI * 0.495;   // never go under the ground
controls.minDistance = 3;
controls.maxDistance = 220;

let current = null;
const loader = new GLTFLoader();

function renderGLB(bytes) {
  return new Promise((resolve, reject) => {
    // copy into a fresh buffer: the GLTF parser takes ownership
    const buf = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    loader.parse(buf, '', (gltf) => {
      if (current) {
        scene.remove(current);
        current.traverse((o) => {
          if (o.isMesh) {
            o.geometry.dispose();
            (Array.isArray(o.material) ? o.material : [o.material])
              .forEach((m) => m.dispose());
          }
        });
      }
      current = gltf.scene;
      current.traverse((o) => {
        if (!o.isMesh) return;
        o.castShadow = true;
        o.receiveShadow = true;
      });
      scene.add(current);
      frameObject(current);
      resize();
      resolve();
    }, (err) => reject(new Error('Could not read the returned model.')));
  });
}

const GROUND_MAX_H = 0.2;      // metres; below this a part is a ground plane

/** Bounding box of the BUILDING, ignoring the site planes.
 *
 * builder.py lays a paved pad and a lawn that extend 6 m and 16 m past the
 * footprint. Framing on the full scene box therefore puts a 14 m house
 * inside a 30 m box and it renders postage-stamp sized. Ground parts are
 * thin slabs, so height alone separates them cleanly. */
function buildingBox(root) {
  const box = new THREE.Box3();
  const part = new THREE.Box3();
  let found = false;
  root.updateWorldMatrix(true, true);
  root.traverse((o) => {
    if (!o.isMesh) return;
    part.setFromObject(o);
    if (part.max.y - part.min.y > GROUND_MAX_H) {
      box.union(part);
      found = true;
    }
  });
  return found ? box : new THREE.Box3().setFromObject(root);
}

function frameObject(obj) {
  const full = new THREE.Box3().setFromObject(obj);
  const centre = full.getCenter(new THREE.Vector3());

  // drop the whole scene so the ground sits at y = 0 and it is centred in XZ
  obj.position.x -= centre.x;
  obj.position.z -= centre.z;
  obj.position.y -= full.min.y;

  const box = buildingBox(obj);
  const size = box.getSize(new THREE.Vector3());
  const mid = box.getCenter(new THREE.Vector3());

  const target = new THREE.Vector3(mid.x, size.y * 0.45, mid.z);
  const radius = size.length() * 0.5;                 // bounding-sphere radius
  // 0.92 leaves a little air around the building without letting the very
  // large lawn plane pull the camera back (see buildingBox above).
  const dist = (radius / Math.sin((camera.fov * Math.PI) / 360)) * 0.92;

  const dir = new THREE.Vector3(0.85, 0.5, 0.95).normalize();
  camera.position.copy(dir).multiplyScalar(dist).add(target);

  camera.near = Math.max(dist / 400, 0.05);
  camera.far = dist * 20;
  camera.updateProjectionMatrix();

  // Fit the shadow frustum to the building. An orthographic shadow camera
  // left at its defaults either misses the model or spreads 2048 px over a
  // huge area and the shadows turn to mush.
  const span = radius * 1.5;
  key.position.set(mid.x + span, span * 1.6, mid.z + span * 0.8);
  key.target.position.copy(target);
  key.target.updateMatrixWorld();
  const cam = key.shadow.camera;
  cam.left = -span; cam.right = span;
  cam.top = span; cam.bottom = -span;
  cam.near = 0.5; cam.far = span * 6;
  cam.updateProjectionMatrix();

  controls.target.copy(target);
  controls.minDistance = radius * 0.5;
  controls.maxDistance = dist * 5;
  controls.update();
}

function resize() {
  const w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  if (canvas.width !== w || canvas.height !== h) renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

new ResizeObserver(resize).observe(canvas);

(function loop() {
  requestAnimationFrame(loop);
  controls.update();
  renderer.render(scene, camera);
})();
