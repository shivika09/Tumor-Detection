// Port of the project's pipeline: grayscale -> autocontrast(cutoff=1) -> bicubic resize 64x64 -> /255 -> PCA -> RBF SVM
const IMG = 64;

function b64f32(s) {
  const bin = atob(s), n = bin.length, u = new Uint8Array(n);
  for (let i = 0; i < n; i++) u[i] = bin.charCodeAt(i);
  return new Float32Array(u.buffer);
}

function toGray(rgba, w, h) {
  const g = new Uint8Array(w * h);
  for (let i = 0, p = 0; i < g.length; i++, p += 4)
    g[i] = (rgba[p] * 19595 + rgba[p + 1] * 38470 + rgba[p + 2] * 7471 + 0x8000) >> 16;
  return g;
}

// Same rule as PIL.ImageOps.autocontrast(img, cutoff=1)
function autocontrast(g, cutoff) {
  const h = new Array(256).fill(0);
  for (let i = 0; i < g.length; i++) h[g[i]]++;
  let n = g.length, cut = Math.floor(n * cutoff / 100), lo, hi;
  for (lo = 0; lo < 256; lo++) {
    if (cut > h[lo]) { cut -= h[lo]; h[lo] = 0; } else { h[lo] -= cut; cut = 0; }
    if (cut <= 0) break;
  }
  cut = Math.floor(n * cutoff / 100);
  for (hi = 255; hi >= 0; hi--) {
    if (cut > h[hi]) { cut -= h[hi]; h[hi] = 0; } else { h[hi] -= cut; cut = 0; }
    if (cut <= 0) break;
  }
  for (lo = 0; lo < 256; lo++) if (h[lo]) break;
  for (hi = 255; hi >= 0; hi--) if (h[hi]) break;
  const lut = new Uint8Array(256);
  if (hi <= lo) { for (let i = 0; i < 256; i++) lut[i] = i; }
  else {
    const scale = 255 / (hi - lo), off = -lo * scale;
    for (let i = 0; i < 256; i++) lut[i] = Math.min(255, Math.max(0, Math.trunc(i * scale + off)));
  }
  const out = new Uint8Array(g.length);
  for (let i = 0; i < g.length; i++) out[i] = lut[g[i]];
  return out;
}

function bicubic(x) {
  const a = -0.5; x = Math.abs(x);
  if (x < 1) return ((a + 2) * x - (a + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * a;
  return 0;
}

function coeffs(inSize, outSize) {
  const scale = inSize / outSize, fs = Math.max(scale, 1), support = 2 * fs, ss = 1 / fs;
  const res = [];
  for (let xx = 0; xx < outSize; xx++) {
    const c = (xx + 0.5) * scale;
    let xmin = Math.max(Math.trunc(c - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(c + support + 0.5), inSize);
    const w = []; let sum = 0;
    for (let x = xmin; x < xmax; x++) { const v = bicubic((x - c + 0.5) * ss); w.push(v); sum += v; }
    for (let i = 0; i < w.length; i++) w[i] /= sum;
    res.push({ xmin, w });
  }
  return res;
}

const clip8 = v => v < 0 ? 0 : v > 255 ? 255 : Math.floor(v + 0.5);

// Pillow-style two-pass (horizontal, then vertical) bicubic resize with 8-bit intermediate
function resize(src, w, h, ow, oh) {
  const cx = coeffs(w, ow), cy = coeffs(h, oh);
  const tmp = new Uint8Array(ow * h);
  for (let y = 0; y < h; y++) for (let x = 0; x < ow; x++) {
    const { xmin, w: ws } = cx[x]; let s = 0;
    for (let i = 0; i < ws.length; i++) s += src[y * w + xmin + i] * ws[i];
    tmp[y * ow + x] = clip8(s);
  }
  const out = new Uint8Array(ow * oh);
  for (let y = 0; y < oh; y++) for (let x = 0; x < ow; x++) {
    const { xmin, w: ws } = cy[y]; let s = 0;
    for (let i = 0; i < ws.length; i++) s += tmp[(xmin + i) * ow + x] * ws[i];
    out[y * ow + x] = clip8(s);
  }
  return out;
}

function preprocess(rgba, w, h) {
  return resize(autocontrast(toGray(rgba, w, h), 1), w, h, IMG, IMG); // 64x64 uint8
}

function makeModel(m) {
  return { k: m.k, mean: b64f32(m.mean), comp: b64f32(m.components), sv: b64f32(m.sv),
           coef: b64f32(m.coef), nsv: m.nsv, b: m.b, gamma: m.gamma };
}

// returns SVM decision score: > 0 means tumor
function decide(M, pix) {
  const D = IMG * IMG, k = M.k, z = new Float64Array(k);
  for (let i = 0; i < D; i++) {
    const v = pix[i] / 255 - M.mean[i], row = i * k;
    for (let j = 0; j < k; j++) z[j] += v * M.comp[row + j];
  }
  let s = M.b;
  for (let n = 0; n < M.nsv; n++) {
    let d2 = 0; const o = n * k;
    for (let j = 0; j < k; j++) { const d = M.sv[o + j] - z[j]; d2 += d * d; }
    s += M.coef[n] * Math.exp(-M.gamma * d2);
  }
  return s;
}

