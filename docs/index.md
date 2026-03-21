---
title: Shapix
description: Elegant runtime shape and dtype checking for NumPy, JAX, and PyTorch arrays — powered by beartype.
hide:
  - navigation
  - toc
---

<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@700;800;900&display=swap" rel="stylesheet">

<style>
.md-typeset h1 { display: none; }
.md-content__button { display: none !important; }
.md-content__inner { padding: 0; margin: 0; max-width: none; }
.md-main__inner { margin-top: 0; }
.md-content { padding-top: 0; }
.md-footer { display: none !important; }
html { overflow-x: hidden; }

/* Make page transparent so fixed visual shows through */
body { background: transparent !important; }

/* ── Full-page visual canvas ── */
#shapix-visual {
  position: fixed;
  inset: 0;
  z-index: 0;
  pointer-events: none;
}

/* ── Hero ── */
.hero {
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 0 1rem 0;
}

.hero__content {
  text-align: center;
  perspective: 800px;
}

/* ── Logo + Title (stacked, 3D tilt via JS) ── */
.hero__logo {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0;
  margin-bottom: 0;
  transform-style: preserve-3d;
  transition: transform 0.08s ease-out;
}

.hero__logo canvas {
  filter: drop-shadow(0 0 22px rgba(140, 90, 255, 0.3));
}

/* ── 3D extruded title ── */
.hero__title {
  display: flex;
  gap: 0;
  transform: rotateX(8deg);
  transform-origin: center bottom;
}

.hero__letter {
  display: inline-block;
  font-family: 'Outfit', sans-serif;
  font-size: 3.2rem;
  font-weight: 900;
  letter-spacing: -0.04em;
  color: #c9b3ff;
  text-shadow:
    0 1px 0 #6b4bbd,
    0 2px 0 #6344b5,
    0 3px 0 #5b3dad,
    0 4px 0 #5336a5,
    0 5px 0 #4b2f9d,
    0 6px 0 #432895,
    0 7px 3px rgba(40,15,100,0.5),
    0 0 16px rgba(124,77,255,0.15);
  animation: letterFloat 3.5s ease-in-out infinite;
}

@keyframes letterFloat {
  0%, 100% { transform: translateY(0); }
  50% { transform: translateY(-6px); }
}

/* ── Tagline & Subtitle ── */
.hero__tagline {
  font-family: 'Outfit', sans-serif;
  font-size: 0.85rem;
  font-weight: 700;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--md-primary-fg-color);
  margin-top: 0;
  margin-bottom: 0.3rem;
  opacity: 0.7;
}

.hero__subtitle {
  font-family: 'Outfit', sans-serif;
  font-size: 1.15rem;
  font-weight: 500;
  color: var(--md-default-fg-color--light);
  max-width: 580px;
  margin: 0 auto 0.75rem;
  line-height: 1.5;
}

.beartype-link {
  white-space: nowrap;
  color: inherit !important;
  transition: color 0.3s ease;
}

.beartype-logo-inline {
  height: 0.06em;
  width: auto;
  vertical-align: 0;
  margin-right: 0.04em;
  opacity: 0.4;
  filter: grayscale(100%) brightness(0.5);
  transition: all 0.3s ease;
}

[data-md-color-scheme="slate"] .beartype-logo-inline {
  filter: grayscale(100%) brightness(0.5) invert(1);
}

.beartype-link:hover .beartype-logo-inline {
  opacity: 1;
  filter: grayscale(0%);
}

.beartype-link:hover {
  color: #8B5E3C !important;
}

/* ── Buttons ── */
.hero__actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 0.5rem;
}

.hero__actions .md-button {
  font-family: 'Outfit', sans-serif;
  font-size: 0.9rem;
  font-weight: 700;
  padding: 0.5rem 1.6rem;
  border-radius: 2rem;
  border: 2px solid var(--md-primary-fg-color);
  color: var(--md-primary-fg-color);
  background: transparent;
  transition: all 0.2s ease;
}

.hero__actions .md-button--primary {
  background: var(--md-primary-fg-color);
  color: white;
  border-color: var(--md-primary-fg-color);
}

.hero__actions .md-button:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 16px rgba(124, 77, 255, 0.3);
  background: var(--md-primary-fg-color--dark);
  color: white;
  border-color: var(--md-primary-fg-color--dark);
}

.hero__actions .md-button--primary:hover {
  background: var(--md-primary-fg-color--dark);
  border-color: var(--md-primary-fg-color--dark);
}

/* ── Features grid with glassmorphism ── */
.features-section {
  position: relative;
  z-index: 1;
  max-width: 61rem;
  margin: 0 auto;
  padding: 0.5rem 1.5rem;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 0.75rem;
  padding: 0.5rem 0;
}

.feature-card {
  padding: 0.8rem 1rem;
  border-radius: 14px;
  background: rgba(255,255,255,0.55);
  backdrop-filter: blur(16px);
  -webkit-backdrop-filter: blur(16px);
  border: 1px solid rgba(124,77,255,0.08);
  transition: border-color 0.2s, box-shadow 0.2s, transform 0.2s;
}

.feature-card:hover {
  border-color: var(--md-accent-fg-color);
  box-shadow: 0 4px 24px rgba(124, 77, 255, 0.1);
  transform: translateY(-2px);
}

[data-md-color-scheme="slate"] .feature-card {
  background: rgba(13,17,23,0.5);
  border-color: rgba(179,136,255,0.1);
}

[data-md-color-scheme="slate"] .feature-card:hover {
  border-color: rgba(179,136,255,0.3);
  box-shadow: 0 4px 24px rgba(179, 136, 255, 0.08);
}

.feature-card h3 {
  margin-top: 0;
  margin-bottom: 0.3rem;
  font-size: 0.95rem;
}

.feature-card h3 .headerlink { display: none; }

.feature-card p {
  color: var(--md-default-fg-color--light);
  font-size: 0.82rem;
  margin-bottom: 0;
}

/* ── Backends with glassmorphism ── */
.backends {
  position: relative;
  z-index: 1;
  text-align: center;
  max-width: 61rem;
  margin: 0 auto 0.5rem;
  padding: 0.75rem 1.5rem 1rem;
  border-radius: 16px;
  background: rgba(255,255,255,0.4);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  border: 1px solid rgba(124,77,255,0.06);
}

[data-md-color-scheme="slate"] .backends {
  background: rgba(13,17,23,0.4);
  border-color: rgba(179,136,255,0.08);
}

.backends__title {
  font-family: 'Outfit', sans-serif;
  font-size: 0.8rem;
  font-weight: 700;
  letter-spacing: 0.2em;
  text-transform: uppercase;
  color: var(--md-primary-fg-color);
  opacity: 0.5;
}

.backends__logos {
  display: flex;
  gap: 3rem;
  justify-content: center;
  align-items: center;
  flex-wrap: wrap;
  margin-top: 0.75rem;
}

a.backend-logo {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  text-decoration: none !important;
  color: inherit !important;
  filter: grayscale(100%) brightness(0.7);
  opacity: 0.5;
  transition: all 0.3s ease;
  cursor: pointer;
}

a.backend-logo:hover {
  filter: grayscale(0%) brightness(1);
  opacity: 1;
  transform: translateY(-2px);
  text-decoration: none !important;
}

.backend-logo svg {
  width: 34px;
  height: 34px;
}

.backend-logo .jax-logo-img {
  height: 32px;
  width: auto;
}

.backend-logo--optree svg {
  width: 38px;
  height: 38px;
}

.backend-logo span {
  font-family: 'Outfit', sans-serif;
  font-size: 1.15rem;
  font-weight: 700;
  letter-spacing: 0.02em;
}

.backend-logo--numpy:hover span { color: #4DABCF; }
.backend-logo--jax:hover span { color: #5e97f6; }
.backend-logo--torch:hover span { color: #EE4C2C; }
.backend-logo--optree:hover span { color: #4CAF50; }

/* ── Responsive ── */
@media (max-width: 600px) {
  .hero__letter { font-size: 2.6rem; }
  .hero__logo canvas { width: 140px; height: 140px; }
  .backends__logos { gap: 1.5rem; }
}
</style>

<div id="shapix-visual"></div>

<div class="hero">
<div class="hero__content" markdown>

<div class="hero__logo">
<canvas id="shapix-logo" width="200" height="200"></canvas>
<div class="hero__title" id="shapix-title">Shapix</div>
</div>

<div class="hero__tagline">Runtime shape checking for the array age</div>

<p class="hero__subtitle">
Elegant shape and dtype validation for NumPy, JAX, and PyTorch arrays — powered by <a href="https://github.com/beartype/beartype" class="beartype-link"><img src="assets/images/beartype_bear.png" alt="beartype" class="beartype-logo-inline">beartype</a>.
</p>

<div class="hero__actions">
<a href="getting-started/" class="md-button md-button--primary">Get Started</a>
<a href="api/" class="md-button">API Reference</a>
<a href="https://github.com/acecchini/shapix" class="md-button">GitHub</a>
</div>

</div>
</div>

<div class="features-section" markdown>

<div class="features-grid" markdown>

<div class="feature-card" markdown>

### :material-lightning-bolt: Zero Boilerplate

Works with standard `@beartype` decorators and `beartype.claw` import hooks. No custom decorator required.

</div>

<div class="feature-card" markdown>

### :material-link-variant: Cross-Argument Consistency

Named dimensions are enforced across all parameters and the return value within a single function call.

</div>

<div class="feature-card" markdown>

### :material-check-decagram: Static Type Checker Friendly

Under `TYPE_CHECKING`, array types resolve to proper `NDArray` / `Array` / `Tensor` aliases. Works with pyright, mypy, and ty.

</div>

<div class="feature-card" markdown>

### :material-book-open-variant: Readable Annotations

`F32[N, C, H, W]` reads like documentation. No string parsing, no magic syntax.

</div>

<div class="feature-card" markdown>

### :material-cog: Full BeartypeConf Support

Unlike jaxtyping, shapix doesn't replace your beartype configuration. Full `BeartypeConf` support out of the box.

</div>

<div class="feature-card" markdown>

### :material-shield-lock: Thread-Safe

Each thread gets independent dimension bindings via `threading.local()`. Safe for parallel workloads.

</div>

</div>

</div>

<div class="backends">

<div class="backends__title">Supported Backends</div>

<div class="backends__logos">

<a href="https://numpy.org/doc/stable/" class="backend-logo backend-logo--numpy" target="_blank" rel="noopener">
<svg viewBox="0 0 24 24" fill="#4DABCF"><path d="M10.315 4.876L6.3048 2.8517l-4.401 2.1965 4.1186 2.0683zm1.8381.9277l4.2045 2.1223-4.3622 2.1906-4.125-2.0718zm5.6153-2.9213l4.3193 2.1658-3.863 1.9402-4.2131-2.1252zm-1.859-.9329L12.021 0 8.1742 1.9193l4.0068 2.0208zm-3.0401 16.7443V24l4.7107-2.3507-.0053-5.3085zm4.7037-4.2057l-.0052-5.2528-4.6985 2.3356v5.2546zm5.6553-.9845v5.327l-4.0178 2.0052-.0029-5.3028zm0-1.8626V6.4214l-4.0253 2.001.0034 5.2633zM11.2062 11.571L8.0333 9.9756v6.895s-3.8804-8.2564-4.2399-8.998c-.0463-.0957-.2371-.2007-.2858-.2262C2.8118 7.2812.773 6.2485.773 6.2485V18.43l2.8204 1.5076v-6.3674s3.8392 7.3775 3.878 7.458c.0389.0807.4245.8582.8362 1.1314.5485.363 2.8992 1.7766 2.8992 1.7766z"/></svg>
<span>NumPy</span>
</a>

<a href="https://jax.readthedocs.io/" class="backend-logo backend-logo--jax" target="_blank" rel="noopener">
<img src="assets/images/jax_logo.svg" alt="JAX" class="jax-logo-img">
<span>JAX</span>
</a>

<a href="https://pytorch.org/docs/stable/" class="backend-logo backend-logo--torch" target="_blank" rel="noopener">
<svg viewBox="0 0 24 24" fill="#EE4C2C"><path d="M12.005 0L4.952 7.053a9.865 9.865 0 000 14.022 9.866 9.866 0 0014.022 0c3.984-3.9 3.986-10.205.085-14.023l-1.744 1.743c2.904 2.905 2.904 7.634 0 10.538s-7.634 2.904-10.538 0-2.904-7.634 0-10.538l4.647-4.646.582-.665zm3.568 3.899a1.327 1.327 0 00-1.327 1.327 1.327 1.327 0 001.327 1.328A1.327 1.327 0 0016.9 5.226 1.327 1.327 0 0015.573 3.9z"/></svg>
<span>PyTorch</span>
</a>

<a href="https://optree.readthedocs.io/" class="backend-logo backend-logo--optree" target="_blank" rel="noopener">
<svg viewBox="0 0 24 24" fill="#4CAF50"><path d="M17 16l-4-4V8.82C14.16 8.4 15 7.3 15 6c0-1.66-1.34-3-3-3S9 4.34 9 6c0 1.3.84 2.4 2 2.82V12l-4 4H3v5h5v-3.05l4-4.2 4 4.2V21h5v-5h-4z"/></svg>
<span>OpTree</span>
</a>

</div>

</div>
