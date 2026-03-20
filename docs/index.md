---
title: Shapix
description: Elegant runtime shape and dtype checking for NumPy, JAX, and PyTorch arrays — powered by beartype.
hide:
  - navigation
  - toc
---

<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&display=swap" rel="stylesheet">

<style>
.md-typeset h1 { display: none; }

/* Hide edit/view source on homepage */
.md-content__button { display: none !important; }

/* Reset content wrapper for full-bleed hero */
.md-content__inner { padding: 0; margin: 0; max-width: none; }
html { overflow-x: hidden; }

/* ── Hero: full viewport, edge to edge ── */
.hero {
  position: relative;
  width: 100vw;
  margin-left: calc(-50vw + 50%);
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 4rem 1rem 2rem;
  overflow: hidden;
}

#shapix-visual {
  position: absolute;
  inset: 0;
  z-index: 0;
}

.hero__content {
  position: relative;
  z-index: 1;
  text-align: center;
}

/* ── Logo + Title ── */
.hero__logo {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 1rem;
  margin-bottom: 0.5rem;
  perspective: 800px;
}

.hero__logo canvas {
  filter: drop-shadow(0 0 25px rgba(124, 77, 255, 0.5));
}

.hero__title {
  display: flex;
  gap: 0;
  perspective: 600px;
}

/* ── 3D semi-transparent glass letters ── */
.hero__letter {
  display: inline-block;
  font-family: 'Space Grotesk', sans-serif;
  font-size: 5.5rem;
  font-weight: 800;
  letter-spacing: -0.03em;
  background: linear-gradient(
    180deg,
    rgba(179, 136, 255, 0.45) 0%,
    rgba(124, 77, 255, 0.2) 45%,
    rgba(234, 128, 252, 0.35) 100%
  );
  background-size: 100% 100%;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  -webkit-text-stroke: 0.8px rgba(179, 136, 255, 0.35);
  filter:
    drop-shadow(0 1px 0 rgba(124, 77, 255, 0.12))
    drop-shadow(0 3px 1px rgba(124, 77, 255, 0.08))
    drop-shadow(0 6px 3px rgba(124, 77, 255, 0.06))
    drop-shadow(0 0 40px rgba(124, 77, 255, 0.35))
    drop-shadow(0 0 80px rgba(179, 136, 255, 0.15));
  animation: letterWave 3s ease-in-out infinite;
  transform-style: preserve-3d;
}

@keyframes letterWave {
  0%, 100% { transform: translateY(0) rotateX(0deg) rotateY(0deg); }
  25% { transform: translateY(-10px) rotateX(12deg) rotateY(-3deg); }
  75% { transform: translateY(5px) rotateX(-6deg) rotateY(2deg); }
}

/* ── Tagline & Subtitle ── */
.hero__tagline {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--md-primary-fg-color);
  margin-bottom: 1rem;
  opacity: 0.6;
}

.hero__subtitle {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.2rem;
  font-weight: 500;
  color: var(--md-default-fg-color--light);
  max-width: 580px;
  margin: 0 auto 2.5rem;
  line-height: 1.7;
}

/* ── Buttons ── */
.hero__actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 2rem;
}

.hero__actions .md-button {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 0.95rem;
  font-weight: 600;
  padding: 0.7rem 2rem;
  border-radius: 2rem;
}

/* ── Features grid ── */
.features-section {
  max-width: 61rem;
  margin: 0 auto;
  padding: 0 1.5rem;
}

.features-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1.5rem;
  padding: 2rem 0;
}

.feature-card {
  padding: 1.5rem;
  border-radius: 12px;
  border: 1px solid var(--md-default-fg-color--lightest);
  transition: border-color 0.2s, box-shadow 0.2s;
}

.feature-card:hover {
  border-color: var(--md-accent-fg-color);
  box-shadow: 0 4px 24px rgba(124, 77, 255, 0.08);
}

.feature-card h3 {
  margin-top: 0;
  font-size: 1.05rem;
}

.feature-card p {
  color: var(--md-default-fg-color--light);
  font-size: 0.9rem;
  margin-bottom: 0;
}

/* ── Backends ── */
.backends {
  text-align: center;
  padding: 2rem 0 3rem;
  max-width: 61rem;
  margin: 0 auto;
}

.backends__logos {
  display: flex;
  gap: 3rem;
  justify-content: center;
  align-items: center;
  flex-wrap: wrap;
  margin-top: 1.5rem;
}

.backend-logo {
  display: flex;
  align-items: center;
  gap: 0.6rem;
  filter: grayscale(100%) brightness(0.7);
  opacity: 0.5;
  transition: all 0.3s ease;
  cursor: default;
}

.backend-logo:hover {
  filter: grayscale(0%) brightness(1);
  opacity: 1;
  transform: translateY(-2px);
}

.backend-logo svg {
  width: 32px;
  height: 32px;
}

.backend-logo span {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 1.15rem;
  font-weight: 600;
  letter-spacing: 0.02em;
}

.backend-logo--numpy span { color: #4DABCF; }
.backend-logo--jax span { color: #5B4B8A; }
.backend-logo--torch span { color: #EE4C2C; }
</style>

<div class="hero">
<div id="shapix-visual"></div>
<div class="hero__content" markdown>

<div class="hero__logo">
<canvas id="shapix-logo" width="140" height="140"></canvas>
<div class="hero__title" id="shapix-title">Shapix</div>
</div>

<div class="hero__tagline">Runtime shape checking for the array age</div>

<p class="hero__subtitle">
Elegant shape and dtype validation for NumPy, JAX, and PyTorch arrays — powered by <a href="https://github.com/beartype/beartype">beartype</a>.
</p>

<div class="hero__actions">

[Get Started](getting-started/index.md){ .md-button .md-button--primary }
[API Reference](api/index.md){ .md-button }
[GitHub](https://github.com/acecchini/shapix){ .md-button }

</div>

</div>
</div>

---

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

---

<div class="backends" markdown>

### Supported Backends

<div class="backends__logos">

<div class="backend-logo backend-logo--numpy">
<svg viewBox="0 0 128 128" fill="#4DABCF"><path d="M54.7 26.5L37 17.2 18.5 27.5l17.8 9.3 18.4-10.3zm3.7 1.8L40 38.5v43.3l18.4-10.3V28.3zm-22 10.2L18.5 48.8v43.3l17.9-10.3V38.5zm40.2-33L58.7 4.2l-18 10.1 17.8 9.3 18-8.6zm3.7 1.8L62.4 17.5v43.3l17.9-10.3V7.3zm-22 10.2L40.4 27.8v43.3l17.9-10.3V17.5z"/></svg>
<span>NumPy</span>
</div>

<div class="backend-logo backend-logo--jax">
<svg viewBox="0 0 128 128" fill="#5B4B8A"><path d="M30 32h14l20 32-20 32H30l20-32-20-32zm34 0h14l20 32-20 32H64l20-32-20-32z"/></svg>
<span>JAX</span>
</div>

<div class="backend-logo backend-logo--torch">
<svg viewBox="0 0 128 128" fill="#EE4C2C"><path d="M64 8L48 28v12c-14 8-24 24-24 42 0 26.5 21.5 48 48 48s48-21.5 48-48c0-30-24-52-48-68l-8-6zm4 24a6 6 0 110 12 6 6 0 010-12z"/></svg>
<span>PyTorch</span>
</div>

</div>

</div>
