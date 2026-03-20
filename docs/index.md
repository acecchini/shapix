---
title: Shapix
description: Elegant runtime shape and dtype checking for NumPy, JAX, and PyTorch arrays — powered by beartype.
hide:
  - navigation
  - toc
---

<style>
.md-typeset h1 { display: none; }

.hero {
  text-align: center;
  padding: 0 1rem 2rem;
}

.hero__visual {
  width: 100%;
  max-width: 960px;
  margin: 0 auto 2rem;
  border-radius: 16px;
  overflow: hidden;
}

.hero__logo {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 0.8rem;
  margin-bottom: 0.5rem;
  perspective: 800px;
}

.hero__logo canvas {
  filter: drop-shadow(0 0 16px rgba(124, 77, 255, 0.5));
}

.hero__title {
  display: flex;
  gap: 0;
  perspective: 600px;
}

.hero__letter {
  display: inline-block;
  font-size: 4rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  background: linear-gradient(135deg, #7c4dff 0%, #b388ff 40%, #ea80fc 70%, #7c4dff 100%);
  background-size: 300% 300%;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  animation: letterWave 3s ease-in-out infinite, shimmer 4s ease-in-out infinite;
  transform-style: preserve-3d;
  filter: drop-shadow(0 4px 8px rgba(124, 77, 255, 0.3));
}

@keyframes letterWave {
  0%, 100% { transform: translateY(0) rotateX(0deg) rotateY(0deg); }
  25% { transform: translateY(-8px) rotateX(10deg) rotateY(-5deg); }
  75% { transform: translateY(4px) rotateX(-5deg) rotateY(3deg); }
}

@keyframes shimmer {
  0%, 100% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
}

.hero__tagline {
  font-size: 0.85rem;
  font-weight: 600;
  letter-spacing: 0.15em;
  text-transform: uppercase;
  color: var(--md-primary-fg-color);
  margin-bottom: 1rem;
  opacity: 0.7;
}

.hero__subtitle {
  font-size: 1.25rem;
  color: var(--md-default-fg-color--light);
  max-width: 640px;
  margin: 0 auto 2rem;
  line-height: 1.6;
}

.hero__actions {
  display: flex;
  gap: 1rem;
  justify-content: center;
  flex-wrap: wrap;
  margin-bottom: 3rem;
}

.hero__actions .md-button {
  font-size: 1rem;
  padding: 0.7rem 2rem;
  border-radius: 2rem;
}

.hero__code {
  max-width: 640px;
  margin: 0 auto;
  text-align: left;
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

.backends {
  text-align: center;
  padding: 2rem 0;
}

.backends__logos {
  display: flex;
  gap: 3rem;
  justify-content: center;
  align-items: center;
  flex-wrap: wrap;
  margin-top: 1rem;
}

.backends__logos span {
  font-size: 1.3rem;
  font-weight: 600;
  color: var(--md-default-fg-color--light);
  letter-spacing: 0.02em;
}
</style>

<div class="hero" markdown>

<div class="hero__visual" id="shapix-visual"></div>

<div class="hero__logo">
<canvas id="shapix-logo" width="80" height="80"></canvas>
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

<div class="hero__code">

```python
from beartype import beartype
from shapix import N, C, H, W
from shapix.numpy import F32

@beartype
def conv2d(x: F32[N, C, H, W], weight: F32[C, C, 3, 3]) -> F32[N, C, H, W]:
    ...
```

</div>

</div>

---

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

---

<div class="backends" markdown>

### Supported Backends

<div class="backends__logos">
<span>NumPy</span>
<span>JAX</span>
<span>PyTorch</span>
</div>

</div>
