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
  padding: 3rem 1rem 2rem;
}

.hero__title {
  font-size: 3.2rem;
  font-weight: 800;
  letter-spacing: -0.02em;
  margin-bottom: 0.5rem;
  background: linear-gradient(135deg, #7c4dff 0%, #b388ff 50%, #ea80fc 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
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

<div class="hero__title">Shapix</div>

<p class="hero__subtitle">
Elegant runtime shape and dtype checking for NumPy, JAX, and PyTorch arrays — powered by <a href="https://github.com/beartype/beartype">beartype</a>.
</p>

<div class="hero__actions">

[Get Started](getting-started/index.md){ .md-button .md-button--primary }
[API Reference](api/index.md){ .md-button }
[GitHub :fontawesome-brands-github:](https://github.com/acecchini/shapix){ .md-button }

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
