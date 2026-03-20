/**
 * Shapix Tesseract — A rotating 4D hypercube visualization.
 *
 * The tesseract represents multi-dimensional arrays (N, C, H, W):
 * each axis of the 4D cube maps to a dimension that shapix validates.
 */
;(function () {
  'use strict'

  const canvas = document.getElementById('shapix-tesseract')
  if (!canvas) return

  const ctx = canvas.getContext('2d')
  let W, H, dpr
  let animId = null
  let mouseX = 0.5, mouseY = 0.5  // normalized 0–1
  let isHovering = false

  // ── Theme detection ───────────────────────────────────────────────
  function isDark() {
    const el = document.querySelector('[data-md-color-scheme]')
    return el && el.getAttribute('data-md-color-scheme') === 'slate'
  }

  // Brand palette
  const COLORS = {
    purple:  [124, 77, 255],   // #7c4dff
    lilac:   [179, 136, 255],  // #b388ff
    pink:    [234, 128, 252],  // #ea80fc
    white:   [255, 255, 255],
  }

  function rgba(c, a) { return `rgba(${c[0]},${c[1]},${c[2]},${a})` }

  // ── 4D Hypercube geometry ─────────────────────────────────────────
  // 16 vertices: every combination of (±1, ±1, ±1, ±1)
  const verts = []
  for (let i = 0; i < 16; i++) {
    verts.push([
      (i & 1) ? 1 : -1,
      (i & 2) ? 1 : -1,
      (i & 4) ? 1 : -1,
      (i & 8) ? 1 : -1,
    ])
  }

  // 32 edges: pairs differing in exactly 1 coordinate
  const edges = []
  for (let i = 0; i < 16; i++) {
    for (let j = i + 1; j < 16; j++) {
      let d = 0
      for (let k = 0; k < 4; k++) if (verts[i][k] !== verts[j][k]) d++
      if (d === 1) edges.push([i, j])
    }
  }

  // Dimension labels on 4 selected vertices (the +1 end of each axis)
  // vertex 1  = (+1,-1,-1,-1) → N
  // vertex 2  = (-1,+1,-1,-1) → C
  // vertex 4  = (-1,-1,+1,-1) → H
  // vertex 8  = (-1,-1,-1,+1) → W
  const dimLabels = [
    { idx: 0b0001, label: 'N' },
    { idx: 0b0010, label: 'C' },
    { idx: 0b0100, label: 'H' },
    { idx: 0b1000, label: 'W' },
  ]

  // ── 4D Rotation ───────────────────────────────────────────────────
  function rot4(v, aXW, aYZ, aXY, aZW) {
    let [x, y, z, w] = v
    let c, s

    c = Math.cos(aXW); s = Math.sin(aXW);
    [x, w] = [x * c - w * s, x * s + w * c]

    c = Math.cos(aYZ); s = Math.sin(aYZ);
    [y, z] = [y * c - z * s, y * s + z * c]

    c = Math.cos(aXY); s = Math.sin(aXY);
    [x, y] = [x * c - y * s, x * s + y * c]

    c = Math.cos(aZW); s = Math.sin(aZW);
    [z, w] = [z * c - w * s, z * s + w * c]

    return [x, y, z, w]
  }

  // ── Projection 4D → 2D (two-stage perspective) ───────────────────
  function project(v4) {
    const d4 = 3.2  // 4D camera distance
    const d3 = 3.2  // 3D camera distance

    const w4 = 1 / (d4 - v4[3])
    const x3 = v4[0] * w4
    const y3 = v4[1] * w4
    const z3 = v4[2] * w4

    const w3 = 1 / (d3 - z3)
    return {
      x: x3 * w3,
      y: y3 * w3,
      depth: w4 * d4,  // normalized depth (0.5 = near, 1.5 = far)
    }
  }

  // ── Resize handler ────────────────────────────────────────────────
  function resize() {
    const rect = canvas.parentElement.getBoundingClientRect()
    dpr = window.devicePixelRatio || 1
    W = rect.width
    H = Math.min(420, Math.max(280, rect.width * 0.55))
    canvas.width = W * dpr
    canvas.height = H * dpr
    canvas.style.width = W + 'px'
    canvas.style.height = H + 'px'
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
  }

  // ── Mouse interaction ─────────────────────────────────────────────
  canvas.addEventListener('mousemove', function (e) {
    const rect = canvas.getBoundingClientRect()
    mouseX = (e.clientX - rect.left) / rect.width
    mouseY = (e.clientY - rect.top) / rect.height
    isHovering = true
  })
  canvas.addEventListener('mouseleave', function () {
    isHovering = false
  })

  // ── Render loop ───────────────────────────────────────────────────
  function render(time) {
    const t = time * 0.001
    const dark = isDark()

    ctx.clearRect(0, 0, W, H)

    const scale = Math.min(W, H) * 0.3
    const cx = W / 2
    const cy = H / 2

    // Base rotation speeds + subtle mouse influence
    let mxOff = 0, myOff = 0
    if (isHovering) {
      mxOff = (mouseX - 0.5) * 0.3
      myOff = (mouseY - 0.5) * 0.3
    }

    const aXW = t * 0.31 + mxOff
    const aYZ = t * 0.23 + myOff
    const aXY = t * 0.17
    const aZW = t * 0.13

    // Project all vertices
    const pts = verts.map(function (v) {
      const r = rot4(v, aXW, aYZ, aXY, aZW)
      const p = project(r)
      return {
        x: cx + p.x * scale,
        y: cy + p.y * scale,
        depth: p.depth,
      }
    })

    // Sort edges by average depth (far first → painter's algorithm)
    const sortedEdges = edges.slice().sort(function (a, b) {
      return (pts[a[0]].depth + pts[a[1]].depth) - (pts[b[0]].depth + pts[b[1]].depth)
    })

    // ── Draw edges ──────────────────────────────────────────────
    sortedEdges.forEach(function (e) {
      const p0 = pts[e[0]]
      const p1 = pts[e[1]]
      const d = (p0.depth + p1.depth) / 2
      const alpha = Math.max(0.06, Math.min(0.8, d * 0.5))

      // Glow layer
      ctx.save()
      ctx.shadowColor = rgba(COLORS.lilac, alpha * 0.5)
      ctx.shadowBlur = 8 + d * 6

      const grad = ctx.createLinearGradient(p0.x, p0.y, p1.x, p1.y)
      grad.addColorStop(0, rgba(COLORS.purple, alpha))
      grad.addColorStop(0.5, rgba(COLORS.lilac, alpha))
      grad.addColorStop(1, rgba(COLORS.pink, alpha))

      ctx.beginPath()
      ctx.moveTo(p0.x, p0.y)
      ctx.lineTo(p1.x, p1.y)
      ctx.strokeStyle = grad
      ctx.lineWidth = 1 + d * 1.5
      ctx.stroke()
      ctx.restore()
    })

    // ── Draw vertices ───────────────────────────────────────────
    // Sort vertices by depth (far first)
    const sortedVerts = pts.map(function (p, i) { return { p: p, i: i } })
      .sort(function (a, b) { return a.p.depth - b.p.depth })

    sortedVerts.forEach(function (item) {
      const p = item.p
      const r = 1.5 + p.depth * 2.5
      const alpha = Math.max(0.1, Math.min(1, p.depth * 0.6))

      // Outer glow
      const glow = ctx.createRadialGradient(p.x, p.y, 0, p.x, p.y, r * 5)
      glow.addColorStop(0, rgba(COLORS.lilac, alpha * 0.3))
      glow.addColorStop(1, rgba(COLORS.lilac, 0))
      ctx.beginPath()
      ctx.arc(p.x, p.y, r * 5, 0, Math.PI * 2)
      ctx.fillStyle = glow
      ctx.fill()

      // Core dot
      ctx.beginPath()
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2)
      ctx.fillStyle = rgba(COLORS.pink, alpha)
      ctx.fill()
    })

    // ── Dimension labels ────────────────────────────────────────
    const labelColor = dark
      ? rgba(COLORS.white, 0.85)
      : rgba([50, 20, 80], 0.85)

    const labelBg = dark
      ? 'rgba(13, 17, 23, 0.7)'
      : 'rgba(255, 255, 255, 0.7)'

    dimLabels.forEach(function (dl) {
      const p = pts[dl.idx]
      const alpha = Math.max(0.15, Math.min(1, p.depth * 0.7))

      ctx.save()
      ctx.globalAlpha = alpha
      ctx.font = '700 14px "Inter", "SF Pro", system-ui, sans-serif'
      ctx.textAlign = 'center'
      ctx.textBaseline = 'middle'

      const tx = p.x + 14
      const ty = p.y - 14

      // Background pill
      const m = ctx.measureText(dl.label)
      const pw = m.width + 12
      const ph = 20
      ctx.fillStyle = labelBg
      ctx.beginPath()
      ctx.roundRect(tx - pw / 2, ty - ph / 2, pw, ph, 6)
      ctx.fill()

      // Border
      ctx.strokeStyle = rgba(COLORS.lilac, 0.4)
      ctx.lineWidth = 1
      ctx.stroke()

      // Text
      ctx.fillStyle = labelColor
      ctx.fillText(dl.label, tx, ty)

      // Connecting line
      ctx.beginPath()
      ctx.moveTo(p.x, p.y)
      ctx.lineTo(tx - pw / 2, ty)
      ctx.strokeStyle = rgba(COLORS.lilac, alpha * 0.3)
      ctx.lineWidth = 1
      ctx.setLineDash([3, 3])
      ctx.stroke()
      ctx.setLineDash([])

      ctx.restore()
    })

    animId = requestAnimationFrame(render)
  }

  // ── Init ──────────────────────────────────────────────────────────
  resize()
  window.addEventListener('resize', resize)
  animId = requestAnimationFrame(render)

  // Pause when off-screen for performance
  if (typeof IntersectionObserver !== 'undefined') {
    new IntersectionObserver(function (entries) {
      if (entries[0].isIntersecting) {
        if (!animId) animId = requestAnimationFrame(render)
      } else {
        if (animId) { cancelAnimationFrame(animId); animId = null }
      }
    }).observe(canvas)
  }

  // Re-render on theme toggle
  new MutationObserver(function () {
    // theme changed — next frame will pick up isDark()
  }).observe(document.querySelector('[data-md-color-scheme]') || document.body, {
    attributes: true,
    attributeFilter: ['data-md-color-scheme'],
  })
})()
