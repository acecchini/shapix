/**
 * Shapix Tesseract — Immersive 4D hypercube visualization.
 *
 * A rotating tesseract rendered with Three.js + UnrealBloomPass,
 * representing the multi-dimensional arrays (N, C, H, W) that shapix validates.
 */
;(async function () {
  'use strict'

  const container = document.getElementById('shapix-tesseract')
  if (!container) return

  // ── Dynamic Three.js imports ──────────────────────────────────────
  const THREE = await import('three')
  const { EffectComposer } = await import('three/addons/postprocessing/EffectComposer.js')
  const { RenderPass } = await import('three/addons/postprocessing/RenderPass.js')
  const { UnrealBloomPass } = await import('three/addons/postprocessing/UnrealBloomPass.js')
  const { OutputPass } = await import('three/addons/postprocessing/OutputPass.js')

  // ── Brand palette ─────────────────────────────────────────────────
  const PURPLE = new THREE.Color(0x7c4dff)
  const LILAC  = new THREE.Color(0xb388ff)
  const PINK   = new THREE.Color(0xea80fc)
  const WHITE  = new THREE.Color(0xffffff)

  // ── Theme detection ───────────────────────────────────────────────
  function isDark() {
    const el = document.querySelector('[data-md-color-scheme]')
    return el && el.getAttribute('data-md-color-scheme') === 'slate'
  }

  // ── Scene setup ───────────────────────────────────────────────────
  const scene = new THREE.Scene()
  const camera = new THREE.PerspectiveCamera(50, 1, 0.1, 100)
  camera.position.set(0, 0, 6)

  const renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  })
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  renderer.toneMapping = THREE.ACESFilmicToneMapping
  renderer.toneMappingExposure = 1.2
  container.appendChild(renderer.domElement)
  renderer.domElement.style.display = 'block'
  renderer.domElement.style.cursor = 'crosshair'

  // ── Postprocessing (bloom) ────────────────────────────────────────
  const composer = new EffectComposer(renderer)
  composer.addPass(new RenderPass(scene, camera))

  const bloomPass = new UnrealBloomPass(
    new THREE.Vector2(window.innerWidth, window.innerHeight),
    1.8,   // strength
    0.6,   // radius
    0.15,  // threshold
  )
  composer.addPass(bloomPass)
  composer.addPass(new OutputPass())

  // ── 4D Hypercube geometry ─────────────────────────────────────────
  // 16 vertices of (±1, ±1, ±1, ±1)
  const verts4D = []
  for (let i = 0; i < 16; i++) {
    verts4D.push([
      (i & 1) ? 1 : -1,
      (i & 2) ? 1 : -1,
      (i & 4) ? 1 : -1,
      (i & 8) ? 1 : -1,
    ])
  }

  // 32 edges: pairs differing in exactly 1 coordinate
  const edgePairs = []
  for (let i = 0; i < 16; i++) {
    for (let j = i + 1; j < 16; j++) {
      let d = 0
      for (let k = 0; k < 4; k++) if (verts4D[i][k] !== verts4D[j][k]) d++
      if (d === 1) edgePairs.push([i, j])
    }
  }

  // Dimension labels
  const dimLabels = [
    { idx: 0b0001, label: 'N', color: PURPLE },
    { idx: 0b0010, label: 'C', color: LILAC },
    { idx: 0b0100, label: 'H', color: PINK },
    { idx: 0b1000, label: 'W', color: WHITE },
  ]

  // ── 4D Rotation ───────────────────────────────────────────────────
  function rotate4D(v, aXW, aYZ, aXY, aZW) {
    let [x, y, z, w] = v
    let c, s

    c = Math.cos(aXW); s = Math.sin(aXW)
    ;[x, w] = [x * c - w * s, x * s + w * c]

    c = Math.cos(aYZ); s = Math.sin(aYZ)
    ;[y, z] = [y * c - z * s, y * s + z * c]

    c = Math.cos(aXY); s = Math.sin(aXY)
    ;[x, y] = [x * c - y * s, x * s + y * c]

    c = Math.cos(aZW); s = Math.sin(aZW)
    ;[z, w] = [z * c - w * s, z * s + w * c]

    return [x, y, z, w]
  }

  function project4Dto3D(v4) {
    const d = 3.0
    const scale = 1 / (d - v4[3])
    return new THREE.Vector3(v4[0] * scale, v4[1] * scale, v4[2] * scale)
  }

  // ── Build edge lines (BufferGeometry updated each frame) ──────────
  const edgePositions = new Float32Array(edgePairs.length * 6)
  const edgeColors = new Float32Array(edgePairs.length * 6)
  const edgeGeom = new THREE.BufferGeometry()
  edgeGeom.setAttribute('position', new THREE.BufferAttribute(edgePositions, 3))
  edgeGeom.setAttribute('color', new THREE.BufferAttribute(edgeColors, 3))

  const edgeMat = new THREE.LineBasicMaterial({
    vertexColors: true,
    transparent: true,
    opacity: 0.9,
    linewidth: 1,
  })
  const edgeMesh = new THREE.LineSegments(edgeGeom, edgeMat)
  scene.add(edgeMesh)

  // ── Vertex spheres (emissive, contribute to bloom) ────────────────
  const vertSpheres = []
  const sphereGeom = new THREE.SphereGeometry(0.04, 12, 12)

  for (let i = 0; i < 16; i++) {
    const mat = new THREE.MeshBasicMaterial({ color: LILAC, transparent: true })
    const mesh = new THREE.Mesh(sphereGeom, mat)
    scene.add(mesh)
    vertSpheres.push(mesh)
  }

  // ── Glow spheres (larger, faint, for bloom contribution) ──────────
  const glowSpheres = []
  const glowGeom = new THREE.SphereGeometry(0.12, 8, 8)

  for (let i = 0; i < 16; i++) {
    const mat = new THREE.MeshBasicMaterial({
      color: PURPLE,
      transparent: true,
      opacity: 0.25,
    })
    const mesh = new THREE.Mesh(glowGeom, mat)
    scene.add(mesh)
    glowSpheres.push(mesh)
  }

  // ── Particle field (ambient floating particles) ───────────────────
  const PARTICLE_COUNT = 600
  const particlePositions = new Float32Array(PARTICLE_COUNT * 3)
  const particleSizes = new Float32Array(PARTICLE_COUNT)
  const particleVelocities = []

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particlePositions[i * 3] = (Math.random() - 0.5) * 12
    particlePositions[i * 3 + 1] = (Math.random() - 0.5) * 8
    particlePositions[i * 3 + 2] = (Math.random() - 0.5) * 8
    particleSizes[i] = Math.random() * 3 + 0.5
    particleVelocities.push({
      x: (Math.random() - 0.5) * 0.003,
      y: (Math.random() - 0.5) * 0.003,
      z: (Math.random() - 0.5) * 0.003,
    })
  }

  const particleGeom = new THREE.BufferGeometry()
  particleGeom.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3))
  particleGeom.setAttribute('size', new THREE.BufferAttribute(particleSizes, 1))

  const particleMat = new THREE.PointsMaterial({
    color: LILAC,
    size: 0.03,
    transparent: true,
    opacity: 0.4,
    sizeAttenuation: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  })
  const particles = new THREE.Points(particleGeom, particleMat)
  scene.add(particles)

  // ── Dimension label sprites ───────────────────────────────────────
  const labelSprites = []

  function createLabelSprite(text, color) {
    const canvas = document.createElement('canvas')
    const size = 128
    canvas.width = size
    canvas.height = size
    const c = canvas.getContext('2d')

    // Background pill
    c.fillStyle = 'rgba(0, 0, 0, 0.5)'
    c.beginPath()
    c.roundRect(size * 0.15, size * 0.25, size * 0.7, size * 0.5, 12)
    c.fill()

    // Border
    c.strokeStyle = `rgba(${Math.round(color.r * 255)}, ${Math.round(color.g * 255)}, ${Math.round(color.b * 255)}, 0.6)`
    c.lineWidth = 2
    c.stroke()

    // Text
    c.fillStyle = '#ffffff'
    c.font = 'bold 48px Inter, SF Pro, system-ui, sans-serif'
    c.textAlign = 'center'
    c.textBaseline = 'middle'
    c.fillText(text, size / 2, size / 2)

    const texture = new THREE.CanvasTexture(canvas)
    texture.minFilter = THREE.LinearFilter
    const mat = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
      depthTest: false,
      blending: THREE.NormalBlending,
    })
    const sprite = new THREE.Sprite(mat)
    sprite.scale.set(0.5, 0.5, 1)
    return sprite
  }

  dimLabels.forEach(function (dl) {
    const sprite = createLabelSprite(dl.label, dl.color)
    scene.add(sprite)
    labelSprites.push({ sprite: sprite, idx: dl.idx })
  })

  // ── Connecting lines for labels ───────────────────────────────────
  const labelLinePositions = new Float32Array(dimLabels.length * 6)
  const labelLineGeom = new THREE.BufferGeometry()
  labelLineGeom.setAttribute('position', new THREE.BufferAttribute(labelLinePositions, 3))

  const labelLineMat = new THREE.LineBasicMaterial({
    color: LILAC,
    transparent: true,
    opacity: 0.2,
  })
  // Use THREE.LineSegments for pairs of points
  const labelLines = new THREE.LineSegments(labelLineGeom, labelLineMat)
  scene.add(labelLines)

  // ── Mouse tracking ────────────────────────────────────────────────
  let mouseX = 0, mouseY = 0
  let targetMouseX = 0, targetMouseY = 0

  container.addEventListener('mousemove', function (e) {
    const rect = container.getBoundingClientRect()
    targetMouseX = ((e.clientX - rect.left) / rect.width - 0.5) * 2
    targetMouseY = ((e.clientY - rect.top) / rect.height - 0.5) * 2
  })
  container.addEventListener('mouseleave', function () {
    targetMouseX = 0
    targetMouseY = 0
  })

  // ── Resize ────────────────────────────────────────────────────────
  function resize() {
    const rect = container.getBoundingClientRect()
    const w = rect.width
    const h = Math.min(520, Math.max(340, w * 0.55))
    container.style.height = h + 'px'
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    renderer.setSize(w, h)
    composer.setSize(w, h)
  }

  resize()
  window.addEventListener('resize', resize)

  // ── Animation loop ────────────────────────────────────────────────
  let animId = null

  function animate(time) {
    const t = time * 0.001

    // Smooth mouse follow
    mouseX += (targetMouseX - mouseX) * 0.05
    mouseY += (targetMouseY - mouseY) * 0.05

    // Background color based on theme
    const dark = isDark()
    scene.background = dark
      ? new THREE.Color(0x0d1117)
      : new THREE.Color(0xfafafa)

    // Bloom adjustments per theme
    bloomPass.strength = dark ? 2.0 : 1.2
    bloomPass.threshold = dark ? 0.1 : 0.3

    // 4D rotation angles (base + mouse influence)
    const aXW = t * 0.25 + mouseX * 0.3
    const aYZ = t * 0.19 + mouseY * 0.3
    const aXY = t * 0.13
    const aZW = t * 0.09

    // Compute projected 3D positions
    const pts = verts4D.map(function (v) {
      const r = rotate4D(v, aXW, aYZ, aXY, aZW)
      return project4Dto3D(r)
    })

    // Update edge positions and colors
    for (let e = 0; e < edgePairs.length; e++) {
      const [i, j] = edgePairs[e]
      const p0 = pts[i], p1 = pts[j]

      edgePositions[e * 6] = p0.x
      edgePositions[e * 6 + 1] = p0.y
      edgePositions[e * 6 + 2] = p0.z
      edgePositions[e * 6 + 3] = p1.x
      edgePositions[e * 6 + 4] = p1.y
      edgePositions[e * 6 + 5] = p1.z

      // Gradient: purple → pink based on position
      const mix0 = (p0.y + 1.5) / 3
      const mix1 = (p1.y + 1.5) / 3
      const c0 = PURPLE.clone().lerp(PINK, Math.max(0, Math.min(1, mix0)))
      const c1 = PURPLE.clone().lerp(PINK, Math.max(0, Math.min(1, mix1)))

      edgeColors[e * 6] = c0.r
      edgeColors[e * 6 + 1] = c0.g
      edgeColors[e * 6 + 2] = c0.b
      edgeColors[e * 6 + 3] = c1.r
      edgeColors[e * 6 + 4] = c1.g
      edgeColors[e * 6 + 5] = c1.b
    }

    edgeGeom.attributes.position.needsUpdate = true
    edgeGeom.attributes.color.needsUpdate = true

    // Update vertex spheres
    for (let i = 0; i < 16; i++) {
      const p = pts[i]
      vertSpheres[i].position.copy(p)
      glowSpheres[i].position.copy(p)

      // Pulsing glow
      const pulse = 0.15 + Math.sin(t * 2 + i * 0.4) * 0.1
      glowSpheres[i].material.opacity = pulse
      glowSpheres[i].scale.setScalar(1 + Math.sin(t * 1.5 + i * 0.7) * 0.3)

      // Color variation
      const mix = (p.y + 1.5) / 3
      vertSpheres[i].material.color.copy(LILAC).lerp(PINK, Math.max(0, Math.min(1, mix)))
    }

    // Update label sprites
    for (let l = 0; l < labelSprites.length; l++) {
      const ls = labelSprites[l]
      const p = pts[ls.idx]
      const offset = 0.4
      ls.sprite.position.set(p.x + offset, p.y + offset, p.z)

      // Fade based on depth
      const depth = p.z
      ls.sprite.material.opacity = Math.max(0.2, Math.min(1, 0.7 + depth * 0.5))

      // Update connecting line
      labelLinePositions[l * 6] = p.x
      labelLinePositions[l * 6 + 1] = p.y
      labelLinePositions[l * 6 + 2] = p.z
      labelLinePositions[l * 6 + 3] = p.x + offset * 0.7
      labelLinePositions[l * 6 + 4] = p.y + offset * 0.7
      labelLinePositions[l * 6 + 5] = p.z
    }
    labelLineGeom.attributes.position.needsUpdate = true

    // Animate particles
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      particlePositions[i * 3] += particleVelocities[i].x
      particlePositions[i * 3 + 1] += particleVelocities[i].y
      particlePositions[i * 3 + 2] += particleVelocities[i].z

      // Wrap around
      for (let k = 0; k < 3; k++) {
        const bound = k === 0 ? 6 : 4
        if (particlePositions[i * 3 + k] > bound) particlePositions[i * 3 + k] = -bound
        if (particlePositions[i * 3 + k] < -bound) particlePositions[i * 3 + k] = bound
      }
    }
    particleGeom.attributes.position.needsUpdate = true

    // Subtle camera movement with mouse
    camera.position.x = mouseX * 0.5
    camera.position.y = -mouseY * 0.3
    camera.lookAt(0, 0, 0)

    composer.render()
    animId = requestAnimationFrame(animate)
  }

  animId = requestAnimationFrame(animate)

  // Pause when off-screen
  if (typeof IntersectionObserver !== 'undefined') {
    new IntersectionObserver(function (entries) {
      if (entries[0].isIntersecting) {
        if (!animId) animId = requestAnimationFrame(animate)
      } else {
        if (animId) { cancelAnimationFrame(animId); animId = null }
      }
    }).observe(container)
  }
})()
