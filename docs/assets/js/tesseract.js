/**
 * Shapix Visual — 3D curvilinear light rays flowing through space.
 *
 * Renders distinct thin light ribbons as 3D Lissajous curves projected
 * with perspective. Each ray has its own lifecycle, color gradient,
 * and smooth parametric motion. Pure WebGL2, zero dependencies.
 */
;(function () {
  'use strict'

  function start() {
    var el = document.getElementById('shapix-visual')
    if (!el) return
    if (!el.offsetWidth) { requestAnimationFrame(function(){ setTimeout(start,30) }); return }

    initBg(el)
    initLogo()
    initTitle()
  }

  /* ================================================================
   * BACKGROUND — 3D curvilinear light rays
   * ================================================================ */
  function initBg(el) {
    var canvas = document.createElement('canvas')
    canvas.style.cssText = 'position:absolute;top:0;left:0;width:100%;height:100%;'
    el.appendChild(canvas)

    var gl = canvas.getContext('webgl2', { antialias: false, alpha: false })
    if (!gl) {
      el.style.background = 'linear-gradient(135deg,#1a0533,#0d1117)'
      return
    }

    var VERT = '#version 300 es\nin vec2 p;void main(){gl_Position=vec4(p,0,1);}'

    // Fragment shader: 3D curvilinear light rays
    var FRAG = `#version 300 es
precision highp float;
uniform float T;
uniform vec2 R;
uniform vec2 M;
uniform float D;
out vec4 O;

#define NC 10   // number of curves
#define NS 50   // samples per curve

// 3D Lissajous curve position for curve i at parameter s
vec3 curve(int i, float s, float t) {
  float fi = float(i);
  float phase = fi * 1.618;  // golden ratio spacing

  // Each curve has unique frequencies and phases
  float fx = 1.0 + mod(fi * 3.0, 4.0);
  float fy = 1.0 + mod(fi * 2.0 + 1.0, 5.0);
  float fz = 0.5 + mod(fi * 1.5, 3.0);

  // Time-varying drift
  float drift = t * (0.15 + fi * 0.02);

  // Amplitude modulation for organic motion
  float ax = 1.8 + sin(t * 0.1 + phase) * 0.5;
  float ay = 1.2 + cos(t * 0.13 + phase * 0.7) * 0.4;
  float az = 1.5 + sin(t * 0.08 + phase * 1.3) * 0.6;

  float x = ax * sin(s * fx * 3.14159 + drift + phase);
  float y = ay * sin(s * fy * 3.14159 + drift * 1.3 + phase * 2.1);
  float z = az * cos(s * fz * 3.14159 + drift * 0.7 + phase * 0.5) - 1.0;

  return vec3(x, y, z);
}

// Perspective projection
vec2 proj(vec3 p) {
  float d = 5.0;
  float w = 1.0 / (d - p.z);
  return vec2(p.x * w, p.y * w);
}

// Envelope: controls ray appearance/disappearance
float envelope(int i, float t) {
  float fi = float(i);
  float cycle = sin(t * (0.12 + fi * 0.015) + fi * 2.39) * 0.5 + 0.5;
  return smoothstep(0.05, 0.4, cycle) * smoothstep(1.0, 0.7, cycle);
}

void main() {
  vec2 uv = (gl_FragCoord.xy - 0.5 * R) / min(R.x, R.y);
  vec2 m = (M - 0.5) * 0.08;
  uv += m;

  vec3 col = vec3(0.0);

  // Color palette
  vec3 c1 = vec3(0.486, 0.302, 1.0);    // #7c4dff purple
  vec3 c2 = vec3(0.702, 0.533, 1.0);    // #b388ff lilac
  vec3 c3 = vec3(0.918, 0.502, 0.988);  // #ea80fc pink
  vec3 c4 = vec3(0.45, 0.88, 1.0);      // cyan accent
  vec3 c5 = vec3(1.0, 0.6, 0.9);        // warm pink

  for (int i = 0; i < NC; i++) {
    float env = envelope(i, T);
    if (env < 0.01) continue;  // skip invisible curves

    float fi = float(i);
    float minD = 1e9;
    float bestS = 0.0;
    float bestDepth = 0.0;

    // Find closest point on curve to this pixel
    for (int j = 0; j < NS; j++) {
      float s = float(j) / float(NS - 1);
      vec3 p3 = curve(i, s, T);

      // Mouse-reactive camera tilt
      p3.x += m.x * 0.5;
      p3.y += m.y * 0.3;

      vec2 p2 = proj(p3);
      float d = length(uv - p2);

      if (d < minD) {
        minD = d;
        bestS = s;
        bestDepth = p3.z;
      }
    }

    // Thin ray with very sharp Gaussian falloff
    float thickness = 300.0 + sin(T * 0.3 + fi) * 100.0;
    float intensity = exp(-minD * minD * thickness);

    // Depth-based attenuation (closer = brighter)
    float depthFade = 1.0 / (1.0 + max(-bestDepth, 0.0) * 0.3);
    intensity *= depthFade;

    // Apply envelope
    intensity *= env;

    // Power gradient along the curve that shifts with time
    float power = smoothstep(0.0, 0.15, bestS) * smoothstep(1.0, 0.85, bestS);
    power *= 0.5 + 0.5 * sin(bestS * 6.28 + T * 0.5 + fi);
    intensity *= max(power, 0.15);

    // Time-varying color gradient per curve
    float colorPhase = bestS + sin(T * 0.2 + fi * 1.3) * 0.3;
    float ci = mod(fi * 0.37 + T * 0.05, 1.0);
    vec3 base1, base2;
    if (ci < 0.25) { base1 = c1; base2 = c2; }
    else if (ci < 0.5) { base1 = c2; base2 = c3; }
    else if (ci < 0.75) { base1 = c3; base2 = c4; }
    else { base1 = c4; base2 = c5; }

    vec3 rayColor = mix(base1, base2, sin(colorPhase * 3.14) * 0.5 + 0.5);
    rayColor = mix(rayColor, vec3(1.0), intensity * 0.3);  // bright core

    col += rayColor * intensity;
  }

  // Subtle ambient glow at center
  float ambient = exp(-length(uv) * 1.5) * 0.03;
  col += vec3(0.3, 0.15, 0.5) * ambient;

  // Tone mapping
  col = col * (2.51 * col + 0.03) / (col * (2.43 * col + 0.59) + 0.14);
  col = clamp(col, 0.0, 1.0);

  // Background
  vec3 bgLight = vec3(0.97, 0.97, 0.99);
  vec3 bgDark = vec3(0.04, 0.05, 0.08);
  vec3 bg = mix(bgLight, bgDark, D);

  // Blend: visual sits on top of background
  float alpha = max(max(col.r, col.g), col.b);
  alpha = smoothstep(0.0, 0.03, alpha);
  col = mix(bg, col + bg * (1.0 - alpha), min(alpha * 2.0, 1.0));

  O = vec4(col, 1.0);
}
`

    function mk(type, src) {
      var s = gl.createShader(type)
      gl.shaderSource(s, src)
      gl.compileShader(s)
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error('[shapix]', gl.getShaderInfoLog(s))
        return null
      }
      return s
    }

    var vs = mk(gl.VERTEX_SHADER, VERT), fs = mk(gl.FRAGMENT_SHADER, FRAG)
    if (!vs || !fs) {
      canvas.remove()
      el.style.background = 'linear-gradient(135deg,#1a0533,#0d1117)'
      return
    }

    var pg = gl.createProgram()
    gl.attachShader(pg, vs); gl.attachShader(pg, fs); gl.linkProgram(pg)
    if (!gl.getProgramParameter(pg, gl.LINK_STATUS)) {
      console.error('[shapix]', gl.getProgramInfoLog(pg))
      canvas.remove()
      el.style.background = 'linear-gradient(135deg,#1a0533,#0d1117)'
      return
    }
    gl.useProgram(pg)

    var buf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW)
    var aL = gl.getAttribLocation(pg, 'p')
    gl.enableVertexAttribArray(aL)
    gl.vertexAttribPointer(aL, 2, gl.FLOAT, false, 0, 0)

    var uT=gl.getUniformLocation(pg,'T'), uR=gl.getUniformLocation(pg,'R'),
        uM=gl.getUniformLocation(pg,'M'), uD=gl.getUniformLocation(pg,'D')

    var mx=.5,my=.5,tmx=.5,tmy=.5,aid=null

    function dk(){var e=document.querySelector('[data-md-color-scheme]');return e&&e.getAttribute('data-md-color-scheme')==='slate'?1:0}

    function resize() {
      var r = el.getBoundingClientRect()
      var d = Math.min(window.devicePixelRatio || 1, 1.5) // cap for perf
      canvas.width = r.width * d; canvas.height = r.height * d
      gl.viewport(0, 0, canvas.width, canvas.height)
    }

    document.addEventListener('mousemove', function(e) {
      tmx = e.clientX / window.innerWidth
      tmy = 1 - e.clientY / window.innerHeight
    })

    function frame(t) {
      mx+=(tmx-mx)*.03; my+=(tmy-my)*.03
      gl.uniform1f(uT, t*.001)
      gl.uniform2f(uR, canvas.width, canvas.height)
      gl.uniform2f(uM, mx, my)
      gl.uniform1f(uD, dk())
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
      aid = requestAnimationFrame(frame)
    }

    resize()
    window.addEventListener('resize', resize)
    aid = requestAnimationFrame(frame)

    if (typeof IntersectionObserver !== 'undefined') {
      new IntersectionObserver(function(e) {
        if (e[0].isIntersecting) { if (!aid) aid=requestAnimationFrame(frame) }
        else { if (aid) { cancelAnimationFrame(aid); aid=null } }
      }).observe(el)
    }
  }

  /* ================================================================
   * LOGO — Animated rotating tesseract wireframe
   * ================================================================ */
  function initLogo() {
    var c = document.getElementById('shapix-logo')
    if (!c) return
    var ctx = c.getContext('2d')
    var S = 80, dpr = Math.min(window.devicePixelRatio || 1, 2)
    c.width = S * dpr; c.height = S * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    var V = []
    for (var i = 0; i < 16; i++)
      V.push([(i&1)?1:-1, (i&2)?1:-1, (i&4)?1:-1, (i&8)?1:-1])
    var E = []
    for (var i = 0; i < 16; i++)
      for (var j = i+1; j < 16; j++) {
        var d = 0
        for (var k = 0; k < 4; k++) if (V[i][k] !== V[j][k]) d++
        if (d === 1) E.push([i, j])
      }

    function rot4(v, a1, a2, a3) {
      var x=v[0],y=v[1],z=v[2],w=v[3],c1,s1
      c1=Math.cos(a1);s1=Math.sin(a1);var nx=x*c1-w*s1;w=x*s1+w*c1;x=nx
      c1=Math.cos(a2);s1=Math.sin(a2);var ny=y*c1-z*s1;z=y*s1+z*c1;y=ny
      c1=Math.cos(a3);s1=Math.sin(a3);nx=x*c1-y*s1;y=x*s1+y*c1;x=nx
      return [x,y,z,w]
    }

    function proj(v){var d=3.2,s=1/(d-v[3]),d2=3,s2=1/(d2-v[2]*s);return{x:v[0]*s*s2,y:v[1]*s*s2,d:s*d}}

    function draw() {
      var t = performance.now() * 0.001
      ctx.clearRect(0, 0, S, S)
      var cx=S/2,cy=S/2,sc=S*0.22
      var pts=V.map(function(v){var r=rot4(v,t*.5,t*.37,t*.23),p=proj(r);return{x:cx+p.x*sc,y:cy+p.y*sc,d:p.d}})

      E.slice().sort(function(a,b){return(pts[a[0]].d+pts[a[1]].d)-(pts[b[0]].d+pts[b[1]].d)}).forEach(function(e){
        var p0=pts[e[0]],p1=pts[e[1]],avg=(p0.d+p1.d)/2,alpha=Math.max(0.15,Math.min(0.9,avg*0.5))
        var g=ctx.createLinearGradient(p0.x,p0.y,p1.x,p1.y)
        g.addColorStop(0,'rgba(124,77,255,'+alpha+')');g.addColorStop(1,'rgba(234,128,252,'+alpha+')')
        ctx.beginPath();ctx.moveTo(p0.x,p0.y);ctx.lineTo(p1.x,p1.y)
        ctx.strokeStyle=g;ctx.lineWidth=0.8+avg*0.8;ctx.shadowColor='rgba(179,136,255,'+alpha*0.4+')';ctx.shadowBlur=4;ctx.stroke();ctx.shadowBlur=0
      })

      pts.forEach(function(p){
        ctx.beginPath();ctx.arc(p.x,p.y,1+p.d*1.2,0,Math.PI*2)
        ctx.fillStyle='rgba(234,128,252,'+Math.max(0.2,Math.min(1,p.d*0.6))+')';ctx.fill()
      })
      requestAnimationFrame(draw)
    }
    draw()
  }

  /* ================================================================
   * TITLE — Animated 3D wave letters
   * ================================================================ */
  function initTitle() {
    var el = document.getElementById('shapix-title')
    if (!el) return
    var text = 'Shapix'
    var html = ''
    for (var i = 0; i < text.length; i++)
      html += '<span class="hero__letter" style="animation-delay:'+i*0.12+'s">'+text[i]+'</span>'
    el.innerHTML = html
  }

  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', start)
  else
    requestAnimationFrame(start)
})()
