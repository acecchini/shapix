/**
 * Shapix Visual v4 — 3D morphing bubbles (ray-marched metaballs),
 * icosahedron logo, and 3D extruded title.
 * Two-pass WebGL2: scene FBO -> composite with subtle bloom.
 * Zero dependencies.
 */
;(function () {
  'use strict'

  var tmx = 0.5, tmy = 0.5, mx = 0.5, my = 0.5
  var tOff = Math.random() * 1000
  var _afs = []       // track all animation frame IDs for cleanup
  var _resizeCb = null // track resize listener for cleanup

  document.addEventListener('mousemove', function (e) {
    tmx = e.clientX / window.innerWidth
    tmy = 1 - e.clientY / window.innerHeight
  })
  document.addEventListener('touchmove', function (e) {
    if (e.touches.length > 0) {
      tmx = e.touches[0].clientX / window.innerWidth
      tmy = 1 - e.touches[0].clientY / window.innerHeight
    }
  }, { passive: true })

  function start() {
    var el = document.getElementById('shapix-visual')
    if (!el) return
    if (!el.offsetWidth) {
      requestAnimationFrame(function () { setTimeout(start, 30) })
      return
    }
    initBg(el)
    initLogo()
    initTitle()
  }

  /* ================================================================
   * BACKGROUND — Ray-marched 3D metaballs / morphing bubbles
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

    gl.getExtension('EXT_color_buffer_float')

    var VERT = '#version 300 es\nlayout(location=0) in vec2 p;void main(){gl_Position=vec4(p,0,1);}'

    // ── Scene shader: ray-marched morphing purple bubbles ──
    var FRAG_SCENE = [
      '#version 300 es',
      'precision highp float;',
      'uniform float T;',
      'uniform vec2 R;',
      'uniform vec2 M;',
      'uniform float D;',
      'out vec4 O;',
      '',
      '#define NB 14',
      '#define STEPS 50',
      '',
      'float smin(float a,float b,float k){',
      '  float h=clamp(.5+.5*(b-a)/k,0.,1.);',
      '  return mix(b,a,h)-k*h*(1.-h);',
      '}',
      '',
      '// Domain warp for organic shape deformation',
      'vec3 warp(vec3 p,float t){',
      '  return p+vec3(',
      '    sin(p.y*2.3+t*.3)*sin(p.z*1.7+t*.2)*.18,',
      '    cos(p.x*1.9+t*.25)*sin(p.y*2.5+t*.35)*.15,',
      '    sin(p.x*2.1+t*.4)*cos(p.z*1.4+t*.15)*.12',
      '  );',
      '}',
      '',
      'vec3 bpos(int i,float t){',
      '  float fi=float(i);',
      '  return vec3(',
      '    sin(t*(.08+fi*.02)+fi*2.1)*(2.2+.5*sin(t*.06+fi)),',
      '    cos(t*(.07+fi*.015)+fi*1.7)*(1.8+.4*cos(t*.05+fi*1.3)),',
      '    sin(t*(.09+fi*.018)+fi*3.1)*(2.5+.6*sin(t*.07+fi*.7))+cos(t*(.05+fi*.013)+fi*1.9)*1.2',
      '  );',
      '}',
      '',
      'float brad(int i,float t){',
      '  float fi=float(i);',
      '  return .35+.2*sin(t*(.15+fi*.04)+fi*2.5)+.1*sin(t*.3+fi*4.);',
      '}',
      '',
      '// Purple-dominant spectrum with hints of red, green, blue',
      'vec3 bcol(int i,float t){',
      '  float fi=float(i);',
      '  float phase=fi*.618+t*.03;',
      '  return vec3(',
      '    .4+.3*sin(phase*6.28)+.1*sin(phase*3.14+1.),',
      '    .15+.18*sin(phase*6.28+4.2)+.08*sin(phase*4.7+2.5),',
      '    .6+.3*sin(phase*6.28+1.5)+.08*cos(phase*2.9)',
      '  );',
      '}',
      '',
      'float scene(vec3 p){',
      '  vec3 wp=warp(p,T);',
      '  float d=length(wp-bpos(0,T))-brad(0,T);',
      '  for(int i=1;i<NB;i++){',
      '    d=smin(d,length(wp-bpos(i,T))-brad(i,T),1.2);',
      '  }',
      '  return d*.75;',
      '}',
      '',
      'vec3 scol(vec3 p){',
      '  vec3 c=vec3(0);float tw=0.;',
      '  for(int i=0;i<NB;i++){',
      '    float w=1./(1.+pow(length(p-bpos(i,T)),4.));',
      '    c+=bcol(i,T)*w;tw+=w;',
      '  }',
      '  return c/tw;',
      '}',
      '',
      'vec3 norm(vec3 p){',
      '  vec2 e=vec2(.002,0);',
      '  return normalize(vec3(',
      '    scene(p+e.xyy)-scene(p-e.xyy),',
      '    scene(p+e.yxy)-scene(p-e.yxy),',
      '    scene(p+e.yyx)-scene(p-e.yyx)',
      '  ));',
      '}',
      '',
      'void main(){',
      '  vec2 uv=(gl_FragCoord.xy-.5*R)/min(R.x,R.y);',
      '  vec2 m=(M-.5);',
      '',
      '  // Camera rotation: mouse + slow auto-orbit',
      '  float ry=m.x*.6,rx=m.y*.4;',
      '  float cy=cos(ry),sy=sin(ry),cx=cos(rx),sx=sin(rx);',
      '  mat3 mRot=mat3(cy,sx*sy,-cx*sy, 0.,cx,sx, sy,-sx*cy,cx*cy);',
      '  float ao=T*.04, ax=sin(T*.025)*.22;',
      '  float cao=cos(ao),sao=sin(ao),cax=cos(ax),sax=sin(ax);',
      '  mat3 aRotY=mat3(cao,0.,sao, 0.,1.,0., -sao,0.,cao);',
      '  mat3 aRotX=mat3(1.,0.,0., 0.,cax,sax, 0.,-sax,cax);',
      '  mat3 cam=mRot*aRotY*aRotX;',
      '',
      '  vec3 ro=cam*vec3(0,0,-6.);',
      '  vec3 rd=cam*normalize(vec3(uv,1.5));',
      '',
      '  // Ray march',
      '  float t=0.,minD=1e9;',
      '  vec3 minP=ro;',
      '  for(int i=0;i<STEPS;i++){',
      '    vec3 p=ro+rd*t;',
      '    float d=scene(p);',
      '    if(d<minD){minD=d;minP=p;}',
      '    if(abs(d)<.003)break;',
      '    t+=d;',
      '    if(t>14.)break;',
      '  }',
      '',
      '  vec3 col=vec3(0);',
      '',
      '  if(minD<.003){',
      '    vec3 p=ro+rd*t;',
      '    vec3 n=norm(p);',
      '    vec3 base=scol(p);',
      '    vec3 l1=normalize(vec3(.3,1.,-.5));',
      '    vec3 l2=normalize(vec3(-.6,-.3,.4));',
      '',
      '    // Half-lambert diffuse (two lights)',
      '    float diff1=dot(n,l1)*.5+.5; diff1=diff1*diff1;',
      '    float diff2=dot(n,l2)*.5+.5; diff2=diff2*diff2;',
      '',
      '    // Blinn-Phong specular',
      '    vec3 h1=normalize(l1-rd);',
      '    float spec=pow(max(dot(n,h1),0.),32.)*.4;',
      '',
      '    // Fresnel rim — tight, subtle edge',
      '    float fres=pow(1.-max(dot(n,-rd),0.),6.);',
      '',
      '    // Subsurface scatter approx',
      '    float sss=pow(max(dot(normalize(rd+l1*.5),n),0.),4.)*.25;',
      '',
      '    // Purple-tinted rim glow',
      '    vec3 rimCol=vec3(.6,.3,.9)*fres;',
      '',
      '    col=base*(.28+diff1*.7+diff2*.25)+rimCol*.2+vec3(.7,.5,1.)*spec*.6+base*sss;',
      '    col*=exp(-t*.03);',
      '  }',
      '',
      '',
      '',
      '  O=vec4(col,1);',
      '}'
    ].join('\n')

    // ── Composite shader: subtle bloom + vignette ──
    var FRAG_COMP = [
      '#version 300 es',
      'precision highp float;',
      'uniform sampler2D uScene;',
      'uniform vec2 R;',
      'uniform float T;',
      'uniform float D;',
      'out vec4 O;',
      '',
      '// Hash for star field',
      'float hash(vec2 p){return fract(sin(dot(p,vec2(127.1,311.7)))*43758.5453);}',
      '',
      'void main(){',
      '  vec2 tc=gl_FragCoord.xy/R;',
      '  vec3 col=texture(uScene,tc).rgb;',
      '',
      '  // Reinhard tone mapping',
      '  col=col/(col+1.);',
      '',
      '  // Background blend',
      '  vec3 bg=mix(vec3(.953,.933,1.),vec3(.024,.032,.065),D);',
      '  float alpha=max(max(col.r,col.g),col.b);',
      '  alpha=smoothstep(.02,.08,alpha);',
      '  float boost=mix(1.3,1.25,D);',
      '  col=mix(bg,col*boost,alpha);',
      '',
      '  // Particles — stars (dark) / dust (light)',
      '  vec2 gp=floor(gl_FragCoord.xy/2.);',
      '  float h=hash(gp);',
      '  if(D>.5&&h>.997){',
      '    float twinkle=.7+.3*sin(T*(.08+h*.15)+h*100.);',
      '    float brightness=smoothstep(.997,1.,h)*twinkle;',
      '    vec3 starCol=mix(vec3(.8,.85,1.),vec3(1.,.95,.8),hash(gp+99.));',
      '    col+=starCol*brightness*(1.-alpha)*.8;',
      '  }',
      '  if(D<.5&&h>.993){',
      '    float shimmer=.6+.4*sin(T*(.1+h*.2)+h*60.);',
      '    float brightness=smoothstep(.993,1.,h)*shimmer;',
      '    vec3 dustCol=mix(vec3(.65,.5,.9),vec3(.8,.65,1.),hash(gp+77.));',
      '    col+=dustCol*brightness*(1.-alpha)*.6;',
      '  }',
      '',
      '  O=vec4(col,1);',
      '}'
    ].join('\n')

    // ── Shader compilation helpers ──
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

    function mkProg(vSrc, fSrc) {
      var vs = mk(gl.VERTEX_SHADER, vSrc), fs = mk(gl.FRAGMENT_SHADER, fSrc)
      if (!vs || !fs) return null
      var pg = gl.createProgram()
      gl.attachShader(pg, vs); gl.attachShader(pg, fs); gl.linkProgram(pg)
      if (!gl.getProgramParameter(pg, gl.LINK_STATUS)) {
        console.error('[shapix]', gl.getProgramInfoLog(pg))
        return null
      }
      return pg
    }

    var sceneProg = mkProg(VERT, FRAG_SCENE)
    var compProg = mkProg(VERT, FRAG_COMP)
    if (!sceneProg || !compProg) {
      canvas.remove()
      el.style.background = 'linear-gradient(135deg,#1a0533,#0d1117)'
      return
    }

    var buf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)

    var uST = gl.getUniformLocation(sceneProg, 'T')
    var uSR = gl.getUniformLocation(sceneProg, 'R')
    var uSM = gl.getUniformLocation(sceneProg, 'M')
    var uSD = gl.getUniformLocation(sceneProg, 'D')
    var uCS = gl.getUniformLocation(compProg, 'uScene')
    var uCR = gl.getUniformLocation(compProg, 'R')
    var uCT = gl.getUniformLocation(compProg, 'T')
    var uCD = gl.getUniformLocation(compProg, 'D')

    var fboObj = null
    function makeFBO(w, h) {
      if (fboObj) {
        gl.deleteTexture(fboObj.tex)
        gl.deleteFramebuffer(fboObj.fbo)
      }
      var tex = gl.createTexture()
      gl.bindTexture(gl.TEXTURE_2D, tex)
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA16F, w, h, 0, gl.RGBA, gl.HALF_FLOAT, null)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      var fb = gl.createFramebuffer()
      gl.bindFramebuffer(gl.FRAMEBUFFER, fb)
      gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0)
      if (gl.checkFramebufferStatus(gl.FRAMEBUFFER) !== gl.FRAMEBUFFER_COMPLETE) {
        gl.bindTexture(gl.TEXTURE_2D, tex)
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null)
        gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0)
      }
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      fboObj = { fbo: fb, tex: tex, w: w, h: h }
    }

    function dk() {
      var e = document.querySelector('[data-md-color-scheme]')
      return e && e.getAttribute('data-md-color-scheme') === 'slate' ? 1 : 0
    }

    function resize() {
      var w = window.innerWidth, h = window.innerHeight
      var d = Math.min(window.devicePixelRatio || 1, 1.5)
      canvas.width = w * d; canvas.height = h * d
      makeFBO(canvas.width, canvas.height)
    }

    function frame(t) {
      mx += (tmx - mx) * .06; my += (tmy - my) * .06
      var time = t * .001 + tOff, dark = dk()

      gl.bindFramebuffer(gl.FRAMEBUFFER, fboObj.fbo)
      gl.viewport(0, 0, fboObj.w, fboObj.h)
      gl.useProgram(sceneProg)
      gl.uniform1f(uST, time)
      gl.uniform2f(uSR, fboObj.w, fboObj.h)
      gl.uniform2f(uSM, mx, my)
      gl.uniform1f(uSD, dark)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)

      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      gl.viewport(0, 0, canvas.width, canvas.height)
      gl.useProgram(compProg)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, fboObj.tex)
      gl.uniform1i(uCS, 0)
      gl.uniform2f(uCR, canvas.width, canvas.height)
      gl.uniform1f(uCT, time)
      gl.uniform1f(uCD, dark)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)

      _afs.push(requestAnimationFrame(frame))
    }

    resize()
    if (_resizeCb) window.removeEventListener('resize', _resizeCb)
    _resizeCb = resize
    window.addEventListener('resize', resize)
    _afs.push(requestAnimationFrame(frame))
  }

  /* ================================================================
   * LOGO — Animated tesseract with sinusoidal vertex forces + breathing
   * ================================================================ */
  function initLogo() {
    var c = document.getElementById('shapix-logo')
    if (!c) return
    var ctx = c.getContext('2d')
    var S = 220, dpr = Math.min(window.devicePixelRatio || 1, 2)
    c.width = S * dpr; c.height = S * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    // 16 tesseract vertices: all (±1, ±1, ±1, ±1)
    var V = []
    for (var i = 0; i < 16; i++)
      V.push([(i & 1) ? 1 : -1, (i & 2) ? 1 : -1, (i & 4) ? 1 : -1, (i & 8) ? 1 : -1])

    // 32 edges: vertices differing in exactly 1 coordinate
    var E = []
    for (var i = 0; i < 16; i++)
      for (var j = i + 1; j < 16; j++) {
        var diff = 0
        for (var k = 0; k < 4; k++)
          if (V[i][k] !== V[j][k]) diff++
        if (diff === 1) E.push([i, j])
      }

    // 24 square faces: fix 2 axes, vary the other 2
    var Faces = []
    for (var a1 = 0; a1 < 4; a1++)
      for (var a2 = a1 + 1; a2 < 4; a2++) {
        var fixed = []
        for (var k = 0; k < 4; k++)
          if (k !== a1 && k !== a2) fixed.push(k)
        for (var f0 = 0; f0 < 2; f0++)
          for (var f1 = 0; f1 < 2; f1++) {
            var fv0 = f0 ? 1 : -1, fv1 = f1 ? 1 : -1
            var quad = []
            for (var vi = 0; vi < 16; vi++)
              if (V[vi][fixed[0]] === fv0 && V[vi][fixed[1]] === fv1)
                quad.push(vi)
            // Sort by angle in (a1,a2) plane for proper winding
            ;(function (ax1, ax2) {
              quad.sort(function (a, b) {
                return Math.atan2(V[a][ax2], V[a][ax1]) - Math.atan2(V[b][ax2], V[b][ax1])
              })
            })(a1, a2)
            Faces.push(quad)
          }
      }

    // 4D rotation in a plane
    function rot4(v, p, q, a) {
      var co = Math.cos(a), si = Math.sin(a)
      var r = [v[0], v[1], v[2], v[3]]
      r[p] = v[p] * co - v[q] * si
      r[q] = v[p] * si + v[q] * co
      return r
    }

    function draw() {
      var t = performance.now() * 0.001 + tOff
      ctx.clearRect(0, 0, S, S)
      var cx = S / 2, cy = S / 2, sc = S * 0.30

      var floatY = Math.sin(t * 0.7) * 6
      var breath = 1 + 0.1 * Math.sin(t * 0.6)
      var tiltX = (my - 0.5) * 0.5
      var tiltY = (mx - 0.5) * 0.5

      var pts = V.map(function (v, idx) {
        // Sinusoidal vertex displacement — each vertex pulled independently
        var d = [
          v[0] + Math.sin(t * 0.4 + idx * 0.7) * 0.15,
          v[1] + Math.cos(t * 0.35 + idx * 1.1) * 0.12,
          v[2] + Math.sin(t * 0.45 + idx * 0.9) * 0.1,
          v[3] + Math.cos(t * 0.3 + idx * 1.3) * 0.13
        ]

        // Apply breathing
        d = [d[0] * breath, d[1] * breath, d[2] * breath, d[3] * breath]

        // 4D rotations in multiple planes
        var r = rot4(d, 0, 3, t * 0.3)
        r = rot4(r, 1, 2, t * 0.2)
        r = rot4(r, 0, 2, t * 0.15)
        r = rot4(r, 1, 3, t * 0.25)

        // 4D → 3D perspective projection
        var w4 = 2.8 - r[3] * 0.35
        var x3 = r[0] / w4, y3 = r[1] / w4, z3 = r[2] / w4

        // Mouse-driven 3D tilt
        var co = Math.cos(tiltX), si = Math.sin(tiltX)
        var ny = y3 * co - z3 * si, nz = y3 * si + z3 * co
        y3 = ny; z3 = nz
        co = Math.cos(tiltY); si = Math.sin(tiltY)
        var nx = x3 * co + z3 * si; nz = -x3 * si + z3 * co
        x3 = nx; z3 = nz

        // 3D → 2D perspective
        var d3 = 3.5, s = d3 / (d3 - z3)
        return { x: cx + x3 * sc * s, y: cy + y3 * sc * s + floatY, z: z3 }
      })

      // Faces sorted back to front — purple-tinted glass
      Faces.slice().sort(function (a, b) {
        var az = 0, bz = 0
        for (var i = 0; i < 4; i++) { az += pts[a[i]].z; bz += pts[b[i]].z }
        return az - bz
      }).forEach(function (f) {
        var avgZ = (pts[f[0]].z + pts[f[1]].z + pts[f[2]].z + pts[f[3]].z) / 4
        var alpha = 0.05 + (avgZ + 1) * 0.065
        alpha = Math.max(0.04, Math.min(0.18, alpha))

        ctx.beginPath()
        ctx.moveTo(pts[f[0]].x, pts[f[0]].y)
        ctx.lineTo(pts[f[1]].x, pts[f[1]].y)
        ctx.lineTo(pts[f[2]].x, pts[f[2]].y)
        ctx.lineTo(pts[f[3]].x, pts[f[3]].y)
        ctx.closePath()
        ctx.fillStyle = 'rgba(180,140,255,' + alpha + ')'
        ctx.fill()
      })

      // Edges — time-evolving purple gradient
      ctx.lineCap = 'round'
      ctx.lineJoin = 'round'

      var sortedEdges = E.slice().sort(function (a, b) {
        return (pts[a[0]].z + pts[a[1]].z) - (pts[b[0]].z + pts[b[1]].z)
      })

      sortedEdges.forEach(function (e) {
        var p0 = pts[e[0]], p1 = pts[e[1]]
        var avgZ = (p0.z + p1.z) / 2
        var alpha = 0.25 + (avgZ + 1) * 0.35
        alpha = Math.max(0.2, Math.min(0.95, alpha))

        var lineW = 1.0 + (avgZ + 1) * 0.8

        // Per-vertex color phases — gradient flows along each edge
        var ph0 = t * 0.15 + e[0] * 0.4
        var ph1 = t * 0.15 + e[1] * 0.4
        var zb = (avgZ + 1) * 30
        var r0 = Math.round(140 + 60 * Math.sin(ph0) + zb)
        var g0 = Math.round(80 + 40 * Math.sin(ph0 + 2.0) + (avgZ + 1) * 20)
        var b0 = Math.round(200 + 55 * Math.sin(ph0 + 4.0))
        var r1 = Math.round(140 + 60 * Math.sin(ph1) + zb)
        var g1 = Math.round(80 + 40 * Math.sin(ph1 + 2.0) + (avgZ + 1) * 20)
        var b1 = Math.round(200 + 55 * Math.sin(ph1 + 4.0))

        var grad = ctx.createLinearGradient(p0.x, p0.y, p1.x, p1.y)
        grad.addColorStop(0, 'rgba(' + r0 + ',' + g0 + ',' + b0 + ',' + alpha + ')')
        grad.addColorStop(1, 'rgba(' + r1 + ',' + g1 + ',' + b1 + ',' + alpha + ')')

        ctx.beginPath()
        ctx.moveTo(p0.x, p0.y)
        ctx.lineTo(p1.x, p1.y)
        ctx.strokeStyle = grad
        ctx.lineWidth = lineW
        ctx.stroke()
      })

      // Single soft glow pass for front edges
      ctx.globalAlpha = 0.15
      ctx.shadowColor = 'rgba(160,100,255,0.5)'
      ctx.shadowBlur = 15
      sortedEdges.forEach(function (e) {
        var p0 = pts[e[0]], p1 = pts[e[1]]
        var avgZ = (p0.z + p1.z) / 2
        if (avgZ < -0.2) return
        var lineW = 1.0 + (avgZ + 1) * 0.8
        ctx.beginPath()
        ctx.moveTo(p0.x, p0.y)
        ctx.lineTo(p1.x, p1.y)
        ctx.strokeStyle = 'rgba(180,130,255,0.6)'
        ctx.lineWidth = lineW
        ctx.stroke()
      })
      ctx.globalAlpha = 1
      ctx.shadowBlur = 0

      _afs.push(requestAnimationFrame(draw))
    }
    draw()
  }

  /* ================================================================
   * TITLE — Letter spans + continuous mouse 3D tilt
   * ================================================================ */
  function initTitle() {
    var el = document.getElementById('shapix-title')
    if (!el) return
    var text = 'Shapix'
    var html = ''
    for (var i = 0; i < text.length; i++)
      html += '<span class="hero__letter" style="animation-delay:' + i * 0.15 + 's">' + text[i] + '</span>'
    el.innerHTML = html

    var logoEl = document.querySelector('.hero__logo')
    if (logoEl) {
      ;(function updateTilt() {
        var dx = (mx - 0.5) * 15
        var dy = -(my - 0.5) * 10
        logoEl.style.transform = 'rotateY(' + dx + 'deg) rotateX(' + (dy + 5) + 'deg)'
        _afs.push(requestAnimationFrame(updateTilt))
      })()
    }
  }

  function cleanup() {
    for (var i = 0; i < _afs.length; i++) cancelAnimationFrame(_afs[i])
    _afs = []
    var old = document.querySelector('#shapix-visual canvas')
    if (old) old.remove()
  }

  function init() {
    cleanup()
    start()
  }

  // Initial load
  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', init)
  else
    requestAnimationFrame(init)

  // SPA navigation: watch for #shapix-visual appearing without a canvas child
  // This works regardless of when MkDocs Material's JS loads
  var _observed = false
  new MutationObserver(function () {
    var el = document.getElementById('shapix-visual')
    if (el && !el.querySelector('canvas')) {
      if (!_observed) { _observed = true; return } // skip initial (handled above)
      init()
    } else if (!el) {
      _observed = true // navigated away from home
    }
  }).observe(document.body, { childList: true, subtree: true })
})()
