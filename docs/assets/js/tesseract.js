/**
 * Shapix Visual v4 — 3D morphing bubbles (ray-marched metaballs),
 * icosahedron logo, and 3D extruded title.
 * Two-pass WebGL2: scene FBO -> composite with subtle bloom.
 * Zero dependencies.
 */
;(function () {
  'use strict'

  var tmx = 0.5, tmy = 0.5, mx = 0.5, my = 0.5

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
      '    sin(t*(.06+fi*.012)+fi*3.1)*(1.4+.35*sin(t*.08+fi*.7))',
      '  );',
      '}',
      '',
      'float brad(int i,float t){',
      '  float fi=float(i);',
      '  return .35+.2*sin(t*(.15+fi*.04)+fi*2.5)+.1*sin(t*.3+fi*4.);',
      '}',
      '',
      '// Purple spectrum that evolves over time',
      'vec3 bcol(int i,float t){',
      '  float fi=float(i);',
      '  float phase=fi*.618+t*.03;',
      '  return vec3(',
      '    .4+.25*sin(phase*6.28),',
      '    .12+.12*sin(phase*6.28+4.2),',
      '    .65+.3*sin(phase*6.28+1.5)',
      '  );',
      '}',
      '',
      'float scene(vec3 p){',
      '  vec3 wp=warp(p,T);',
      '  float d=length(wp-bpos(0,T))-brad(0,T);',
      '  for(int i=1;i<NB;i++){',
      '    d=smin(d,length(wp-bpos(i,T))-brad(i,T),.5);',
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
      '  float ao=T*.04;',
      '  mat3 aRot=mat3(cos(ao),0.,sin(ao), 0.,1.,0., -sin(ao),0.,cos(ao));',
      '  mat3 cam=mRot*aRot;',
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
      '    vec3 ldir=normalize(vec3(.3,1.,-.5));',
      '',
      '    // Half-lambert diffuse',
      '    float diff=dot(n,ldir)*.5+.5;',
      '    diff=diff*diff;',
      '',
      '    // Fresnel rim',
      '    float fres=pow(1.-max(dot(n,-rd),0.),3.);',
      '',
      '    // Subsurface scatter approx',
      '    float sss=pow(max(dot(normalize(rd+ldir*.5),n),0.),4.)*.2;',
      '',
      '    // Purple-tinted rim glow',
      '    vec3 rimCol=vec3(.6,.3,.9)*fres;',
      '',
      '    col=base*(.18+diff*.55)+rimCol*.25+base*sss;',
      '    col*=exp(-t*.04);',
      '  }',
      '',
      '  // Ambient glow from nearest blob',
      '  float glow=exp(-minD*minD*2.);',
      '  col+=scol(minP)*glow*.06;',
      '',
      '  // Subtle purple atmosphere',
      '  col+=mix(vec3(.04,.015,.1),vec3(.03,.02,.08),uv.x*.5+.5)*D*.04;',
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
      'void main(){',
      '  vec2 tc=gl_FragCoord.xy/R;',
      '  vec2 uv=(gl_FragCoord.xy-.5*R)/min(R.x,R.y);',
      '  vec3 scene=texture(uScene,tc).rgb;',
      '',
      '  // Very subtle bloom',
      '  vec3 bloom=vec3(0);float tw=0.;',
      '  for(int i=0;i<8;i++){',
      '    float a=float(i)*.785398;',
      '    vec2 d=vec2(cos(a),sin(a));',
      '    bloom+=texture(uScene,tc+d*6./R).rgb;',
      '    bloom+=texture(uScene,tc+d*20./R).rgb*.4;',
      '    tw+=1.4;',
      '  }',
      '  bloom/=tw;',
      '  vec3 col=scene+bloom*.12;',
      '',
      '  // Reinhard tone mapping',
      '  col=col/(col+1.);',
      '',
      '  // Gentle vignette',
      '  col*=1.-dot(uv,uv)*.2;',
      '',
      '  // Minimal grain',
      '  float grain=fract(sin(dot(gl_FragCoord.xy+fract(T*100.),vec2(12.9898,78.233)))*43758.5453);',
      '  col+=(grain-.5)*.01;',
      '',
      '  // Background blend',
      '  vec3 bg=mix(vec3(.97,.97,.99),vec3(.04,.048,.08),D);',
      '  float alpha=max(max(col.r,col.g),col.b);',
      '  alpha=smoothstep(0.,.015,alpha);',
      '  float boost=mix(1.3,1.,D);',
      '  col=mix(bg,col*boost,min(alpha*2.,1.));',
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

    var aid = null

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
      var time = t * .001, dark = dk()

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

      aid = requestAnimationFrame(frame)
    }

    resize()
    window.addEventListener('resize', resize)
    aid = requestAnimationFrame(frame)

    document.addEventListener('visibilitychange', function () {
      if (document.hidden) {
        if (aid) { cancelAnimationFrame(aid); aid = null }
      } else {
        if (!aid) aid = requestAnimationFrame(frame)
      }
    })
  }

  /* ================================================================
   * LOGO — Large icosahedron with solid faces and glowing edges
   * ================================================================ */
  function initLogo() {
    var c = document.getElementById('shapix-logo')
    if (!c) return
    var ctx = c.getContext('2d')
    var S = 280, dpr = Math.min(window.devicePixelRatio || 1, 2)
    c.width = S * dpr; c.height = S * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    var phi = (1 + Math.sqrt(5)) / 2
    var raw = [
      [0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
      [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
      [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
    ]
    var V = raw.map(function (v) {
      var l = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])
      return [v[0] / l, v[1] / l, v[2] / l]
    })

    var E = [], adj = []
    for (var i = 0; i < 12; i++) adj.push([])
    for (var i = 0; i < 12; i++)
      for (var j = i + 1; j < 12; j++) {
        var dx = V[i][0] - V[j][0], dy = V[i][1] - V[j][1], dz = V[i][2] - V[j][2]
        if (dx * dx + dy * dy + dz * dz < 1.25) {
          E.push([i, j])
          adj[i].push(j)
          adj[j].push(i)
        }
      }

    var F = []
    for (var i = 0; i < 12; i++)
      for (var ji = 0; ji < adj[i].length; ji++) {
        var jj = adj[i][ji]
        if (jj <= i) continue
        for (var ki = ji + 1; ki < adj[i].length; ki++) {
          var kk = adj[i][ki]
          if (kk <= jj) continue
          if (adj[jj].indexOf(kk) !== -1) F.push([i, jj, kk])
        }
      }

    function rotY(v, a) {
      var co = Math.cos(a), si = Math.sin(a)
      return [v[0] * co + v[2] * si, v[1], -v[0] * si + v[2] * co]
    }
    function rotX(v, a) {
      var co = Math.cos(a), si = Math.sin(a)
      return [v[0], v[1] * co - v[2] * si, v[1] * si + v[2] * co]
    }
    function rotZ(v, a) {
      var co = Math.cos(a), si = Math.sin(a)
      return [v[0] * co - v[1] * si, v[0] * si + v[1] * co, v[2]]
    }

    function draw() {
      var t = performance.now() * 0.001
      ctx.clearRect(0, 0, S, S)
      var cx = S / 2, cy = S / 2, sc = S * 0.4

      var floatY = Math.sin(t * 0.7) * 8
      var tiltX = (my - 0.5) * 0.6
      var tiltY = (mx - 0.5) * 0.6

      var pts = V.map(function (v) {
        var r = rotY(v, t * 0.3)
        r = rotX(r, t * 0.2)
        r = rotZ(r, t * 0.12)
        r = rotX(r, tiltX)
        r = rotY(r, tiltY)
        var d = 3.2, s = 1 / (d - r[2])
        return { x: cx + r[0] * sc * s, y: cy + r[1] * sc * s + floatY, z: r[2], d: s * d }
      })

      // Faces (back to front)
      F.slice().sort(function (a, b) {
        return (pts[a[0]].z + pts[a[1]].z + pts[a[2]].z) - (pts[b[0]].z + pts[b[1]].z + pts[b[2]].z)
      }).forEach(function (f) {
        var p0 = pts[f[0]], p1 = pts[f[1]], p2 = pts[f[2]]
        var avgZ = (p0.z + p1.z + p2.z) / 3
        var alpha = 0.12 + (avgZ + 1) * 0.16
        alpha = Math.max(0.08, Math.min(0.42, alpha))
        var hue = ((avgZ + 1) * 0.25 + t * 0.03) % 1
        var r = Math.round(90 + hue * 100)
        var g = Math.round(50 + (1 - hue) * 60)
        var b = Math.round(200 + hue * 55)

        ctx.beginPath()
        ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y)
        ctx.closePath()
        ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')'
        ctx.fill()
      })

      // Edges (back to front)
      E.slice().sort(function (a, b) {
        return (pts[a[0]].z + pts[a[1]].z) - (pts[b[0]].z + pts[b[1]].z)
      }).forEach(function (e) {
        var p0 = pts[e[0]], p1 = pts[e[1]]
        var avgZ = (p0.z + p1.z) / 2
        var alpha = 0.3 + (avgZ + 1) * 0.3
        alpha = Math.max(0.2, Math.min(0.9, alpha))

        var g = ctx.createLinearGradient(p0.x, p0.y, p1.x, p1.y)
        g.addColorStop(0, 'rgba(124,77,255,' + alpha + ')')
        g.addColorStop(0.5, 'rgba(179,136,255,' + alpha + ')')
        g.addColorStop(1, 'rgba(210,150,255,' + alpha + ')')

        ctx.beginPath()
        ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y)
        ctx.strokeStyle = g
        ctx.lineWidth = 1.2 + (avgZ + 1) * 1.2
        ctx.shadowColor = 'rgba(160,120,255,' + alpha * 0.7 + ')'
        ctx.shadowBlur = 12
        ctx.stroke()
        ctx.shadowBlur = 0
      })

      // Vertices
      pts.forEach(function (p) {
        var alpha = 0.4 + (p.z + 1) * 0.3
        alpha = Math.max(0.3, Math.min(0.95, alpha))
        var size = 1.8 + (p.z + 1) * 1.2

        ctx.beginPath()
        ctx.arc(p.x, p.y, size, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(210,175,255,' + alpha + ')'
        ctx.shadowColor = 'rgba(160,120,255,' + alpha * 0.8 + ')'
        ctx.shadowBlur = 14
        ctx.fill()
        ctx.shadowBlur = 0
      })

      requestAnimationFrame(draw)
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
        requestAnimationFrame(updateTilt)
      })()
    }
  }

  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', start)
  else
    requestAnimationFrame(start)
})()
