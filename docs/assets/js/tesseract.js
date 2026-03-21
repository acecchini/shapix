/**
 * Shapix Visual v3 — Subdued 3D light rays with bloom post-processing,
 * icosahedron logo, and animated 3D title with mouse interaction.
 * Two-pass WebGL2 rendering: scene FBO -> composite with bloom/CA/vignette.
 * Zero dependencies.
 */
;(function () {
  'use strict'

  // Global mouse state (shared across all components)
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
   * BACKGROUND — 3D light rays with multi-pass post-processing
   * Stronger mouse interaction, more curves, reduced brightness
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

    // ── Scene shader: curves + particles + nebula ──
    var FRAG_SCENE = [
      '#version 300 es',
      'precision highp float;',
      'uniform float T;',
      'uniform vec2 R;',
      'uniform vec2 M;',
      'uniform float D;',
      'out vec4 O;',
      '',
      '#define NC 16',
      '#define NS 56',
      '#define PI 3.14159265',
      '#define NPART 30',
      '',
      'vec3 hash33(vec3 p){',
      '  p=fract(p*vec3(.1031,.103,.0973));',
      '  p+=dot(p,p.yxz+33.33);',
      '  return fract((p.xxy+p.yxx)*p.zyx);',
      '}',
      '',
      'float noise3(vec3 p){',
      '  vec3 i=floor(p),f=fract(p);',
      '  f=f*f*(3.-2.*f);',
      '  float a=dot(hash33(i),vec3(1))/3.,',
      '        b=dot(hash33(i+vec3(1,0,0)),vec3(1))/3.,',
      '        c=dot(hash33(i+vec3(0,1,0)),vec3(1))/3.,',
      '        d=dot(hash33(i+vec3(1,1,0)),vec3(1))/3.,',
      '        e=dot(hash33(i+vec3(0,0,1)),vec3(1))/3.,',
      '        f1=dot(hash33(i+vec3(1,0,1)),vec3(1))/3.,',
      '        g=dot(hash33(i+vec3(0,1,1)),vec3(1))/3.,',
      '        h=dot(hash33(i+vec3(1,1,1)),vec3(1))/3.;',
      '  return mix(mix(mix(a,b,f.x),mix(c,d,f.x),f.y),',
      '             mix(mix(e,f1,f.x),mix(g,h,f.x),f.y),f.z);',
      '}',
      '',
      'float fbm(vec3 p){return noise3(p)*.6+noise3(p*2.03)*.3+noise3(p*4.01)*.1;}',
      '',
      'vec3 hsl2rgb(float h,float s,float l){',
      '  vec3 rgb=clamp(abs(mod(h*6.+vec3(0,4,2),6.)-3.)-1.,0.,1.);',
      '  return l+s*(rgb-.5)*(1.-abs(2.*l-1.));',
      '}',
      '',
      'vec3 curve(int i,float s,float t){',
      '  float fi=float(i),phase=fi*1.618;',
      '  float fx=1.+mod(fi*3.,4.),fy=1.+mod(fi*2.+1.,5.),fz=.5+mod(fi*1.5,3.);',
      '  float speed=.7+mod(fi*.73,.8);',
      '  float drift=t*(.15+fi*.02)*speed;',
      '  float ax=2.8+sin(t*.1+phase)*.8;',
      '  float ay=2.0+cos(t*.13+phase*.7)*.7;',
      '  float az=2.5+sin(t*.08+phase*1.3)*.9;',
      '  float x=ax*sin(s*fx*PI+drift+phase);',
      '  float y=ay*sin(s*fy*PI+drift*1.3+phase*2.1);',
      '  float z=az*cos(s*fz*PI+drift*.7+phase*.5)-.2;',
      '  vec3 p=vec3(x,y,z);',
      '  float ns=s*5.+fi*2.+t*.15;',
      '  p.x+=sin(ns*1.3+t*.4)*cos(ns*.7+fi)*.25;',
      '  p.y+=cos(ns*1.1+t*.3)*sin(ns*.9+fi*1.5)*.20;',
      '  p.z+=sin(ns*.8+t*.2)*cos(ns*1.2+fi*.7)*.15;',
      '  return p;',
      '}',
      '',
      'vec2 proj(vec3 p,float cam){',
      '  float w=1./(cam-p.z);',
      '  return vec2(p.x*w,p.y*w);',
      '}',
      '',
      'float envelope(int i,float t){',
      '  float fi=float(i);',
      '  float cycle=sin(t*(.12+fi*.015)+fi*2.39)*.5+.5;',
      '  return smoothstep(.05,.4,cycle)*smoothstep(1.,.7,cycle);',
      '}',
      '',
      'void main(){',
      '  vec2 uv=(gl_FragCoord.xy-.5*R)/min(R.x,R.y);',
      '  vec2 m=(M-.5);',
      '',
      '  // Mouse lens warp',
      '  uv+=m*.12*length(uv);',
      '',
      '  // Strong camera rotation from mouse',
      '  float ry=m.x*1.0,rx=m.y*.8;',
      '  float cy2=cos(ry),sy2=sin(ry),cx2=cos(rx),sx2=sin(rx);',
      '  mat3 cam=mat3(cy2,sx2*sy2,-cx2*sy2, 0.,cx2,sx2, sy2,-sx2*cy2,cx2*cy2);',
      '',
      '  float zoom=5.5+sin(T*.15)*.15;',
      '',
      '  vec3 col=vec3(0);',
      '  vec3 c1=vec3(.486,.302,1),c2=vec3(.702,.533,1),c3=vec3(.918,.502,.988),',
      '       c4=vec3(.45,.88,1),c5=vec3(1,.6,.9),c6=vec3(.3,.95,.7);',
      '',
      '  for(int i=0;i<NC;i++){',
      '    float env=envelope(i,T);',
      '    if(env<.01)continue;',
      '    float fi=float(i),minD=1e9,bestS=0.,bestZ=0.;',
      '    vec2 bestDelta=vec2(0);',
      '    float wf=.7+mod(fi*.47,.6);',
      '',
      '    for(int j=0;j<NS;j++){',
      '      float s=float(j)/float(NS-1);',
      '      vec3 p3=cam*curve(i,s,T);',
      '      vec2 p2=proj(p3,zoom);',
      '      vec2 delta=uv-p2;',
      '      float d=length(delta);',
      '      if(d<minD){minD=d;bestS=s;bestZ=p3.z;bestDelta=delta;}',
      '    }',
      '',
      '    float dn=clamp((bestZ+2.5)/4.5,0.,1.);',
      '    float bt=mix(50.,3000.,dn*dn)*wf;',
      '    float core=exp(-minD*minD*bt*2.5);',
      '    float inner=exp(-minD*minD*bt*.3);',
      '    float halo=exp(-minD*minD*bt*.025)*.1;',
      '    float db=mix(.15,1.3,dn);',
      '',
      '    float pw=smoothstep(0.,.12,bestS)*smoothstep(1.,.88,bestS);',
      '    pw*=.5+.5*sin(bestS*6.28+T*.5+fi);',
      '    pw=max(pw,.12);',
      '',
      '    float nc=3.+mod(fi*1.3,4.);',
      '    float pulse=fract(bestS*nc-T*(.6+fi*.08));',
      '    float pi2=exp(-pulse*pulse*30.)*2.;',
      '    float nf=exp(-minD*minD*bt*5.)*pi2;',
      '',
      '    float ci=mod(fi*.37+T*.05,1.);',
      '    vec3 b1,b2;',
      '    if(ci<.17){b1=c1;b2=c2;}',
      '    else if(ci<.33){b1=c2;b2=c3;}',
      '    else if(ci<.5){b1=c3;b2=c4;}',
      '    else if(ci<.67){b1=c4;b2=c6;}',
      '    else if(ci<.83){b1=c6;b2=c5;}',
      '    else{b1=c5;b2=c1;}',
      '    float cp=bestS+sin(T*.2+fi*1.3)*.3;',
      '    vec3 rc=mix(b1,b2,sin(cp*PI)*.5+.5);',
      '    float hue=mod(fi*.083+bestS*.15+T*.02,1.);',
      '    rc=mix(rc,hsl2rgb(hue,.7,.55),.25);',
      '',
      '    vec3 ct=vec3(0);',
      '    ct+=vec3(.72,.6,.95)*core*1.1;',
      '    ct+=rc*inner*.9;',
      '    ct+=rc*.5*halo;',
      '    ct+=vec3(.7,.58,.92)*nf*.6;',
      '    ct*=env*db*pw;',
      '',
      '    float fl=exp(-bestDelta.y*bestDelta.y*bt*3.)*exp(-bestDelta.x*bestDelta.x*5.);',
      '    fl*=pi2*.3;',
      '    ct+=rc*.4*fl*env;',
      '',
      '    col+=ct;',
      '  }',
      '',
      '  for(int i=0;i<NPART;i++){',
      '    float fi=float(i);',
      '    vec3 seed=vec3(fi*1.73,fi*2.31,fi*.97);',
      '    vec3 pos=hash33(seed)*5.-2.5;',
      '    pos.y+=sin(T*.4+pos.x*2.)*.4;',
      '    pos.x+=cos(T*.3+pos.z*1.5)*.3;',
      '    pos.z+=sin(T*.2+fi*.5)*.5-.3;',
      '    pos=cam*pos;',
      '    vec2 pp=proj(pos,zoom);',
      '    float d=length(uv-pp);',
      '    float dp=clamp((pos.z+2.)/4.,0.,1.);',
      '    float sparkle=sin(T*(3.+fi*.5)+fi*10.)*.5+.5;',
      '    col+=hsl2rgb(mod(fi*.05+T*.01,1.),.6,.55)*exp(-d*d*20000.)*dp*sparkle*.3;',
      '  }',
      '',
      '  vec3 np=vec3(uv*2.,T*.025);',
      '  float neb=fbm(np)*.5+fbm(np*1.5+30.)*.3;',
      '  col+=mix(vec3(.12,.04,.25),vec3(.04,.12,.25),neb)*neb*.09*D;',
      '',
      '  col+=vec3(.22,.1,.38)*exp(-length(uv)*1.2)*.04;',
      '',
      '  O=vec4(col,1);',
      '}'
    ].join('\n')

    // ── Composite shader: bloom + CA + mouse warp + vignette + grain ──
    var FRAG_COMP = [
      '#version 300 es',
      'precision highp float;',
      'uniform sampler2D uScene;',
      'uniform vec2 R;',
      'uniform float T;',
      'uniform float D;',
      'uniform vec2 M;',
      'out vec4 O;',
      '',
      'void main(){',
      '  vec2 tc=gl_FragCoord.xy/R;',
      '  vec2 uv=(gl_FragCoord.xy-.5*R)/min(R.x,R.y);',
      '  vec2 m=(M-.5);',
      '',
      '  // Subtle mouse lens warp',
      '  tc+=m*.008*length(uv);',
      '',
      '  vec3 scene=texture(uScene,tc).rgb;',
      '',
      '  vec3 bloom=vec3(0);float tw=0.;',
      '  for(int i=0;i<8;i++){',
      '    float a=float(i)*.785398;',
      '    vec2 d=vec2(cos(a),sin(a));',
      '    bloom+=texture(uScene,tc+d*5./R).rgb;',
      '    bloom+=texture(uScene,tc+d*18./R).rgb*.5;',
      '    bloom+=texture(uScene,tc+d*45./R).rgb*.25;',
      '    tw+=1.75;',
      '  }',
      '  bloom/=tw;',
      '  vec3 col=scene+bloom*.4;',
      '',
      '  float ca=length(uv)*.006;',
      '  vec2 cd=uv*ca;',
      '  col.r=mix(col.r,texture(uScene,tc+cd).r+bloom.r*.4,.5);',
      '  col.b=mix(col.b,texture(uScene,tc-cd).b+bloom.b*.4,.5);',
      '',
      '  col=col*(2.51*col+.03)/(col*(2.43*col+.59)+.14);',
      '  col=clamp(col,0.,1.);',
      '',
      '  col*=1.-dot(uv,uv)*.35;',
      '',
      '  float grain=fract(sin(dot(gl_FragCoord.xy+fract(T*100.),vec2(12.9898,78.233)))*43758.5453);',
      '  col+=(grain-.5)*.015;',
      '',
      '  vec3 bg=mix(vec3(.97,.97,.99),vec3(.035,.042,.07),D);',
      '  float alpha=max(max(col.r,col.g),col.b);',
      '  alpha=smoothstep(0.,.04,alpha);',
      '  col=mix(bg,col+bg*(1.-alpha),min(alpha*1.5,1.));',
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

    // Fullscreen quad
    var buf = gl.createBuffer()
    gl.bindBuffer(gl.ARRAY_BUFFER, buf)
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW)
    gl.enableVertexAttribArray(0)
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0)

    // Scene uniforms
    var uST = gl.getUniformLocation(sceneProg, 'T')
    var uSR = gl.getUniformLocation(sceneProg, 'R')
    var uSM = gl.getUniformLocation(sceneProg, 'M')
    var uSD = gl.getUniformLocation(sceneProg, 'D')
    // Composite uniforms
    var uCS = gl.getUniformLocation(compProg, 'uScene')
    var uCR = gl.getUniformLocation(compProg, 'R')
    var uCT = gl.getUniformLocation(compProg, 'T')
    var uCD = gl.getUniformLocation(compProg, 'D')
    var uCM = gl.getUniformLocation(compProg, 'M')

    // ── FBO for scene render ──
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

      // Pass 1: Scene -> FBO
      gl.bindFramebuffer(gl.FRAMEBUFFER, fboObj.fbo)
      gl.viewport(0, 0, fboObj.w, fboObj.h)
      gl.useProgram(sceneProg)
      gl.uniform1f(uST, time)
      gl.uniform2f(uSR, fboObj.w, fboObj.h)
      gl.uniform2f(uSM, mx, my)
      gl.uniform1f(uSD, dark)
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)

      // Pass 2: Composite -> screen
      gl.bindFramebuffer(gl.FRAMEBUFFER, null)
      gl.viewport(0, 0, canvas.width, canvas.height)
      gl.useProgram(compProg)
      gl.activeTexture(gl.TEXTURE0)
      gl.bindTexture(gl.TEXTURE_2D, fboObj.tex)
      gl.uniform1i(uCS, 0)
      gl.uniform2f(uCR, canvas.width, canvas.height)
      gl.uniform1f(uCT, time)
      gl.uniform1f(uCD, dark)
      gl.uniform2f(uCM, mx, my)
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
   * LOGO — Icosahedron with glowing edges and semi-transparent faces
   * ================================================================ */
  function initLogo() {
    var c = document.getElementById('shapix-logo')
    if (!c) return
    var ctx = c.getContext('2d')
    var S = 160, dpr = Math.min(window.devicePixelRatio || 1, 2)
    c.width = S * dpr; c.height = S * dpr
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    // Icosahedron geometry
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

    // Build edges
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

    // Build triangular faces
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
      var cx = S / 2, cy = S / 2, sc = S * 0.33

      // Floating bob animation
      var floatY = Math.sin(t * 0.8) * 5

      // Mouse tilt
      var tiltX = (my - 0.5) * 0.5
      var tiltY = (mx - 0.5) * 0.5

      var pts = V.map(function (v) {
        var r = rotY(v, t * 0.35)
        r = rotX(r, t * 0.22)
        r = rotZ(r, t * 0.13)
        r = rotX(r, tiltX)
        r = rotY(r, tiltY)
        var d = 2.8, s = 1 / (d - r[2])
        return { x: cx + r[0] * sc * s, y: cy + r[1] * sc * s + floatY, z: r[2], d: s * d }
      })

      // Draw faces (back to front by z)
      F.slice().sort(function (a, b) {
        return (pts[a[0]].z + pts[a[1]].z + pts[a[2]].z) - (pts[b[0]].z + pts[b[1]].z + pts[b[2]].z)
      }).forEach(function (f) {
        var p0 = pts[f[0]], p1 = pts[f[1]], p2 = pts[f[2]]
        var avgZ = (p0.z + p1.z + p2.z) / 3
        var alpha = 0.06 + (avgZ + 1) * 0.1
        alpha = Math.max(0.04, Math.min(0.22, alpha))
        var hue = ((avgZ + 1) * 0.25 + t * 0.04) % 1
        var r = Math.round(95 + hue * 90)
        var g = Math.round(55 + (1 - hue) * 55)
        var b = Math.round(210 + hue * 45)

        ctx.beginPath()
        ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y)
        ctx.closePath()
        ctx.fillStyle = 'rgba(' + r + ',' + g + ',' + b + ',' + alpha + ')'
        ctx.fill()
      })

      // Draw edges (back to front)
      E.slice().sort(function (a, b) {
        return (pts[a[0]].z + pts[a[1]].z) - (pts[b[0]].z + pts[b[1]].z)
      }).forEach(function (e) {
        var p0 = pts[e[0]], p1 = pts[e[1]]
        var avgZ = (p0.z + p1.z) / 2
        var alpha = 0.15 + (avgZ + 1) * 0.3
        alpha = Math.max(0.1, Math.min(0.75, alpha))

        var g = ctx.createLinearGradient(p0.x, p0.y, p1.x, p1.y)
        g.addColorStop(0, 'rgba(124,77,255,' + alpha + ')')
        g.addColorStop(0.5, 'rgba(179,136,255,' + alpha + ')')
        g.addColorStop(1, 'rgba(210,150,255,' + alpha + ')')

        ctx.beginPath()
        ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y)
        ctx.strokeStyle = g
        ctx.lineWidth = 0.8 + (avgZ + 1) * 0.9
        ctx.shadowColor = 'rgba(160,120,255,' + alpha * 0.7 + ')'
        ctx.shadowBlur = 8
        ctx.stroke()
        ctx.shadowBlur = 0
      })

      // Draw vertices with glow
      pts.forEach(function (p) {
        var alpha = 0.2 + (p.z + 1) * 0.35
        alpha = Math.max(0.15, Math.min(0.85, alpha))
        var size = 1.2 + (p.z + 1) * 0.9

        ctx.beginPath()
        ctx.arc(p.x, p.y, size, 0, Math.PI * 2)
        ctx.fillStyle = 'rgba(210,175,255,' + alpha + ')'
        ctx.shadowColor = 'rgba(160,120,255,' + alpha * 0.8 + ')'
        ctx.shadowBlur = 10
        ctx.fill()
        ctx.shadowBlur = 0
      })

      requestAnimationFrame(draw)
    }
    draw()
  }

  /* ================================================================
   * TITLE — Animated letters + continuous mouse 3D tilt
   * ================================================================ */
  function initTitle() {
    var el = document.getElementById('shapix-title')
    if (!el) return
    var text = 'Shapix'
    var html = ''
    for (var i = 0; i < text.length; i++)
      html += '<span class="hero__letter" style="animation-delay:' + i * 0.12 + 's">' + text[i] + '</span>'
    el.innerHTML = html

    // Continuous mouse 3D tilt on the logo+title container
    var logoEl = document.querySelector('.hero__logo')
    if (logoEl) {
      ;(function updateTilt() {
        var dx = (mx - 0.5) * 15
        var dy = -(my - 0.5) * 10
        logoEl.style.transform = 'rotateY(' + dx + 'deg) rotateX(' + dy + 'deg)'
        requestAnimationFrame(updateTilt)
      })()
    }
  }

  if (document.readyState === 'loading')
    document.addEventListener('DOMContentLoaded', start)
  else
    requestAnimationFrame(start)
})()
