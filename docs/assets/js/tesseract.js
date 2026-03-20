/**
 * Shapix Visual Suite
 * 1. Background: WebGL2 spiraling 4D particle flow with light rays
 * 2. Logo: Animated rotating tesseract wireframe (canvas 2D)
 * 3. Title: 3D letters with synchronized wave motion
 */
;(function () {
  'use strict'

  /* ================================================================
   * LOGO — Animated rotating tesseract wireframe
   * ================================================================ */
  function initLogo() {
    var c = document.getElementById('shapix-logo')
    if (!c) return
    var ctx = c.getContext('2d')
    var S = 80, dpr = Math.min(window.devicePixelRatio || 1, 2)
    c.width = S * dpr; c.height = S * dpr
    c.style.width = S + 'px'; c.style.height = S + 'px'
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    // 4D hypercube: 16 vertices, 32 edges
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
      var x=v[0],y=v[1],z=v[2],w=v[3], c1,s1
      c1=Math.cos(a1);s1=Math.sin(a1);var nx=x*c1-w*s1;w=x*s1+w*c1;x=nx
      c1=Math.cos(a2);s1=Math.sin(a2);var ny=y*c1-z*s1;z=y*s1+z*c1;y=ny
      c1=Math.cos(a3);s1=Math.sin(a3);nx=x*c1-y*s1;y=x*s1+y*c1;x=nx
      return [x,y,z,w]
    }

    function proj(v) {
      var d=3.2, s=1/(d-v[3])
      var d2=3, s2=1/(d2-v[2]*s)
      return { x: v[0]*s*s2, y: v[1]*s*s2, depth: s*d }
    }

    function draw(t) {
      ctx.clearRect(0, 0, S, S)
      var cx=S/2, cy=S/2, sc=S*0.22
      var a1=t*0.5, a2=t*0.37, a3=t*0.23
      var pts = V.map(function(v) {
        var r = rot4(v, a1, a2, a3)
        var p = proj(r)
        return { x: cx+p.x*sc, y: cy+p.y*sc, d: p.depth }
      })

      // Sort edges by depth
      var sorted = E.slice().sort(function(a,b) {
        return (pts[a[0]].d+pts[a[1]].d) - (pts[b[0]].d+pts[b[1]].d)
      })

      sorted.forEach(function(e) {
        var p0=pts[e[0]], p1=pts[e[1]]
        var avg=(p0.d+p1.d)/2
        var alpha=Math.max(0.1, Math.min(0.9, avg*0.5))
        var g = ctx.createLinearGradient(p0.x, p0.y, p1.x, p1.y)
        g.addColorStop(0, 'rgba(124,77,255,'+alpha+')')
        g.addColorStop(0.5, 'rgba(179,136,255,'+alpha+')')
        g.addColorStop(1, 'rgba(234,128,252,'+alpha+')')
        ctx.beginPath(); ctx.moveTo(p0.x, p0.y); ctx.lineTo(p1.x, p1.y)
        ctx.strokeStyle = g
        ctx.lineWidth = 0.8 + avg * 0.8
        ctx.shadowColor = 'rgba(179,136,255,' + alpha*0.5 + ')'
        ctx.shadowBlur = 4
        ctx.stroke()
        ctx.shadowBlur = 0
      })

      // Vertex dots
      pts.forEach(function(p) {
        var r = 1 + p.d * 1.2
        var alpha = Math.max(0.2, Math.min(1, p.d * 0.6))
        ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI*2)
        ctx.fillStyle = 'rgba(234,128,252,' + alpha + ')'
        ctx.fill()
      })

      requestAnimationFrame(function() { draw(performance.now() * 0.001) })
    }
    draw(0)
  }

  /* ================================================================
   * TITLE — 3D animated letters with wave motion
   * ================================================================ */
  function initTitle() {
    var el = document.getElementById('shapix-title')
    if (!el) return

    var text = 'Shapix'
    var html = ''
    for (var i = 0; i < text.length; i++) {
      html += '<span class="hero__letter" style="animation-delay:' + (i * 0.12) + 's">' + text[i] + '</span>'
    }
    el.innerHTML = html
  }

  /* ================================================================
   * BACKGROUND — WebGL2 spiraling 4D flow field
   * ================================================================ */
  function initBg() {
    var el = document.getElementById('shapix-visual')
    if (!el) return

    if (!el.offsetWidth) {
      requestAnimationFrame(function () { setTimeout(initBg, 50) })
      return
    }

    var canvas = document.createElement('canvas')
    canvas.style.cssText = 'display:block;width:100%;border-radius:16px;'
    el.appendChild(canvas)

    var gl = canvas.getContext('webgl2', { antialias: false, alpha: true, premultipliedAlpha: false })
    if (!gl) {
      el.style.cssText += 'height:400px;background:linear-gradient(135deg,#7c4dff,#b388ff,#ea80fc);'
      return
    }

    var VERT = '#version 300 es\nin vec2 p;void main(){gl_Position=vec4(p,0,1);}'
    var FRAG = [
      '#version 300 es',
      'precision highp float;',
      'uniform float T;',
      'uniform vec2 R;',
      'uniform vec2 M;',
      'uniform float D;',
      'out vec4 O;',
      '',
      'vec3 h33(vec3 p){p=fract(p*vec3(.1031,.1030,.0973));p+=dot(p,p.yxz+33.33);return fract((p.xxy+p.yxx)*p.zyx);}',
      '',
      'float n3(vec3 p){',
      '  vec3 i=floor(p),f=fract(p);f=f*f*(3.-2.*f);',
      '  return mix(mix(mix(dot(h33(i),f),dot(h33(i+vec3(1,0,0)),f-vec3(1,0,0)),f.x),',
      '    mix(dot(h33(i+vec3(0,1,0)),f-vec3(0,1,0)),dot(h33(i+vec3(1,1,0)),f-vec3(1,1,0)),f.x),f.y),',
      '    mix(mix(dot(h33(i+vec3(0,0,1)),f-vec3(0,0,1)),dot(h33(i+vec3(1,0,1)),f-vec3(1,0,1)),f.x),',
      '    mix(dot(h33(i+vec3(0,1,1)),f-vec3(0,1,1)),dot(h33(i+vec3(1,1,1)),f-vec3(1,1,1)),f.x),f.y),f.z);',
      '}',
      '',
      'float fbm(vec3 p){',
      '  float v=0.,a=.5;',
      '  for(int i=0;i<5;i++){',
      '    float t=T*.12+float(i)*.5;',
      '    float c1=cos(t),s1=sin(t),c2=cos(t*.7),s2=sin(t*.7);',
      '    p.xz=mat2(c1,-s1,s1,c1)*p.xz;',
      '    p.yz=mat2(c2,-s2,s2,c2)*p.yz;',
      '    v+=a*n3(p);p=p*2.03+.5;a*=.48;',
      '  }',
      '  return v;',
      '}',
      '',
      'void main(){',
      '  vec2 uv=(gl_FragCoord.xy-.5*R)/min(R.x,R.y);',
      '  uv+=(M-.5)*.06;',
      '  float r=length(uv),ang=atan(uv.y,uv.x);',
      '  ang+=T*.15+r*3.;',
      '  vec2 sp=vec2(cos(ang),sin(ang))*r;',
      '',
      '  float n1=fbm(vec3(sp*1.5,T*.08));',
      '  float n2=fbm(vec3(uv*2.+n1*.4,T*.06+7.));',
      '  float n3a=fbm(vec3(sp*2.5-n2*.3,T*.1+14.));',
      '  vec2 w=uv+vec2(n1,n2)*.3;',
      '',
      '  // Light rays',
      '  float rays=0.;',
      '  for(int i=0;i<3;i++){',
      '    float fi=float(i),s=1.+fi*.4;',
      '    vec2 rv=w*s+vec2(fi*.13,-fi*.09);',
      '    float a2=atan(rv.y,rv.x),rl=length(rv);',
      '    float ray=pow(sin(a2*(6.+fi*2.)+T*(1.2+fi*.3)+rl*4.)*.5+.5,6.-fi);',
      '    rays+=ray*exp(-rl*(1.5+fi*.3))*(.4-fi*.1);',
      '  }',
      '',
      '  // Particles',
      '  float pts=0.;',
      '  for(int i=0;i<12;i++){',
      '    float fi=float(i);',
      '    vec2 pp=vec2(cos(T*.35+fi*2.094)*(1.+sin(T*.18+fi)*.5),sin(T*.27+fi*1.571)*(.7+cos(T*.13+fi)*.4));',
      '    pts+=.004/(dot(w-pp,w-pp)+.0008);',
      '  }',
      '  pts=min(pts,4.);',
      '',
      '  // Spiraling streaks',
      '  float streak=pow(sin(ang*8.+r*12.-T*2.5)*.5+.5,10.)*.5*exp(-r*1.5);',
      '  float flow=pow(max(n2*n3a*2.,0.),.7);',
      '',
      '  // Colors',
      '  vec3 purple=vec3(.486,.302,1.),lilac=vec3(.702,.533,1.),pink=vec3(.918,.502,.988),deep=vec3(.20,.08,.40),cyan=vec3(.45,.88,1.);',
      '  float grad=n1*.4+r*.3+sin(T*.15)*.15;',
      '  vec3 fc=mix(mix(purple,lilac,sin(grad*3.14)*.5+.5),mix(pink,cyan,cos(grad*2.+1.)*.5+.5),n2*.5+.5);',
      '',
      '  vec3 col=deep*exp(-r*1.2)*.2;',
      '  col+=fc*flow*.3;',
      '  col+=mix(purple,pink,sin(T*.25+r)*.5+.5)*rays;',
      '  col+=mix(lilac,cyan,n3a)*streak;',
      '  col+=mix(lilac,vec3(1.),pts*.15)*pts*.12;',
      '  col*=max(1.-pow(r*.65,2.8),0.);',
      '  col=clamp(col*(2.51*col+.03)/(col*(2.43*col+.59)+.14),0.,1.);',
      '',
      '  vec3 bg=mix(vec3(.98),vec3(.051,.067,.09),D);',
      '  float a3=smoothstep(0.,.08,max(max(col.r,col.g),col.b));',
      '  col=mix(bg,col,min(a3*1.5+.15,1.));',
      '  O=vec4(col,1.);',
      '}',
    ].join('\n')

    function mk(type, src) {
      var s = gl.createShader(type)
      gl.shaderSource(s, src)
      gl.compileShader(s)
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error('[shapix] shader:', gl.getShaderInfoLog(s))
        return null
      }
      return s
    }

    var vs = mk(gl.VERTEX_SHADER, VERT), fs = mk(gl.FRAGMENT_SHADER, FRAG)
    if (!vs || !fs) {
      canvas.remove()
      el.style.cssText += 'height:400px;background:linear-gradient(135deg,#7c4dff,#b388ff,#ea80fc);'
      return
    }

    var pg = gl.createProgram()
    gl.attachShader(pg, vs); gl.attachShader(pg, fs); gl.linkProgram(pg)
    if (!gl.getProgramParameter(pg, gl.LINK_STATUS)) {
      console.error('[shapix] link:', gl.getProgramInfoLog(pg))
      canvas.remove()
      el.style.cssText += 'height:400px;background:linear-gradient(135deg,#7c4dff,#b388ff,#ea80fc);'
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
      var w = el.offsetWidth || el.parentElement.offsetWidth
      var h = Math.min(520, Math.max(340, w * .5))
      var d = Math.min(window.devicePixelRatio || 1, 2)
      canvas.width = w * d; canvas.height = h * d
      canvas.style.height = h + 'px'
      gl.viewport(0, 0, canvas.width, canvas.height)
    }

    el.addEventListener('mousemove', function(e) {
      var r = el.getBoundingClientRect()
      tmx = (e.clientX-r.left)/r.width; tmy = 1-(e.clientY-r.top)/r.height
    })
    el.addEventListener('mouseleave', function() { tmx=.5; tmy=.5 })

    function frame(t) {
      mx+=(tmx-mx)*.04; my+=(tmy-my)*.04
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
   * INIT ALL
   * ================================================================ */
  function start() {
    initLogo()
    initTitle()
    initBg()
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start)
  } else {
    requestAnimationFrame(start)
  }
})()
