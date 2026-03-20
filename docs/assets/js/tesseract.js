/**
 * Shapix 4D Visual — Spiraling particle flow with volumetric light rays.
 * Pure WebGL2 fragment shader, zero dependencies.
 */
;(function () {
  'use strict'

  var el = document.getElementById('shapix-visual')
  if (!el) return

  var canvas = document.createElement('canvas')
  canvas.style.cssText = 'display:block;width:100%;border-radius:16px;'
  el.appendChild(canvas)

  var gl = canvas.getContext('webgl2', { antialias: false, alpha: true, premultipliedAlpha: false })
  if (!gl) {
    el.style.cssText = 'height:400px;border-radius:16px;background:linear-gradient(135deg,#7c4dff,#b388ff,#ea80fc);'
    return
  }

  /* ── Shaders ─────────────────────────────────────────────────────── */
  var VERT = '#version 300 es\nin vec2 p;void main(){gl_Position=vec4(p,0,1);}'

  var FRAG = `#version 300 es
precision highp float;
uniform float T;
uniform vec2 R;
uniform vec2 M;
uniform float D;
out vec4 O;

// --- Hash & noise ---
vec3 hash33(vec3 p){
  p=fract(p*vec3(.1031,.1030,.0973));
  p+=dot(p,p.yxz+33.33);
  return fract((p.xxy+p.yxx)*p.zyx);
}

float noise(vec3 p){
  vec3 i=floor(p),f=fract(p);
  f=f*f*(3.-2.*f);
  float a=dot(hash33(i),f-vec3(0)),
        b=dot(hash33(i+vec3(1,0,0)),f-vec3(1,0,0)),
        c=dot(hash33(i+vec3(0,1,0)),f-vec3(0,1,0)),
        d=dot(hash33(i+vec3(1,1,0)),f-vec3(1,1,0)),
        e=dot(hash33(i+vec3(0,0,1)),f-vec3(0,0,1)),
        g=dot(hash33(i+vec3(1,0,1)),f-vec3(1,0,1)),
        h=dot(hash33(i+vec3(0,1,1)),f-vec3(0,1,1)),
        j=dot(hash33(i+vec3(1,1,1)),f-vec3(1,1,1));
  return mix(mix(mix(a,b,f.x),mix(c,d,f.x),f.y),
             mix(mix(e,g,f.x),mix(h,j,f.x),f.y),f.z);
}

// --- FBM with 4D rotation per octave ---
float fbm(vec3 p){
  float v=0.,a=.5;
  for(int i=0;i<6;i++){
    float t=T*.12+float(i)*.5;
    float c1=cos(t),s1=sin(t),c2=cos(t*.7),s2=sin(t*.7);
    p.xz=mat2(c1,-s1,s1,c1)*p.xz;
    p.yz=mat2(c2,-s2,s2,c2)*p.yz;
    v+=a*noise(p);
    p=p*2.03+.5;
    a*=.48;
  }
  return v;
}

void main(){
  vec2 uv=(gl_FragCoord.xy-.5*R)/min(R.x,R.y);
  vec2 m=(M-.5)*2.;
  uv+=m*.06;

  // --- Spiraling warp ---
  float r=length(uv),ang=atan(uv.y,uv.x);
  ang+=T*.15+r*3.;
  vec2 sp=vec2(cos(ang),sin(ang))*r;

  // --- Layered noise ---
  float n1=fbm(vec3(sp*1.5,T*.08));
  float n2=fbm(vec3(uv*2.+n1*.4,T*.06+7.));
  float n3=fbm(vec3(sp*2.5-n2*.3,T*.1+14.));

  // --- Warped coords ---
  vec2 w=uv+vec2(n1,n2)*.3;

  // --- Light rays ---
  float rays=0.;
  for(int i=0;i<3;i++){
    float fi=float(i);
    float s=1.+fi*.4;
    vec2 rv=w*s+vec2(fi*.13,-fi*.09);
    float a2=atan(rv.y,rv.x);
    float rl=length(rv);
    float ray=sin(a2*(6.+fi*2.)+T*(1.2+fi*.3)+rl*4.)*.5+.5;
    ray=pow(ray,6.-fi);
    ray*=exp(-rl*(1.5+fi*.3));
    rays+=ray*(.4-fi*.1);
  }

  // --- Particle hotspots ---
  float pts=0.;
  for(int i=0;i<16;i++){
    float fi=float(i);
    float a1=T*.35+fi*2.094;
    float a2=T*.27+fi*1.571;
    float r1=1.+sin(T*.18+fi)*.5;
    float r2=.7+cos(T*.13+fi)*.4;
    vec2 pp=vec2(cos(a1)*r1,sin(a2)*r2);
    float d2=dot(w-pp,w-pp);
    pts+=.004/(d2+.0008);
  }
  pts=min(pts,4.);

  // --- Spiraling streaks ---
  float streak=sin(ang*8.+r*12.-T*2.5)*.5+.5;
  streak=pow(streak,10.)*.5*exp(-r*1.5);

  // --- Flow intensity ---
  float flow=pow(max(n2*n3*2.,0.),.7);

  // --- Color palette ---
  vec3 purple=vec3(.486,.302,1.);
  vec3 lilac=vec3(.702,.533,1.);
  vec3 pink=vec3(.918,.502,.988);
  vec3 deep=vec3(.20,.08,.40);
  vec3 cyan=vec3(.45,.88,1.);

  float grad=n1*.4+r*.3+sin(T*.15)*.15;
  vec3 c1=mix(purple,lilac,sin(grad*3.14)*.5+.5);
  vec3 c2=mix(pink,cyan,cos(grad*2.+1.)*.5+.5);
  vec3 flowCol=mix(c1,c2,n2*.5+.5);

  // --- Compose ---
  vec3 col=deep*exp(-r*1.2)*.2;      // center glow
  col+=flowCol*flow*.3;               // flow field
  vec3 rayCol=mix(purple,pink,sin(T*.25+r)*.5+.5);
  col+=rayCol*rays;                   // light rays
  col+=mix(lilac,cyan,n3)*streak;     // spiraling streaks
  col+=mix(lilac,vec3(1.),pts*.15)*pts*.12; // particles

  // --- Vignette ---
  col*=max(1.-pow(r*.65,2.8),0.);

  // --- Tone map (ACES) ---
  col=clamp(col*(2.51*col+.03)/(col*(2.43*col+.59)+.14),0.,1.);

  // --- Background blend ---
  vec3 bg=mix(vec3(.98),vec3(.051,.067,.09),D);
  float a3=smoothstep(0.,.08,max(max(col.r,col.g),col.b));
  col=mix(bg,col,min(a3*1.5+.15,1.));

  O=vec4(col,1.);
}`

  /* ── Compile & link ──────────────────────────────────────────────── */
  function mkShader(type, src) {
    var s = gl.createShader(type)
    gl.shaderSource(s, src)
    gl.compileShader(s)
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('[shapix] shader error:', gl.getShaderInfoLog(s))
      return null
    }
    return s
  }

  var vs = mkShader(gl.VERTEX_SHADER, VERT)
  var fs = mkShader(gl.FRAGMENT_SHADER, FRAG)
  if (!vs || !fs) {
    console.error('[shapix] shader compilation failed')
    el.style.cssText = 'height:400px;border-radius:16px;background:linear-gradient(135deg,#7c4dff,#b388ff,#ea80fc);'
    return
  }

  var pg = gl.createProgram()
  gl.attachShader(pg, vs)
  gl.attachShader(pg, fs)
  gl.linkProgram(pg)
  if (!gl.getProgramParameter(pg, gl.LINK_STATUS)) {
    console.error('[shapix] link error:', gl.getProgramInfoLog(pg))
    el.style.cssText = 'height:400px;border-radius:16px;background:linear-gradient(135deg,#7c4dff,#b388ff,#ea80fc);'
    return
  }
  gl.useProgram(pg)

  /* ── Fullscreen quad ─────────────────────────────────────────────── */
  var buf = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, buf)
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1,1,-1,-1,1,1,1]), gl.STATIC_DRAW)
  var aLoc = gl.getAttribLocation(pg, 'p')
  gl.enableVertexAttribArray(aLoc)
  gl.vertexAttribPointer(aLoc, 2, gl.FLOAT, false, 0, 0)

  /* ── Uniforms ────────────────────────────────────────────────────── */
  var uT = gl.getUniformLocation(pg, 'T')
  var uR = gl.getUniformLocation(pg, 'R')
  var uM = gl.getUniformLocation(pg, 'M')
  var uD = gl.getUniformLocation(pg, 'D')

  /* ── State ───────────────────────────────────────────────────────── */
  var mx = .5, my = .5, tmx = .5, tmy = .5, aid = null

  function dark() {
    var e = document.querySelector('[data-md-color-scheme]')
    return e && e.getAttribute('data-md-color-scheme') === 'slate' ? 1 : 0
  }

  function resize() {
    var w = el.offsetWidth || el.parentElement.offsetWidth
    var h = Math.min(520, Math.max(340, w * .5))
    var d = Math.min(window.devicePixelRatio || 1, 2)
    canvas.width = w * d
    canvas.height = h * d
    canvas.style.height = h + 'px'
    gl.viewport(0, 0, canvas.width, canvas.height)
  }

  el.addEventListener('mousemove', function (e) {
    var r = el.getBoundingClientRect()
    tmx = (e.clientX - r.left) / r.width
    tmy = 1 - (e.clientY - r.top) / r.height
  })
  el.addEventListener('mouseleave', function () { tmx = .5; tmy = .5 })

  function frame(t) {
    mx += (tmx - mx) * .04
    my += (tmy - my) * .04
    gl.uniform1f(uT, t * .001)
    gl.uniform2f(uR, canvas.width, canvas.height)
    gl.uniform2f(uM, mx, my)
    gl.uniform1f(uD, dark())
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)
    aid = requestAnimationFrame(frame)
  }

  resize()
  window.addEventListener('resize', resize)
  aid = requestAnimationFrame(frame)

  if (typeof IntersectionObserver !== 'undefined') {
    new IntersectionObserver(function (e) {
      if (e[0].isIntersecting) { if (!aid) aid = requestAnimationFrame(frame) }
      else { if (aid) { cancelAnimationFrame(aid); aid = null } }
    }).observe(el)
  }
})()
