/**
 * Shapix 4D Visual — Spiraling particle flow field with volumetric light rays.
 *
 * A fullscreen WebGL2 fragment shader that renders a rotating 4D noise field
 * with spiraling particles, light rays, and dynamic purple-pink gradients.
 * Represents the multi-dimensional arrays that shapix validates.
 */
;(function () {
  'use strict'

  var container = document.getElementById('shapix-tesseract')
  if (!container) return

  // ── WebGL2 setup ──────────────────────────────────────────────────
  var canvas = document.createElement('canvas')
  canvas.style.display = 'block'
  canvas.style.width = '100%'
  canvas.style.cursor = 'crosshair'
  container.appendChild(canvas)

  var gl = canvas.getContext('webgl2', {
    antialias: true,
    alpha: true,
    premultipliedAlpha: false,
    powerPreference: 'high-performance',
  })

  if (!gl) {
    // Fallback: just show a gradient background
    container.style.background = 'linear-gradient(135deg, #7c4dff 0%, #b388ff 50%, #ea80fc 100%)'
    container.style.height = '400px'
    container.style.borderRadius = '16px'
    return
  }

  // ── Shaders ───────────────────────────────────────────────────────
  var VERT = '#version 300 es\nin vec2 a;void main(){gl_Position=vec4(a,0,1);}'

  var FRAG = [
    '#version 300 es',
    'precision highp float;',
    'uniform float u_time;',
    'uniform vec2 u_res;',
    'uniform vec2 u_mouse;',
    'uniform float u_dark;',
    'out vec4 O;',
    '',
    '// Simplex-style 4D noise helpers',
    'vec4 mod289(vec4 x){return x-floor(x*(1./289.))*289.;}',
    'float mod289(float x){return x-floor(x*(1./289.))*289.;}',
    'vec4 perm(vec4 x){return mod289(((x*34.)+10.)*x);}',
    '',
    'float noise4D(vec4 v){',
    '  const vec4 C=vec4(.138196601,.276393202,.414589803,-.447213595);',
    '  vec4 i=floor(v+dot(v,vec4(.309016994)));',
    '  vec4 x0=v-i+dot(i,C.xxxx);',
    '  vec4 i0;',
    '  vec3 isX=step(x0.yzw,x0.xxx);',
    '  vec3 isYZ=step(x0.zww,x0.yyz);',
    '  i0.x=isX.x+isX.y+isX.z;',
    '  i0.yzw=1.-isX;',
    '  i0.y+=isYZ.x+isYZ.y;',
    '  i0.zw+=1.-isYZ.xy;',
    '  i0.z+=isYZ.z;',
    '  i0.w+=1.-isYZ.z;',
    '  vec4 i3=clamp(i0,0.,1.);',
    '  vec4 i2=clamp(i0-1.,0.,1.);',
    '  vec4 i1=clamp(i0-2.,0.,1.);',
    '  vec4 x1=x0-i1+C.xxxx;',
    '  vec4 x2=x0-i2+2.*C.xxxx;',
    '  vec4 x3=x0-i3+3.*C.xxxx;',
    '  vec4 x4=x0-4.*C.xxxx+1.;',
    '  i=mod289(i);',
    '  float j0=perm(perm(perm(perm(i.w)+i.z)+i.y)+i.x).x;',
    '  vec4 j1=perm(perm(perm(perm(',
    '    i.w+vec4(i1.w,i2.w,i3.w,1.))',
    '    +i.z+vec4(i1.z,i2.z,i3.z,1.))',
    '    +i.y+vec4(i1.y,i2.y,i3.y,1.))',
    '    +i.x+vec4(i1.x,i2.x,i3.x,1.));',
    '  vec4 ip=vec4(1./294.,1./49.,2./7.,0.);',
    '  vec4 p0=ip.zzzz;',
    '  p0.x=fract(j0*ip.x)*7.-1.;',
    '  p0.y=fract(floor(j0*ip.x)*ip.x)*7.-1.;',
    '  p0.z=fract(floor(j0*ip.y)*ip.x)*7.-1.;',
    '  p0.w=1.5-abs(p0.x)-abs(p0.y)-abs(p0.z);',
    '  vec4 s0=vec4(lessThan(p0,vec4(0.)))*2.-1.;',
    '  p0.xyz+=s0.xyz*floor(-p0.www);',
    '  float d0=dot(x0,p0.xyz*vec3(p0.w>0.?1.:-1.))+.2;',
    '  // Simplified: use hash-based noise',
    '  return d0;',
    '}',
    '',
    '// Better hash-based 3D noise for reliable results',
    'float hash(vec3 p){',
    '  p=fract(p*vec3(.1031,.1030,.0973));',
    '  p+=dot(p,p.yxz+33.33);',
    '  return fract((p.x+p.y)*p.z);',
    '}',
    '',
    'float noise3D(vec3 p){',
    '  vec3 i=floor(p);vec3 f=fract(p);',
    '  f=f*f*(3.-2.*f);',
    '  return mix(mix(mix(hash(i),hash(i+vec3(1,0,0)),f.x),',
    '    mix(hash(i+vec3(0,1,0)),hash(i+vec3(1,1,0)),f.x),f.y),',
    '    mix(mix(hash(i+vec3(0,0,1)),hash(i+vec3(1,0,1)),f.x),',
    '    mix(hash(i+vec3(0,1,1)),hash(i+vec3(1,1,1)),f.x),f.y),f.z);',
    '}',
    '',
    '// Fractal Brownian Motion with 4D rotation',
    'float fbm(vec3 p,float t){',
    '  float v=0.,a=.5;',
    '  // Apply 4D spiral rotation to sampling coordinates',
    '  float s4=sin(t*.15),c4=cos(t*.15);',
    '  float s5=sin(t*.11),c5=cos(t*.11);',
    '  for(int i=0;i<6;i++){',
    '    // Rotate in XZ and YZ planes (simulating 4D rotation)',
    '    float px=p.x*c4-p.z*s4;',
    '    float pz=p.x*s4+p.z*c4;',
    '    float py=p.y*c5-pz*s5;',
    '    float pz2=p.y*s5+pz*c5;',
    '    v+=a*noise3D(vec3(px,py,pz2));',
    '    p=p*2.02+vec3(.3);',
    '    a*=.49;',
    '    // Rotate rotation angles for each octave',
    '    s4=sin(t*.15+float(i)*.7);c4=cos(t*.15+float(i)*.7);',
    '    s5=sin(t*.11+float(i)*.5);c5=cos(t*.11+float(i)*.5);',
    '  }',
    '  return v;',
    '}',
    '',
    '// Spiral flow field',
    'vec2 spiral(vec2 p,float t){',
    '  float r=length(p);',
    '  float a=atan(p.y,p.x);',
    '  a+=t*.3+r*2.;',
    '  return vec2(cos(a),sin(a))*r;',
    '}',
    '',
    '// Light ray function',
    'float rays(vec2 uv,float t){',
    '  float a=atan(uv.y,uv.x);',
    '  float r=length(uv);',
    '  float ray=sin(a*8.+t*1.5+r*4.)*.5+.5;',
    '  ray=pow(ray,4.);',
    '  ray*=exp(-r*1.8);',
    '  return ray;',
    '}',
    '',
    '// Particle hotspots',
    'float particles(vec3 p,float t){',
    '  float v=0.;',
    '  for(int i=0;i<12;i++){',
    '    float fi=float(i);',
    '    // Spiral orbit in 4D projected to 3D',
    '    float a1=t*.4+fi*2.094;',
    '    float a2=t*.3+fi*1.571;',
    '    float r1=1.2+sin(t*.2+fi)*.4;',
    '    float r2=.8+cos(t*.15+fi)*.3;',
    '    vec3 pp=vec3(cos(a1)*r1,sin(a2)*r2,sin(a1)*cos(a2)*.6);',
    '    float d=length(p-pp);',
    '    v+=.008/(d*d+.001);',
    '  }',
    '  return v;',
    '}',
    '',
    'void main(){',
    '  vec2 uv=(gl_FragCoord.xy-.5*u_res)/min(u_res.x,u_res.y);',
    '  float t=u_time;',
    '',
    '  // Mouse influence',
    '  vec2 m=(u_mouse-.5)*2.;',
    '  uv+=m*.08;',
    '',
    '  // Spiraling UV for flow field',
    '  vec2 suv=spiral(uv,t);',
    '',
    '  // 4D noise field (using time + spiral as extra dimensions)',
    '  float n1=fbm(vec3(suv*1.5,t*.2),t);',
    '  float n2=fbm(vec3(uv*2.+n1*.5,t*.15+5.),t+10.);',
    '  float n3=fbm(vec3(suv*3.-n2*.3,t*.1+10.),t+20.);',
    '',
    '  // Warp coordinates with noise for organic flow',
    '  vec2 wuv=uv+vec2(n1,n2)*.25;',
    '',
    '  // Volumetric light rays from center',
    '  float r=rays(wuv,t);',
    '  float r2=rays(wuv*1.3+vec2(.1,-.1),t*1.3+1.);',
    '  float r3=rays(wuv*.7-vec2(.05,.08),t*.7+2.5);',
    '',
    '  // Combine light rays',
    '  float light=(r+r2*.6+r3*.4)*.35;',
    '',
    '  // Particle hotspots in 3D',
    '  float pts=particles(vec3(wuv*1.8,n1*.5),t);',
    '  pts=min(pts,3.);',
    '',
    '  // Flow field intensity',
    '  float flow=n2*n3*2.;',
    '  flow=pow(max(flow,0.),.8);',
    '',
    '  // Spiraling streaks',
    '  float streak=sin(atan(suv.y,suv.x)*6.+length(suv)*8.-t*2.)*.5+.5;',
    '  streak=pow(streak,8.)*.4;',
    '  streak*=exp(-length(uv)*1.2);',
    '',
    '  // ── Color palette ────────────────────────────────────────',
    '  vec3 purple=vec3(.486,.302,1.);',    // #7c4dff
    '  vec3 lilac=vec3(.702,.533,1.);',     // #b388ff
    '  vec3 pink=vec3(.918,.502,.988);',    // #ea80fc
    '  vec3 deep=vec3(.259,.102,.502);',    // deep violet
    '  vec3 cyan=vec3(.4,.85,1.);',         // accent
    '',
    '  // Dynamic gradient based on position + time + noise',
    '  float grad=n1*.5+length(uv)*.3+sin(t*.2)*.2;',
    '  vec3 col1=mix(purple,lilac,sin(grad*3.14)*.5+.5);',
    '  vec3 col2=mix(pink,cyan,cos(grad*2.5+1.)*.5+.5);',
    '  vec3 flowCol=mix(col1,col2,n2);',
    '',
    '  // Compose final color',
    '  vec3 col=vec3(0.);',
    '',
    '  // Background glow',
    '  float bgGlow=exp(-length(uv)*1.5)*.15;',
    '  col+=deep*bgGlow;',
    '',
    '  // Flow field',
    '  col+=flowCol*flow*.25;',
    '',
    '  // Light rays',
    '  vec3 rayCol=mix(purple,pink,sin(t*.3+length(uv))*.5+.5);',
    '  col+=rayCol*light;',
    '',
    '  // Spiraling streaks',
    '  vec3 streakCol=mix(lilac,cyan,n3);',
    '  col+=streakCol*streak;',
    '',
    '  // Particles',
    '  vec3 ptCol=mix(lilac,vec3(1.),pts*.2);',
    '  col+=ptCol*pts*.15;',
    '',
    '  // Edge vignette',
    '  float vig=1.-pow(length(uv)*0.7,2.5);',
    '  vig=max(vig,0.);',
    '  col*=vig;',
    '',
    '  // Tone mapping (ACES-like)',
    '  col=col*(2.51*col+.03)/(col*(2.43*col+.59)+.14);',
    '',
    '  // Theme-aware background blend',
    '  vec3 bg=mix(vec3(.98,.98,.99),vec3(.051,.067,.09),u_dark);',
    '  float alpha=max(max(col.r,col.g),col.b);',
    '  alpha=smoothstep(0.,.15,alpha);',
    '  col=mix(bg,col,min(alpha*1.5+.1,1.));',
    '',
    '  O=vec4(col,1.);',
    '}',
  ].join('\n')

  // ── Compile shader program ────────────────────────────────────────
  function compile(type, src) {
    var s = gl.createShader(type)
    gl.shaderSource(s, src)
    gl.compileShader(s)
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
      console.error('Shader error:', gl.getShaderInfoLog(s))
      return null
    }
    return s
  }

  var vs = compile(gl.VERTEX_SHADER, VERT)
  var fs = compile(gl.FRAGMENT_SHADER, FRAG)
  if (!vs || !fs) return

  var prog = gl.createProgram()
  gl.attachShader(prog, vs)
  gl.attachShader(prog, fs)
  gl.linkProgram(prog)
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    console.error('Link error:', gl.getProgramInfoLog(prog))
    return
  }
  gl.useProgram(prog)

  // ── Fullscreen quad ───────────────────────────────────────────────
  var buf = gl.createBuffer()
  gl.bindBuffer(gl.ARRAY_BUFFER, buf)
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW)

  var aLoc = gl.getAttribLocation(prog, 'a')
  gl.enableVertexAttribArray(aLoc)
  gl.vertexAttribPointer(aLoc, 2, gl.FLOAT, false, 0, 0)

  // ── Uniforms ──────────────────────────────────────────────────────
  var uTime = gl.getUniformLocation(prog, 'u_time')
  var uRes = gl.getUniformLocation(prog, 'u_res')
  var uMouse = gl.getUniformLocation(prog, 'u_mouse')
  var uDark = gl.getUniformLocation(prog, 'u_dark')

  // ── State ─────────────────────────────────────────────────────────
  var mouseX = 0.5, mouseY = 0.5
  var targetMX = 0.5, targetMY = 0.5
  var animId = null

  function isDark() {
    var el = document.querySelector('[data-md-color-scheme]')
    return el && el.getAttribute('data-md-color-scheme') === 'slate' ? 1.0 : 0.0
  }

  // ── Resize ────────────────────────────────────────────────────────
  function resize() {
    var rect = container.getBoundingClientRect()
    var dpr = Math.min(window.devicePixelRatio, 2)
    var w = rect.width
    var h = Math.min(520, Math.max(360, w * 0.5))
    canvas.width = w * dpr
    canvas.height = h * dpr
    canvas.style.height = h + 'px'
    gl.viewport(0, 0, canvas.width, canvas.height)
  }

  // ── Events ────────────────────────────────────────────────────────
  container.addEventListener('mousemove', function (e) {
    var rect = container.getBoundingClientRect()
    targetMX = (e.clientX - rect.left) / rect.width
    targetMY = 1.0 - (e.clientY - rect.top) / rect.height
  })
  container.addEventListener('mouseleave', function () {
    targetMX = 0.5
    targetMY = 0.5
  })

  // ── Render loop ───────────────────────────────────────────────────
  function render(time) {
    var t = time * 0.001

    // Smooth mouse interpolation
    mouseX += (targetMX - mouseX) * 0.04
    mouseY += (targetMY - mouseY) * 0.04

    gl.uniform1f(uTime, t)
    gl.uniform2f(uRes, canvas.width, canvas.height)
    gl.uniform2f(uMouse, mouseX, mouseY)
    gl.uniform1f(uDark, isDark())

    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4)

    animId = requestAnimationFrame(render)
  }

  // ── Init ──────────────────────────────────────────────────────────
  resize()
  window.addEventListener('resize', resize)
  animId = requestAnimationFrame(render)

  // Pause when off-screen
  if (typeof IntersectionObserver !== 'undefined') {
    new IntersectionObserver(function (entries) {
      if (entries[0].isIntersecting) {
        if (!animId) animId = requestAnimationFrame(render)
      } else {
        if (animId) { cancelAnimationFrame(animId); animId = null }
      }
    }).observe(container)
  }
})()
