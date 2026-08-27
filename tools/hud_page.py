"""The HUD document. Kept apart from the server so neither file becomes unreadable."""

PAGE = r"""<!doctype html><meta charset=utf-8>
<title>Motion Studio</title>
<meta name=viewport content="width=device-width,initial-scale=1">
<style>
:root{
  --bg:#08090b; --panel:#0e1013; --sunk:#050607; --line:#1c2027; --line2:#262c35;
  --fg:#e8eaed; --dim:#7d8590; --faint:#4d545e;
  --ok:#2ea043; --bad:#e5534b; --warn:#c9a227; --accent:#3b82f6;
  --mono:ui-monospace,"SF Mono",Menlo,monospace;
}
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{background:var(--bg);color:var(--fg);font:13px/1.45 -apple-system,BlinkMacSystemFont,"Inter",system-ui,sans-serif;overflow:hidden}
.num{font-family:var(--mono);font-variant-numeric:tabular-nums}

#top{height:44px;display:flex;align-items:center;gap:16px;padding:0 16px;border-bottom:1px solid var(--line);background:var(--panel)}
#top b{font-size:13px;font-weight:600;letter-spacing:-.01em}
.tag{font-family:var(--mono);font-size:10px;letter-spacing:.1em;color:var(--warn);border:1px solid #3a3212;background:#191505;padding:2px 7px;border-radius:3px}
#top .sp{flex:1}
#clock{font-family:var(--mono);font-size:11px;color:var(--faint)}

#grid{display:grid;grid-template-columns:1fr 300px 300px;grid-template-rows:1fr 210px;height:calc(100% - 44px)}
@media(max-width:1180px){#grid{grid-template-columns:1fr 280px;grid-template-rows:1fr 200px 200px}}
.pane{border-right:1px solid var(--line);border-bottom:1px solid var(--line);display:flex;flex-direction:column;min-height:0;min-width:0;background:var(--panel)}
.pane:last-child{border-right:0}
.ph{height:30px;flex:0 0 30px;display:flex;align-items:center;gap:8px;padding:0 11px;border-bottom:1px solid var(--line);background:var(--sunk)}
.ph h2{font-size:10px;font-weight:600;letter-spacing:.09em;text-transform:uppercase;color:var(--dim)}
.ph .sp{flex:1}
.pb{flex:1;overflow:auto;min-height:0}
.pb.pad{padding:10px 11px}

#stage{position:relative;background:radial-gradient(ellipse at 50% 40%,#12161c 0%,#08090b 70%);grid-column:1;grid-row:1/2}
@media(max-width:1180px){#stage{grid-row:1/2;grid-column:1/-1}}
#stage canvas{display:block;width:100%;height:100%}
#hint{position:absolute;left:12px;bottom:10px;font-family:var(--mono);font-size:10px;color:var(--faint);pointer-events:none}
#load{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-family:var(--mono);font-size:11px;color:var(--dim);background:var(--bg)}
#hud3d{position:absolute;left:12px;top:12px;font-family:var(--mono);font-size:11px;line-height:1.7;pointer-events:none;text-shadow:0 1px 3px #000}
#hud3d .k{color:var(--faint)}
#hud3d .l{color:#7dd3fc}#hud3d .r{color:#fca5a5}
#clr{position:absolute;left:12px;top:96px;font-family:var(--mono);font-size:11px;
  line-height:1.6;pointer-events:none;text-shadow:0 1px 3px #000}
#clr .blocked{color:#fca5a5;border:1px solid #5b2320;background:#2a100fdd;
  padding:3px 8px;border-radius:4px;display:inline-block}
#clr .clear{color:var(--faint)}

button{background:#161a20;color:var(--fg);border:1px solid var(--line2);border-radius:4px;padding:4px 9px;font:inherit;font-size:12px;cursor:pointer;transition:.12s}
button:hover:not(:disabled){background:#1d222a;border-color:#39424f}
button:active:not(:disabled){transform:translateY(1px)}
button:disabled{opacity:.35;cursor:default}
button.go{border-color:#1d4ed8;background:#0f1f3d;color:#bfdbfe}
button.go:hover:not(:disabled){background:#15294f}
button.stop{border-color:#5b2320;background:#2a100f;color:#fca5a5}
select,input[type=text]{background:var(--sunk);color:var(--fg);border:1px solid var(--line2);border-radius:4px;padding:4px 7px;font:inherit;font-size:12px}
input[type=text]::placeholder{color:var(--faint)}

.jrow{display:grid;grid-template-columns:88px 1fr 52px;gap:8px;align-items:center;padding:2px 0}
.jrow label{font-family:var(--mono);font-size:10px;color:var(--dim);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.jrow .v{font-family:var(--mono);font-size:10px;color:var(--fg);text-align:right}
input[type=range]{-webkit-appearance:none;appearance:none;height:3px;background:var(--line2);border-radius:2px;outline:0}
input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:11px;height:11px;border-radius:50%;background:var(--accent);cursor:grab;border:2px solid var(--panel)}
input[type=range]::-webkit-slider-thumb:active{cursor:grabbing;background:#93c5fd}
.grp{font-size:9px;letter-spacing:.1em;text-transform:uppercase;color:var(--faint);margin:9px 0 3px;padding-bottom:3px;border-bottom:1px solid var(--line)}
.grp:first-child{margin-top:0}

table{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:11px}
th{text-align:right;color:var(--faint);font-weight:400;font-size:9px;letter-spacing:.07em;text-transform:uppercase;padding:0 0 4px}
th:first-child,td:first-child{text-align:left}
td{text-align:right;padding:2px 0;color:var(--dim);border-bottom:1px solid #12151a}
td.hi{color:var(--fg)}
.ok{color:var(--ok)}.bad{color:var(--bad)}.warn{color:var(--warn)}

#log{font-family:var(--mono);font-size:11px;line-height:1.5;white-space:pre-wrap;word-break:break-word;padding:9px 11px;color:var(--dim)}
#log .cmd{color:var(--accent)}
.job{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:5px 0;border-bottom:1px solid #12151a}
.job span{font-size:12px}
.dot{width:6px;height:6px;border-radius:50%;background:var(--faint);flex:0 0 6px}
.dot.run{background:var(--warn);animation:p 1s infinite}
@keyframes p{50%{opacity:.25}}
.pill{font-family:var(--mono);font-size:10px;padding:1px 7px;border-radius:3px;border:1px solid}
.pill.ok{border-color:#1a4c28;background:#0c1f13;color:#4ade80}
.pill.bad{border-color:#5b2320;background:#2a100f;color:#fca5a5}
.chk{display:flex;justify-content:space-between;gap:8px;font-family:var(--mono);font-size:10px;padding:1.5px 0;color:var(--faint)}
.chk b{font-weight:400}
</style>

<div id=top>
  <b>Motion Studio</b>
  <span class=tag>SIM ONLY</span>
  <span class=sp></span>
  <span id=clock></span>
</div>

<div id=grid>
  <div id=stage class=pane>
    <div id=load>loading robot…</div>
    <div id=hud3d></div>
    <div id=clr></div>
    <div id=hint>drag a link to turn its joint · drag empty space to orbit · scroll zoom</div>
  </div>

  <div class=pane style="grid-column:2;grid-row:1">
    <div class=ph><h2>Kinematics</h2><span class=sp></span>
      <button id=home>rest</button><button id=neutral>neutral</button><button id=zero>zero</button></div>
    <div class="pb pad" id=sliders></div>
  </div>

  <div class=pane style="grid-column:3;grid-row:1">
    <div class=ph><h2>Frames</h2><span class=sp></span>
      <input type=text id=label placeholder=label size=8><button id=save>record</button></div>
    <div class="pb pad"><div id=meas></div><div id=saved class=num style="font-size:10px;color:var(--faint);margin-top:8px"></div></div>
  </div>

  <div class=pane style="grid-column:2/-1;grid-row:2">
    <div class=ph><h2>Run</h2><span class=sp></span>
      <select id=camsel title="capture device"></select>
      <select id=cap></select><input type=text id=newcap placeholder=new size=7>
      <button id=stopbtn class=stop>stop</button></div>
    <div class="pb pad"><div id=jobs></div></div>
  </div>

  <div class=pane style="grid-row:2;grid-column:1" id=logpane>
    <div class=ph><h2>Console</h2><span class=sp></span><span id=status class=num style="font-size:10px;color:var(--faint)"></span></div>
    <div class=pb><div id=log>idle</div></div>
  </div>
</div>

<script type="importmap">
{"imports":{"three":"/vendor/three.module.js","three/addons/":"/vendor/","three/examples/jsm/":"/vendor/","three/examples/jsm/loaders/":"/vendor/","three/examples/jsm/utils/":"/vendor/"}}
</script>
<script type="module">
import * as THREE from 'three';
import {OrbitControls} from '/vendor/OrbitControls.js';
import {STLLoader} from '/vendor/STLLoader.js';
import {GLTFLoader} from '/vendor/GLTFLoader.js';
import URDFLoader from '/vendor/URDFLoader.js';
import {URDFDragControls} from '/vendor/URDFDragControls.js';

const $=id=>document.getElementById(id);
const D2R=Math.PI/180, R2D=180/Math.PI;
const stage=$('stage');

const scene=new THREE.Scene();
scene.background=null;
const camera=new THREE.PerspectiveCamera(38,1,0.01,100);
camera.position.set(1.6,1.1,1.6);
const renderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
renderer.setPixelRatio(Math.min(devicePixelRatio,2));
renderer.shadowMap.enabled=true; renderer.shadowMap.type=THREE.PCFSoftShadowMap;
renderer.toneMapping=THREE.ACESFilmicToneMapping; renderer.toneMappingExposure=1.05;
stage.appendChild(renderer.domElement);

scene.add(new THREE.HemisphereLight(0x9db8d6,0x0a0c10,1.5));
const key=new THREE.DirectionalLight(0xffffff,2.2); key.position.set(2.4,4,2.2);
key.castShadow=true; key.shadow.mapSize.set(2048,2048);
key.shadow.camera.top=1.6;key.shadow.camera.bottom=-1.6;key.shadow.camera.left=-1.6;key.shadow.camera.right=1.6;
key.shadow.bias=-0.0008; scene.add(key);
const rim=new THREE.DirectionalLight(0x6ea8ff,0.7); rim.position.set(-2.5,1.5,-2); scene.add(rim);

const grid=new THREE.GridHelper(6,60,0x2a323d,0x151a20);
grid.material.transparent=true; grid.material.opacity=.55; scene.add(grid);
const floor=new THREE.Mesh(new THREE.PlaneGeometry(6,6),new THREE.ShadowMaterial({opacity:.42}));
floor.rotation.x=-Math.PI/2; floor.receiveShadow=true; scene.add(floor);

const controls=new OrbitControls(camera,renderer.domElement);
controls.enableDamping=true; controls.dampingFactor=.08;
controls.target.set(0,0.85,0); controls.minDistance=.5; controls.maxDistance=6;

function resize(){
  const w=stage.clientWidth,h=stage.clientHeight;
  camera.aspect=w/h; camera.updateProjectionMatrix(); renderer.setSize(w,h,false);
}
addEventListener('resize',resize); resize();

const stl=new STLLoader(), gltf=new GLTFLoader();
const loader=new URDFLoader();
loader.loadMeshCb=(path,manager,done)=>{
  const ext=path.split('.').pop().toLowerCase();
  if(ext==='glb'||ext==='gltf'){
    gltf.load(path,r=>done(r.scene),undefined,e=>done(null,e));
  }else if(ext==='stl'){
    stl.load(path,g=>done(new THREE.Mesh(g,new THREE.MeshStandardMaterial(
      {color:0xb8c0cc,metalness:.35,roughness:.55}))),undefined,e=>done(null,e));
  }else done(null,new Error('unsupported '+ext));
};

let robot=null, JOINTS=[], HOME={}, NEUTRAL={};
loader.load('/robot/urdf/galbot_one_golf_fixed_base.urdf',model=>{
  // URDFLoader calls this inside a promise chain whose .catch routes to onError,
  // so anything thrown here vanishes unless it is caught and shown.
  try{
    robot=model;
    robot.rotation.x=-Math.PI/2;          // URDF Z-up -> three.js Y-up
    robot.traverse(o=>{ if(o.isMesh){o.castShadow=true;o.receiveShadow=true;} });
    scene.add(robot);
    const b=new THREE.Box3().setFromObject(robot);
    if(isFinite(b.min.y)){ const c=b.getCenter(new THREE.Vector3());
      controls.target.set(c.x,c.y,c.z); controls.update(); }
    const el=$('load'); if(el)el.remove();
    boot();
    enableDirectDrag();
  }catch(e){ fail(e); }
},undefined,e=>fail(e));

function fail(e){
  const el=$('load');
  if(el){el.textContent='robot failed to load — '+String(e&&e.message||e).slice(0,180);
    el.style.color='#fca5a5';}
  console.error('urdf load failed',e);
}

function boot(){
  fetch('/api/robot/meta').then(r=>r.json()).then(meta=>{
    HOME=meta.home_deg||{}; NEUTRAL=meta.neutral_deg||{};
    JOINTS=meta.joints.filter(j=>robot.joints[j.name]);
    let html='',grp=null;
    JOINTS.forEach(j=>{
      if(j.group!==grp){grp=j.group;html+=`<div class=grp>${grp}</div>`;}
      html+=`<div class=jrow><label title="${j.name}">${j.name.replace(/_joint/,' j')}</label>
        <input type=range data-j="${j.name}" min="${j.min_deg}" max="${j.max_deg}" step=.5 value="0">
        <span class=v id="v_${j.name}">0.0</span></div>`;
    });
    $('sliders').innerHTML=html;
    // The home pose also fixes the leg column. Those joints get no slider, but
    // leaving them at zero puts the torso at the wrong height and every world
    // coordinate reported here disagrees with the simulator's.
    Object.keys(HOME).forEach(n=>{
      if(robot.joints[n] && !JOINTS.some(j=>j.name===n)) robot.setJointValue(n,HOME[n]*D2R);
    });
    document.querySelectorAll('#sliders input').forEach(s=>{
      s.oninput=()=>setJoint(s.dataset.j,+s.value);
    });
    setAll(n=>HOME[n]||0);
    $('saved').textContent=meta.saved?meta.saved+' frames recorded':'';
  });
}
function setJoint(name,deg){
  robot.setJointValue(name,deg*D2R);
  checkClearance();
  const el=$('v_'+name); if(el)el.textContent=deg.toFixed(1);
  const s=document.querySelector(`#sliders input[data-j="${name}"]`);
  if(s&&+s.value!==deg)s.value=deg;
  measure();
}
function setAll(fn){ JOINTS.forEach(j=>setJoint(j.name,fn(j.name))); }
$('home').onclick=()=>setAll(n=>HOME[n]||0);
$('neutral').onclick=()=>setAll(n=>NEUTRAL[n]||0);
$('zero').onclick=()=>setAll(()=>0);

const W=new THREE.Vector3();
// The robot is rotated -90 deg about X to put URDF Z-up into three.js Y-up, so
// three (x,y,z) = URDF (x, z, -y). Inverting that is (x, -z, y) -- getting the
// sign wrong here silently mirrors every number the panel reports.
const wp=n=>{const o=robot.links[n]; if(!o)return null; o.getWorldPosition(W);
  return [W.x,-W.z,W.y].map(v=>+v.toFixed(4));};
const SIDES={left:['left_arm_link1','left_arm_link4','left_gripper_tcp_link'],
             right:['right_arm_link1','right_arm_link4','right_gripper_tcp_link']};
const sub=(a,b)=>[a[0]-b[0],a[1]-b[1],a[2]-b[2]];
const len=v=>Math.hypot(v[0],v[1],v[2]);

function swivel(s,e,w){
  const a=sub(w,s),na=len(a); if(na<1e-9)return null;
  const ax=a.map(v=>v/na), u0=sub(e,s), d=u0[0]*ax[0]+u0[1]*ax[1]+u0[2]*ax[2];
  const p=[u0[0]-d*ax[0],u0[1]-d*ax[1],u0[2]-d*ax[2]], r=len(p);
  if(r<1e-6)return null;
  const u=p.map(v=>v/r), ref=[0,0,-1], dr=ref[0]*ax[0]+ref[1]*ax[1]+ref[2]*ax[2];
  let rp=[ref[0]-dr*ax[0],ref[1]-dr*ax[1],ref[2]-dr*ax[2]], nr=len(rp);
  if(nr<1e-6){const f=[1,0,0],df=f[0]*ax[0]+f[1]*ax[1]+f[2]*ax[2];
    rp=[f[0]-df*ax[0],f[1]-df*ax[1],f[2]-df*ax[2]];nr=len(rp); if(nr<1e-6)return null;}
  const rh=rp.map(v=>v/nr);
  const cr=[ax[1]*rh[2]-ax[2]*rh[1],ax[2]*rh[0]-ax[0]*rh[2],ax[0]*rh[1]-ax[1]*rh[0]];
  return {deg:Math.atan2(cr[0]*u[0]+cr[1]*u[1]+cr[2]*u[2],
                         rh[0]*u[0]+rh[1]*u[1]+rh[2]*u[2])*R2D, r};
}
let LAST={};
function measureOnly(){ measure(); }
function measure(){
  if(!robot)return;
  const rows=[]; LAST={};
  for(const side in SIDES){
    const [sn,en,tn]=SIDES[side], s=wp(sn),e=wp(en),t=wp(tn);
    if(!s||!e||!t)continue;
    const sw=swivel(s,e,t), lat=+(e[1]-s[1]).toFixed(4);
    LAST[side]={shoulder_m:s,elbow_m:e,tcp_m:t,elbow_lateral_m:lat,
      swivel_deg:sw?+sw.deg.toFixed(2):null,reach_m:+len(sub(t,s)).toFixed(4)};
    const cls=Math.abs(lat)>.30?'bad':Math.abs(lat)>.24?'warn':'ok';
    rows.push(`<tr><td class=hi>${side}</td><td class="hi ${cls}">${lat.toFixed(4)}</td>
      <td>${sw?sw.deg.toFixed(1):'—'}</td><td>${len(sub(t,s)).toFixed(3)}</td></tr>`);
  }
  $('meas').innerHTML='<table><tr><th>arm<th>elbow lat<th>swivel<th>reach</tr>'+rows.join('')+'</table>'+
    (LAST.left?`<table style="margin-top:9px"><tr><th>pt<th>x<th>y<th>z</tr>`+
      ['shoulder','elbow','tcp'].map(k=>{const v=LAST.left[k+'_m'];
        return `<tr><td class=hi>L ${k}</td>`+v.map(n=>`<td>${n.toFixed(3)}</td>`).join('')+'</tr>';}).join('')+
      '</table>':'');
  const L=LAST.left||{},R=LAST.right||{};
  $('hud3d').innerHTML=
    `<span class=k>L elbow lat</span> <span class=l>${(L.elbow_lateral_m??0).toFixed(4)} m</span><br>`+
    `<span class=k>L swivel</span> <span class=l>${L.swivel_deg??'—'}&deg;</span><br>`+
    `<span class=k>R elbow lat</span> <span class=r>${(R.elbow_lateral_m??0).toFixed(4)} m</span><br>`+
    `<span class=k>R swivel</span> <span class=r>${R.swivel_deg??'—'}&deg;</span>`;
}
// Grab a link and turn its joint directly. The sliders and the model stay in
// step because both go through setJoint, which is the single writer.
//
// The library's PointerURDFDragControls is not used: it computes mouse position
// as pageX - domElement.offsetLeft over offsetWidth, which is only correct when
// the canvas sits at the document origin. This canvas lives in a grid cell 44 px
// down, so every ray missed. The base class is driven here with rays built from
// getBoundingClientRect instead, which is correct wherever the canvas sits.
let dragging=null;
function enableDirectDrag(){
  const dc=new URDFDragControls(scene);
  const ray=new THREE.Raycaster(), ndc=new THREE.Vector2();
  const aim=e=>{
    const r=renderer.domElement.getBoundingClientRect();
    ndc.x=((e.clientX-r.left)/r.width)*2-1;
    ndc.y=-((e.clientY-r.top)/r.height)*2+1;
    ray.setFromCamera(ndc,camera);
    dc.moveRay(ray.ray);
  };
  dc.updateJoint=(joint,angle)=>{
    const lim=JOINTS.find(j=>j.name===joint.name);
    let deg=angle*R2D;
    if(lim)deg=Math.max(lim.min_deg,Math.min(lim.max_deg,deg));
    setJoint(joint.name,+deg.toFixed(2));
  };
  dc.onHover=j=>{ if(!dragging){renderer.domElement.style.cursor='grab';mark(j.name,true);} };
  dc.onUnhover=j=>{ if(!dragging){renderer.domElement.style.cursor='';mark(j.name,false);} };
  dc.onDragStart=j=>{dragging=j.name;renderer.domElement.style.cursor='grabbing';mark(j.name,true);};
  dc.onDragEnd=j=>{dragging=null;renderer.domElement.style.cursor='';mark(j.name,false);};

  const el=renderer.domElement;
  el.addEventListener('mousemove',e=>{aim(e);});
  el.addEventListener('mousedown',e=>{
    aim(e);
    // Only take the gesture when a joint is actually under the cursor, so empty
    // space still orbits.
    if(dc.hovered){controls.enabled=false;dc.setGrabbed(true);}
  });
  addEventListener('mouseup',e=>{
    if(dragging!==null){aim(e);dc.setGrabbed(false);}
    controls.enabled=true;
  });
}
// Self-clearance, from the project's own ClearanceChecker at the SIM floor --
// the same judgement the supervisor makes, so a pose refused here is a pose the
// pipeline would refuse. Checked on a trailing debounce because it is a server
// round trip; the last configuration that passed is kept so a blocked drag
// springs back instead of leaving the arm inside itself.
let lastSafe=null, clrTimer=null, clrBusy=false;
function snapshotJoints(){
  const o={}; JOINTS.forEach(j=>{
    const s=document.querySelector(`#sliders input[data-j="${j.name}"]`);
    if(s)o[j.name]=(+s.value)*D2R;});
  return o;
}
function checkClearance(){
  clearTimeout(clrTimer);
  clrTimer=setTimeout(async()=>{
    if(clrBusy)return; clrBusy=true;
    const joints=snapshotJoints();
    try{
      const r=await fetch('/api/clearance',{method:'POST',
        headers:{'content-type':'application/json'},body:JSON.stringify({joints})});
      const c=await r.json();
      if(c.blocked){
        $('clr').innerHTML=`<span class=blocked>BLOCKED ${(c.min_distance_m*1000).toFixed(1)} mm `+
          `&lt; ${(c.floor_m*1000).toFixed(0)} mm floor<br>${c.pair||''}</span>`;
        if(lastSafe)Object.entries(lastSafe).forEach(([n,v])=>setJointRaw(n,v*R2D));
      }else{
        lastSafe=joints;
        $('clr').innerHTML=`<span class=clear>clearance ${(c.min_distance_m*1000).toFixed(1)} mm</span>`;
      }
    }catch(e){ /* server busy with a run; leave the last reading up */ }
    finally{ clrBusy=false; }
  },90);
}
// setJointRaw does not re-trigger the clearance check, so springing back cannot
// loop on itself.
function setJointRaw(name,deg){
  robot.setJointValue(name,deg*D2R);
  const el=$('v_'+name); if(el)el.textContent=(+deg).toFixed(1);
  const s=document.querySelector(`#sliders input[data-j="${name}"]`);
  if(s)s.value=deg;
  measureOnly();
}

function mark(name,on){
  const row=document.querySelector(`#sliders input[data-j="${name}"]`);
  const box=row&&row.closest('.jrow');
  if(box)box.style.background=on?'#152034':'';
}

$('save').onclick=async()=>{
  const joints={};
  JOINTS.forEach(j=>{joints[j.name]=(+document.querySelector(`#sliders input[data-j="${j.name}"]`).value)*D2R;});
  const r=await fetch('/api/pose/save',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({joints,label:$('label').value,measurements:LAST})});
  const j=await r.json(); $('saved').textContent=j.saved+' frames recorded'; $('label').value='';
};

(function loop(){requestAnimationFrame(loop);controls.update();renderer.render(scene,camera);})();

/* ---- run / console / gate ---- */
let cur=null, CAMSTATE=null;
async function poll(){
  const s=await(await fetch('/api/state')).json();
  const sel=$('cap'),names=s.captures.map(c=>c.name);
  if(sel.options.length!==names.length){sel.innerHTML=names.map(n=>`<option>${n}</option>`).join('');
    if(cur&&names.includes(cur))sel.value=cur;}
  cur=sel.value||null;
  const r=s.runner;
  if(!$('jobs').dataset.done){
    $('jobs').dataset.done='1';
    $('jobs').innerHTML=s.jobs.map(j=>`<div class=job><span class=dot data-d="${j.id}"></span>
      <span>${j.label}</span><span style="flex:1"></span>
      <button class="go j" data-job="${j.id}">run</button></div>`).join('');
    document.querySelectorAll('.j').forEach(b=>b.onclick=async()=>{
      const n=$('newcap').value.trim()||cur; if(!n)return;
      await fetch('/api/run',{method:'POST',headers:{'content-type':'application/json'},
        body:JSON.stringify({job:b.dataset.job,capture:n})});
      $('newcap').value='';cur=n;poll();});
  }
  document.querySelectorAll('.j').forEach(b=>b.disabled=r.busy||!!b.dataset.blocked);
  applyCameraState();
  document.querySelectorAll('.dot').forEach(d=>
    d.classList.toggle('run',r.busy&&r.job===d.dataset.d));
  $('stopbtn').disabled=!r.busy;
  $('status').textContent=r.busy?`${r.job} · ${r.started}`:(r.exit===null?'':`exit ${r.exit}`);
  const log=$('log');const bottom=log.parentElement.scrollTop+log.parentElement.clientHeight>=log.parentElement.scrollHeight-30;
  log.innerHTML=r.log.length?r.log.map(l=>l.startsWith('$')?`<span class=cmd>${esc(l)}</span>`:esc(l)).join('\n'):'idle';
  if(bottom)log.parentElement.scrollTop=log.parentElement.scrollHeight;
  const cap=s.captures.find(c=>c.name===cur);
  const g=cap&&cap.gate;
  let extra='';
  if(cap){const b=cap.bound;
    extra=`<div class=chk style="margin-top:8px"><b>${cap.name}</b><span class="${b.consistent?'ok':b.consistent===false?'bad':''}">${b.consistent===null?'no capture':b.consistent?'bound':'UNBOUND'}${b.frames?' · '+b.frames+'f':''}</span></div>`;
    extra+=cap.files.filter(f=>f.present).map(f=>`<div class=chk><b>${f.name}</b><span>${f.mtime}</span></div>`).join('');
    if(g)extra+=`<div style="margin-top:8px"><span class="pill ${g.verdict==='PASS'?'ok':'bad'}">${g.verdict}</span></div>`+
      g.checks.map(c=>`<div class=chk><b class="${c.passed?'':'bad'}">${c.name}</b><span class="${c.passed?'ok':'bad'}">${c.passed?'pass':'fail'}</span></div>`).join('');
  }
  const holder=$('jobs');
  let ex=document.getElementById('extra'); if(!ex){ex=document.createElement('div');ex.id='extra';holder.after(ex);}
  ex.innerHTML=extra;
}
function applyCameraState(){
  const c=CAMSTATE;
  if(!c||!c.checked||c.usable)return;
  const btn=document.querySelector('.j[data-job="capture"]');
  if(btn&&!btn.dataset.blocked){
    btn.dataset.blocked='1';
    btn.disabled=true; btn.textContent='terminal only';
    btn.title='No camera access in this process. Run from your terminal:\n'+(c.terminal_hint||'');
    const row=btn.closest('.job');
    const lbl=row&&row.querySelectorAll('span')[1];
    if(lbl)lbl.innerHTML='New capture (webcam) '+
      '<span style="color:var(--warn)">&mdash; no camera access here, run it from your terminal</span>';
  }
  const sel=$('camsel'); if(sel)sel.disabled=true;
}
const esc=s=>s.replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
// Camera picker. OpenCV can only address a device by index and macOS will hand
// a nearby iPhone out as index 0, so the choice is shown by name and remembered.
fetch('/api/cameras').then(r=>r.json()).then(c=>{
  const sel=$('camsel');
  if(!c.devices.length){sel.innerHTML='<option value=0>camera 0</option>';return;}
  sel.innerHTML='<option value="-1">auto: built-in</option>'+c.devices.map(d=>
    `<option value="${d.index}"${!c.auto&&d.index===c.selected?' selected':''}>`+
    `${d.phone?'phone: ':''}${d.name}</option>`).join('');
  if(c.auto)sel.value='-1';
  sel.title=c.why||'capture device';
  // Capture needs a camera grant this process may not have. Say so on the button
  // instead of letting it launch a subprocess that dies in a traceback.
  CAMSTATE=c; applyCameraState();
  const chosen=c.devices.find(d=>d.index===c.selected);
  if(!c.auto&&chosen&&chosen.phone)sel.style.borderColor='var(--bad)';
  sel.onchange=async()=>{
    await fetch('/api/camera',{method:'POST',headers:{'content-type':'application/json'},
      body:JSON.stringify({index:+sel.value})});
    const d=c.devices.find(x=>x.index===+sel.value);
    sel.style.borderColor=d&&d.phone?'var(--bad)':'';
  };
});
$('stopbtn').onclick=()=>fetch('/api/stop',{method:'POST'}).then(poll);
$('cap').onchange=e=>{cur=e.target.value;poll();};
setInterval(poll,1000); poll();
setInterval(()=>{$('clock').textContent=new Date().toTimeString().slice(0,8);},1000);
</script>
"""
