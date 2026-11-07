'use strict';
const $ = id => document.getElementById(id);
const plan = $('plan'), pc = plan.getContext('2d'), frame = $('frame'), fc = frame.getContext('2d');
let state, videos = [], selected = null, selectedWall = null, mode = 'select', draft = null, pointer = null, dragging = false;
let image = null, imageCamera = null, frameRequest = 0, videoRequest = 0, pick = true, testPoint = null, saveTimer, dirty = false, saveQueue = Promise.resolve();
let planAssets = [], planImage = null, planImageRequest = 0, activePair = 0, pairSide = 'video', testVideoPoint = null, calibrationTimer;
const calibrationPlan = $('calibrationPlan'), qc = calibrationPlan.getContext('2d');
let frameView = {scale:1,ox:0,oy:0}, calibrationView = {scale:1,ox:0,oy:0};
const frameZoom = {zoom:1,panX:0,panY:0}, planZoom = {zoom:1,panX:0,panY:0};
const unit = () => state?.coordinate_units==='relative'?'ед.':'м';
let view = {scale: 1, ox: 0, oy: 0};
const camera = () => state?.cameras.find(c => c.id === selected);
const escapeId = prefix => prefix + '_' + crypto.randomUUID().replaceAll('-', '').slice(0, 12);
function toast(text) { $('toast').textContent = text; $('toast').style.display = 'block'; clearTimeout(toast.timer); toast.timer = setTimeout(() => $('toast').style.display = 'none', 4500); }
async function api(url, options) { const response = await fetch(url, options); const result = await response.json(); if (!response.ok) throw Error(result.error || `Ошибка ${response.status}`); return result; }
const post = (url, data) => api(url, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(data)});
function changed() { dirty = true; $('saveStatus').textContent = 'Изменения…'; clearTimeout(saveTimer); saveTimer = setTimeout(save, 400); }
function save() { const snapshot = JSON.stringify(state); saveQueue = saveQueue.then(async () => { try { const result=await post('/api/project?preserve_timeout=1', JSON.parse(snapshot)); const unchanged=snapshot===JSON.stringify(state);if(result.timeout_s!=null&&state.processing)state.processing.timeout_s=result.timeout_s;if (unchanged) {dirty = false; $('saveStatus').textContent = 'Сохранено на компьютере';} } catch (error) { $('saveStatus').textContent = 'Не сохранено'; toast(error.message); } }); return saveQueue; }
window.addEventListener('beforeunload', e => { if (dirty) { e.preventDefault(); e.returnValue = ''; } });
function projectToPixel(p) { return [view.ox + p[0] * view.scale, view.oy + p[1] * view.scale]; }
function pixelToProject(e, snap = false) { const box = plan.getBoundingClientRect(); let x = (e.clientX - box.left - view.ox) / view.scale, y = (e.clientY - box.top - view.oy) / view.scale; if (x < 0 || y < 0 || x > state.room.width_m || y > state.room.height_m) return null; if (snap && $('snap').checked) {x = Math.round(x * 4) / 4; y = Math.round(y * 4) / 4;} return [Math.max(0, Math.min(state.room.width_m, x)), Math.max(0, Math.min(state.room.height_m, y))]; }
function line(a, b, color, width = 1, dash = []) { a = projectToPixel(a); b = projectToPixel(b); pc.beginPath(); pc.setLineDash(dash); pc.moveTo(...a); pc.lineTo(...b); pc.strokeStyle = color; pc.lineWidth = width; pc.stroke(); pc.setLineDash([]); }
function dot(ctx, x, y, label, color = '#d27835', size = 10) { ctx.beginPath(); ctx.arc(x, y, size, 0, Math.PI*2); ctx.fillStyle = color; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke(); ctx.fillStyle = '#fff'; ctx.font = `bold ${size * 1.2}px -apple-system, sans-serif`; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(label, x, y + .5); }
function drawPlan() {
  if (!state) return;
  const box = plan.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
  const w = box.width, h = box.height; if (!w || !h) return;
  plan.width = Math.round(w*dpr); plan.height = Math.round(h*dpr); pc.setTransform(dpr,0,0,dpr,0,0); pc.clearRect(0,0,w,h);
  const rw = state.room.width_m, rh = state.room.height_m;
  view.scale = Math.min((w-56)/rw,(h-56)/rh); view.ox = (w-rw*view.scale)/2; view.oy = (h-rh*view.scale)/2;
  pc.fillStyle = '#f7faf5'; pc.fillRect(view.ox,view.oy,rw*view.scale,rh*view.scale);
  if(planImage){pc.save();pc.globalAlpha=state.plan_background?.opacity??.85;pc.drawImage(planImage,view.ox,view.oy,rw*view.scale,rh*view.scale);pc.restore();}
  const step = Math.max(1, Math.ceil(Math.max(rw,rh)/50));
  if(!planImage) for (let x=0;x<=rw;x+=step) line([x,0],[x,rh],'#e3eae1');
  if(!planImage) for (let y=0;y<=rh;y+=step) line([0,y],[rw,y],'#e3eae1');
  pc.strokeStyle = '#c8d4c9'; pc.lineWidth = 1.5; pc.strokeRect(view.ox,view.oy,rw*view.scale,rh*view.scale);
  pc.font = '10px -apple-system,sans-serif'; pc.fillStyle = '#7f9387'; pc.textAlign='left';pc.textBaseline='alphabetic';
  pc.fillText('0, 0',view.ox,view.oy-10);pc.fillText(`${rw} ${unit()}`,view.ox+rw*view.scale-25,view.oy-10);pc.fillText(`${rh} ${unit()}`,view.ox+3,view.oy+rh*view.scale+17);
  for (const c of state.cameras) {
    const [x,y]=projectToPixel(c.position), yaw=c.yaw_deg*Math.PI/180, half=c.fov_deg*Math.PI/360, r=c.range_m*view.scale;
    pc.save(); pc.beginPath();pc.rect(view.ox,view.oy,rw*view.scale,rh*view.scale);pc.clip();
    pc.beginPath();pc.moveTo(x,y);pc.arc(x,y,r,yaw-half,yaw+half);pc.closePath();pc.fillStyle=c.id===selected?'#1479611a':'#14796108';pc.fill();pc.strokeStyle='#70a48a';pc.setLineDash([4,4]);pc.lineWidth=1;pc.stroke();pc.restore();
  }
  for (const wall of state.walls) line(wall.a,wall.b,wall.id===selectedWall?'#d27835':'#314c45',5);
  if(draft && pointer) line(draft,pointer,'#147961',3,[6,4]);
  for (const c of state.cameras) {
    const [x,y]=projectToPixel(c.position), yaw=c.yaw_deg*Math.PI/180;
    pc.beginPath();pc.arc(x,y,12,0,Math.PI*2);pc.fillStyle=c.id===selected?'#147961':'#5b796b';pc.fill();pc.lineWidth=2;pc.strokeStyle='#fff';pc.stroke();
    pc.beginPath();pc.moveTo(x+Math.cos(yaw)*9,y+Math.sin(yaw)*9);pc.lineTo(x+Math.cos(yaw)*24,y+Math.sin(yaw)*24);pc.strokeStyle='#147961';pc.lineWidth=3;pc.stroke();
    pc.font='11px -apple-system,sans-serif';pc.textAlign='center';pc.textBaseline='alphabetic';pc.fillStyle='#18302d';pc.fillText(c.name,x,y-19);
  }
  const c=camera(); if(c) c.plan_points.forEach((p,i)=>dot(pc,...projectToPixel(p),String(i+1)));
  if(testPoint) {const [x,y]=projectToPixel(testPoint);dot(pc,x,y,'✓','#147961',11);pc.font='11px -apple-system';pc.textAlign='left';pc.fillStyle='#147961';pc.fillText(`${testPoint[0].toFixed(2)}, ${testPoint[1].toFixed(2)} ${unit()}`,x+16,y-12);}
  drawCalibrationPlan();
  $('planSummary').textContent=`${state.walls.length} стен · ${state.cameras.length} камер`;
}
function viewport(canvas, ctx, sw, sh, zoom) {
 const box=canvas.getBoundingClientRect(),w=box.width,h=box.height,dpr=devicePixelRatio||1;
 if(!w||!h)return null;
 canvas.width=Math.round(w*dpr);canvas.height=Math.round(h*dpr);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);
 const scale=Math.min((w-24)/sw,(h-24)/sh)*zoom.zoom;
 return {scale,ox:w/2+zoom.panX-sw*scale/2,oy:h/2+zoom.panY-sh*scale/2,w,h};
}
function sourcePoint(canvas, view, e, width, height){const box=canvas.getBoundingClientRect();const p=[(e.clientX-box.left-view.ox)/view.scale,(e.clientY-box.top-view.oy)/view.scale];return p[0]>=0&&p[1]>=0&&p[0]<=width&&p[1]<=height?p:null;}
function drawFrame() {
 if(!image||imageCamera!==selected){fc.clearRect(0,0,frame.width,frame.height);$('noFrame').hidden=false;return;}
 $('noFrame').hidden=true;const v=viewport(frame,fc,image.naturalWidth,image.naturalHeight,frameZoom);if(!v)return;frameView=v;
 fc.drawImage(image,v.ox,v.oy,image.naturalWidth*v.scale,image.naturalHeight*v.scale);
 const c=camera();if(c)c.video_points.forEach((p,i)=>dot(fc,v.ox+p[0]*v.scale,v.oy+p[1]*v.scale,String(i+1),i===activePair?'#b85f20':'#d27835',10));
 if(testVideoPoint)dot(fc,v.ox+testVideoPoint[0]*v.scale,v.oy+testVideoPoint[1]*v.scale,'✓','#147961',10);
}
function drawCalibrationPlan(){
 if(!state)return;const v=viewport(calibrationPlan,qc,state.room.width_m,state.room.height_m,planZoom);if(!v)return;calibrationView=v;
 const xy=p=>[v.ox+p[0]*v.scale,v.oy+p[1]*v.scale];
 qc.fillStyle='#f7faf5';qc.fillRect(v.ox,v.oy,state.room.width_m*v.scale,state.room.height_m*v.scale);
 if(planImage){qc.save();qc.globalAlpha=state.plan_background?.opacity??.85;qc.drawImage(planImage,v.ox,v.oy,state.room.width_m*v.scale,state.room.height_m*v.scale);qc.restore();}
 qc.strokeStyle='#314c4580';qc.lineWidth=2;for(const wall of state.walls){qc.beginPath();qc.moveTo(...xy(wall.a));qc.lineTo(...xy(wall.b));qc.stroke();}
 for(const c of state.cameras){const p=xy(c.position);qc.fillStyle=c.id===selected?'#147961':'#738c7f';qc.beginPath();qc.arc(...p,5,0,Math.PI*2);qc.fill();}
 if(camera())camera().plan_points.forEach((p,i)=>dot(qc,...xy(p),String(i+1),i===activePair?'#b85f20':'#d27835',10));
 if(testPoint)dot(qc,...xy(testPoint),'✓','#147961',11);
}
function pairInstruction(){const c=camera();if(!c)return;$('pairInstruction').textContent=!pick?'Проверка: нажмите на другую точку пола в кадре. Зелёная отметка появится на плане.':c.video_points.length===4&&c.plan_points.length===4?'Все четыре пары заданы. Проверьте проекцию на независимой точке пола.':`Пара №${activePair+1}: нажмите ${pairSide==='video'?'точку пола на кадре слева':'ту же точку на плане справа'}.`;}
function placePair(side,p){const c=camera();if(!c)return;const key=side==='video'?'video_points':'plan_points';if(activePair>c[key].length)return toast('Сначала завершите предыдущую пару');c[key][activePair]=p;
 if(side==='video')pairSide='plan';else if(c.video_points[activePair]){const missing=[0,1,2,3].find(i=>!c.video_points[i]||!c.plan_points[i]);if(missing!==undefined)activePair=missing;pairSide='video';}else pairSide='video';
 invalidate();clearTimeout(calibrationTimer);if(c.video_points.length===4&&c.plan_points.length===4)calibrationTimer=setTimeout(()=>calculateCalibration(c),250);
}
function setMode(value) { mode=value;draft=null;pointer=null;document.querySelectorAll('[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===mode)); const hints={select:'Выберите камеру или стену. Камеру можно перетащить.',wall:'Два клика — начало и конец прямой стены. Esc отменяет.',camera:'Нажмите на схему, чтобы установить камеру.',calibrate:'Выберите четыре точки пола на схеме в том же порядке, что на кадре.'};$('modeHint').textContent=hints[mode];drawPlan(); }
function renderCameras() { $('cameraList').replaceChildren();$('cameraCount').textContent=state.cameras.length;$('emptyCameras').hidden=state.cameras.length>0;for(const c of state.cameras){const button=document.createElement('button');button.className='camera-item'+(c.id===selected?' active':'');const name=document.createElement('span');name.textContent=c.name;const badge=document.createElement('small');badge.textContent=c.calibration?'Привязана ✓':'Без привязки';button.append(name,badge);button.onclick=async()=>{await selectCamera(c.id);$('cameraPanel').scrollIntoView({behavior:'smooth',block:'start'});};$('cameraList').append(button);} }
function updatePairs() {
  const c=camera();if(!c)return;$('pairs').replaceChildren();for(let i=0;i<4;i++){const row=document.createElement('button');row.type='button';row.className='pair-row'+(i===activePair?' active':'');row.disabled=i>Math.min(c.video_points.length,c.plan_points.length);row.setAttribute('aria-label',`Выбрать пару ${i+1}`);const n=document.createElement('b');n.textContent=i+1;const a=document.createElement('span'),b=document.createElement('span');a.textContent=c.video_points[i]?c.video_points[i].map(v=>v.toFixed(0)).join(', '):'—';b.textContent=c.plan_points[i]?c.plan_points[i].map(v=>v.toFixed(2)).join(', '):'—';row.append(n,a,b);row.onclick=()=>{activePair=i;pick=true;$('pickMode').classList.add('active');$('testMode').classList.remove('active');pairSide=c.video_points[i]&&!c.plan_points[i]?'plan':'video';updatePairs();drawFrame();drawPlan();};$('pairs').append(row);}pairInstruction();
  $('calibrateBtn').disabled=c.video_points.length!==4||c.plan_points.length!==4;
  $('calibrationBadge').textContent=c.calibration?'Привязана ✓':'Не привязана';$('calibrationBadge').classList.toggle('ok',Boolean(c.calibration));
  $('calibrationResult').className='result';$('calibrationResult').textContent=c.calibration?'Привязка готова. Включите «Проверить кликом» и нажмите на другую точку пола в кадре.':`Кадр: ${c.video_points.length}/4 · Схема: ${c.plan_points.length}/4. Номера должны соответствовать.`;
}
function invalidate() { const c=camera();if(c)c.calibration=null;testPoint=null;testVideoPoint=null;updatePairs();renderCameras();drawFrame();drawPlan();changed(); }
async function selectCamera(id) {activePair=0;pairSide='video';testVideoPoint=null;Object.assign(frameZoom,{zoom:1,panX:0,panY:0});Object.assign(planZoom,{zoom:1,panX:0,panY:0});selected=id;selectedWall=null;testPoint=null;pick=true;$('pickMode').classList.add('active');$('testMode').classList.remove('active');const c=camera();$('cameraPanel').hidden=!c;if(!c)return;
  $('selectedTitle').textContent=c.name;$('cameraName').value=c.name;$('cameraHeight').value=c.height_m;$('cameraYaw').value=c.yaw_deg;$('cameraFov').value=c.fov_deg;$('cameraRange').value=c.range_m;$('cameraOffset').value=c.time_offset_s||0;renderLinks();
  refreshVideoSelect();updatePairs();renderCameras();drawPlan();image=null;drawFrame();if(c.video_id)await loadVideo(c.video_id,false);
}
function refreshVideoSelect() { const current=camera()?.video_id||'';$('videoSelect').replaceChildren();const empty=document.createElement('option');empty.value='';empty.textContent='Выберите видео из data/raw';$('videoSelect').append(empty);for(const v of videos){const o=document.createElement('option');o.value=v.id;o.textContent=v.name;$('videoSelect').append(o);}$('videoSelect').value=current;if(current&&!videos.some(v=>v.id===current))toast('Источник камеры не найден. Выберите видео заново.'); }
async function loadVideo(id, reset) {const c=camera();if(!c)return;const cid=c.id, request=++videoRequest;try{const meta=await api('/api/video?id='+encodeURIComponent(id));if(selected!==cid||request!==videoRequest)return;c.video_id=id;c.video_meta=meta;if(reset){c.video_points=[];c.plan_points=[];c.calibration=null;c.frame_index=0;testPoint=null;}$('frameSlider').max=Math.max(0,meta.frame_count-1);$('frameSlider').value=c.frame_index||0;$('frameLabel').textContent=c.frame_index||0;updatePairs();changed();await loadFrame();}catch(e){toast(e.message);} }
async function loadFrame() { const c=camera();if(!c?.video_id)return;const token=++frameRequest,cid=c.id,index=Number($('frameSlider').value);c.frame_index=index;$('frameLabel').textContent=index;const img=new Image();img.onload=()=>{if(token!==frameRequest||selected!==cid)return;image=img;imageCamera=cid;drawFrame();changed();};img.onerror=()=>toast('Не удалось прочитать кадр. Выберите другой.');img.src=`/api/frame?id=${encodeURIComponent(c.video_id)}&frame=${index}`; }
function distanceToSegment(p,a,b){const dx=b[0]-a[0],dy=b[1]-a[1],len=dx*dx+dy*dy;const t=len?Math.max(0,Math.min(1,((p[0]-a[0])*dx+(p[1]-a[1])*dy)/len)):0;return Math.hypot(p[0]-a[0]-t*dx,p[1]-a[1]-t*dy);}
plan.addEventListener('pointerdown',e=>{const p=pixelToProject(e,mode==='wall'||mode==='camera');if(!p)return;
 if(mode==='wall'){if(!draft){draft=p;pointer=p;}else{if(Math.hypot(p[0]-draft[0],p[1]-draft[1])<.05)return;state.walls.push({id:escapeId('wall'),a:draft,b:p});draft=null;changed();}drawPlan();return;}
 if(mode==='camera'){let num=1;while(state.cameras.some(c=>c.name===`Камера ${num}`))num++;const c={id:escapeId('cam'),name:`Камера ${num}`,position:p,height_m:2.5,yaw_deg:0,fov_deg:60,range_m:5,time_offset_s:0,overlap_with:[],transition_to:[],video_id:'',frame_index:0,video_points:[],plan_points:[],calibration:null};state.cameras.push(c);selectCamera(c.id);setMode('select');changed();return;}
 if(mode==='calibrate'){if(!camera())return toast('Сначала выберите камеру');placePair('plan',p);return;}
 const hit=state.cameras.find(c=>Math.hypot(c.position[0]-p[0],c.position[1]-p[1])*view.scale<18);if(hit){selectCamera(hit.id);dragging=true;plan.setPointerCapture(e.pointerId);}else{const hitWall=state.walls.find(w=>distanceToSegment(p,w.a,w.b)*view.scale<8);selectedWall=hitWall?.id||null;drawPlan();}
});
plan.addEventListener('pointermove',e=>{const p=pixelToProject(e,mode==='wall'||dragging);if(p){$('cursorCoords').textContent=`x ${p[0].toFixed(2)} · y ${p[1].toFixed(2)} ${unit()}`;pointer=p;if(dragging&&camera()){camera().position=p;drawPlan();}else if(draft)drawPlan();}});
plan.addEventListener('pointerup',()=>{if(dragging){dragging=false;changed();}});plan.addEventListener('pointercancel',()=>{dragging=false;});
frame.addEventListener('click',e=>{if(e.altKey||frame._didPan)return;const c=camera();if(!c||!image||imageCamera!==selected)return;const p=sourcePoint(frame,frameView,e,image.naturalWidth,image.naturalHeight);if(!p)return;
 if(pick){placePair('video',p);return;}if(!c.calibration)return toast('Сначала задайте четыре пары');const h=c.calibration.matrix,d=h[2][0]*p[0]+h[2][1]*p[1]+h[2][2];if(Math.abs(d)<1e-10)return toast('Проекция близка к горизонту');const q=[(h[0][0]*p[0]+h[0][1]*p[1]+h[0][2])/d,(h[1][0]*p[0]+h[1][1]*p[1]+h[1][2])/d];if(!q.every(Number.isFinite))return toast('Проекция не определена');testPoint=q;testVideoPoint=p;drawPlan();drawFrame();$('calibrationResult').textContent=`Проверочная точка: ${q[0].toFixed(2)}, ${q[1].toFixed(2)} ${unit()}`+(q[0]<0||q[1]<0||q[0]>state.room.width_m||q[1]>state.room.height_m?' Вне схемы: проверьте пары.':' Зелёные отметки обозначают одно место.');});
calibrationPlan.addEventListener('click',e=>{if(e.altKey||calibrationPlan._didPan||!pick)return;const p=sourcePoint(calibrationPlan,calibrationView,e,state.room.width_m,state.room.height_m);if(p)placePair('plan',p);});
for(const b of document.querySelectorAll('[data-mode]'))b.onclick=()=>setMode(b.dataset.mode);
$('undoWall').onclick=()=>{if(draft){draft=null;drawPlan();return;}state.walls.pop();drawPlan();changed();};
$('deleteSelected').onclick=()=>{if(selectedWall){state.walls=state.walls.filter(w=>w.id!==selectedWall);selectedWall=null;}else if(selected){state.cameras=state.cameras.filter(c=>c.id!==selected);for(const c of state.cameras){c.overlap_with=(c.overlap_with||[]).filter(id=>id!==selected);c.transition_to=(c.transition_to||[]).filter(id=>id!==selected);}selected=null;$('cameraPanel').hidden=true;testPoint=null;}else return toast('Выберите камеру или стену');renderCameras();drawPlan();changed();};
$('resizeRoom').onclick=()=>{const width=Number($('roomWidth').value),height=Number($('roomHeight').value);if(!Number.isFinite(width)||!Number.isFinite(height)||width<1||height<1||width>1000||height>1000)return toast('Размеры схемы: от 1 до 1000');const sx=width/state.room.width_m,sy=height/state.room.height_m,scale=p=>[p[0]*sx,p[1]*sy];
 for(const wall of state.walls){wall.a=scale(wall.a);wall.b=scale(wall.b);}for(const c of state.cameras){c.position=scale(c.position);c.plan_points=c.plan_points.map(scale);c.range_m=Math.max(.1,Math.min(1000,c.range_m*Math.min(sx,sy)));if(c.calibration){c.calibration.matrix[0]=c.calibration.matrix[0].map(v=>v*sx);c.calibration.matrix[1]=c.calibration.matrix[1].map(v=>v*sy);c.calibration.plan_points=c.plan_points;}}
 if(testPoint)testPoint=scale(testPoint);if(state.processing)state.processing.fusion_distance_m=Math.max(.1,Math.min(50,state.processing.fusion_distance_m*Math.min(sx,sy)));state.room={width_m:width,height_m:height};syncRoom();if(camera())updatePairs();drawPlan();changed();};
for(const [field,other] of [['roomWidth','roomHeight'],['roomHeight','roomWidth']])$(field).oninput=()=>{if(!planImage||!$('keepPlanAspect').checked)return;const value=Number($(field).value);if(value>0)$(other).value=field==='roomWidth'?value*planImage.naturalHeight/planImage.naturalWidth:value*planImage.naturalWidth/planImage.naturalHeight;};
for(const [input,key] of [['cameraName','name'],['cameraHeight','height_m'],['cameraYaw','yaw_deg'],['cameraFov','fov_deg'],['cameraRange','range_m'],['cameraOffset','time_offset_s']])$(input).onchange=()=>{const c=camera();if(!c)return;const el=$(input);if(!el.reportValidity())return;c[key]=key==='name'?el.value:Number(el.value);$('selectedTitle').textContent=c.name;renderCameras();drawPlan();changed();};
$('videoSelect').onchange=()=>{if($('videoSelect').value)loadVideo($('videoSelect').value,true);else{const c=camera();if(c){c.video_id='';c.video_meta=null;image=null;c.video_points=[];c.plan_points=[];invalidate();}}};
$('frameSlider').oninput=()=>{$('frameLabel').textContent=$('frameSlider').value;};$('frameSlider').onchange=loadFrame;$('loadFrame').onclick=loadFrame;
$('uploadBtn').onclick=()=>$('videoFile').click();$('videoFile').onchange=async()=>{const file=$('videoFile').files[0],cid=selected;if(!file)return;$('uploadBtn').disabled=true;toast('Копируем видео в локальный проект…');try{const r=await api('/api/upload?name='+encodeURIComponent(file.name),{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file});videos=await api('/api/videos');refreshVideoSelect();if(selected===cid)await loadVideo(r.id,true);toast('Видео добавлено');}catch(e){toast(e.message);}finally{$('uploadBtn').disabled=false;$('videoFile').value='';}};
function removePair(index){const c=camera();if(!c)return;pick=true;$('pickMode').classList.add('active');$('testMode').classList.remove('active');c.video_points.splice(index,1);c.plan_points.splice(index,1);activePair=Math.min(index,Math.min(c.video_points.length,c.plan_points.length),3);pairSide='video';invalidate();}
$('undoVideo').onclick=()=>{if(camera())removePair(Math.max(camera().video_points.length,camera().plan_points.length)-1);};$('undoPlan').onclick=()=>removePair(activePair);$('resetPoints').onclick=()=>{const c=camera();if(c){c.video_points=[];c.plan_points=[];activePair=0;pick=true;$('pickMode').classList.add('active');$('testMode').classList.remove('active');pairSide='video';invalidate();}};
$('pickMode').onclick=()=>{pick=true;$('pickMode').classList.add('active');$('testMode').classList.remove('active');pairInstruction();};$('testMode').onclick=()=>{if(!camera()?.calibration)return toast('Сначала рассчитайте привязку');pick=false;pairInstruction();$('testMode').classList.add('active');$('pickMode').classList.remove('active');toast('Нажмите на точку пола в кадре');};
async function calculateCalibration(c=camera()){if(!c)return;const cid=c.id,signature=JSON.stringify([c.video_points,c.plan_points]);try{const result=await post('/api/calibrate',{video_points:c.video_points,plan_points:c.plan_points});if(!state.cameras.includes(c)||signature!==JSON.stringify([c.video_points,c.plan_points]))return;c.calibration=result;if(selected===cid){updatePairs();drawFrame();}renderCameras();drawPlan();changed();toast('Привязка рассчитана. Проверьте её на другой точке пола.');}catch(e){$('calibrationResult').className='result error';$('calibrationResult').textContent=e.message;}};
$('calibrateBtn').onclick=()=>calculateCalibration(camera());
$('exportBtn').onclick=()=>{const blob=new Blob([JSON.stringify(state,null,2)],{type:'application/json'});const link=document.createElement('a');link.href=URL.createObjectURL(blob);link.download='office-project.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);};
$('importBtn').onclick=()=>$('importFile').click();$('importFile').onchange=async()=>{const file=$('importFile').files[0];if(!file)return;try{const next=JSON.parse(await file.text());clearTimeout(saveTimer);await saveQueue;await post('/api/project',next);state=await api('/api/project');selected=null;selectedWall=null;testPoint=null;draft=null;image=null;dirty=false;$('cameraPanel').hidden=true;syncRoom();renderCameras();await loadPlanBackground();drawPlan();$('saveStatus').textContent='Проект загружен';toast('Проект открыт');}catch(e){toast(e.message);}finally{$('importFile').value='';}};
window.addEventListener('keydown',e=>{if(e.key==='Escape'){draft=null;dragging=false;setMode('select');}});
function syncRoom(){$('roomWidth').value=state.room.width_m;$('roomHeight').value=state.room.height_m;$('coordinateUnits').value=state.coordinate_units||'meters';$('widthLabel').textContent='Ширина, '+unit();$('heightLabel').textContent='Глубина, '+unit();}
new ResizeObserver(()=>{drawPlan();if(image)drawFrame();}).observe(plan.parentElement);
(async()=>{try{[state,videos,planAssets]=await Promise.all([api('/api/project'),api('/api/videos'),api('/api/plans')]);syncRoom();renderPlanAssets();await loadPlanBackground();renderCameras();drawPlan();setMode('select');$('saveStatus').textContent='Сохранено на компьютере';}catch(e){$('saveStatus').textContent='Ошибка загрузки';toast(e.message);}})();

function renderLinks(){const current=camera();if(!current)return;for(const [container,key] of [['overlapLinks','overlap_with'],['transitionLinks','transition_to']]){$(container).replaceChildren();let count=0;for(const other of state.cameras){if(other.id===current.id)continue;count++;const label=document.createElement('label'),input=document.createElement('input');input.type='checkbox';input.checked=(current[key]||[]).includes(other.id);input.onchange=()=>{current[key]=(current[key]||[]).filter(id=>id!==other.id);if(input.checked)current[key].push(other.id);if(key==='overlap_with'){other[key]=(other[key]||[]).filter(id=>id!==current.id);if(input.checked)other[key].push(current.id);}changed();};label.append(input,document.createTextNode(other.name));$(container).append(label);}if(!count)$(container).textContent='Добавьте ещё одну камеру';}}
$('sequenceVideos').onclick=async()=>{if(!state.cameras.length)return toast('Сначала добавьте камеры');const gap=Number($('recordingGap').value);if(!Number.isFinite(gap)||gap<0||gap>300)return toast('Пауза: от 0 до 300 секунд');let offset=0;const offsets=[];try{for(const c of state.cameras){if(!c.video_id)throw Error('Для каждой камеры нужно выбрать видео');const meta=await api('/api/video?id='+encodeURIComponent(c.video_id));offsets.push(offset);offset+=meta.frame_count/meta.fps+gap;}state.cameras.forEach((c,i)=>c.time_offset_s=offsets[i]);if(camera())$('cameraOffset').value=camera().time_offset_s;changed();toast('Ролики расставлены по порядку камер с указанными паузами. Связи камер задаются отдельно.');}catch(e){toast(e.message);}};
$('simultaneousVideos').onclick=()=>{state.cameras.forEach(c=>c.time_offset_s=0);if(camera())$('cameraOffset').value=0;changed();toast('Начало всех записей: 0 секунд');};
$('reviewNav').onclick=async e=>{e.preventDefault();clearTimeout(saveTimer);await save();if(!dirty)location.href='/review';};

function renderPlanAssets(){const select=$('planImageSelect');select.replaceChildren(new Option('Без изображения',''));for(const p of planAssets)select.append(new Option(p.name,p.id));select.value=state.plan_background?.image_id||'';$('planOpacity').value=state.plan_background?.opacity??.85;}
async function loadPlanBackground(){const token=++planImageRequest;planImage=null;drawPlan();const id=state.plan_background?.image_id;if(!id)return;await new Promise(resolve=>{const img=new Image();img.onload=()=>{if(token===planImageRequest){planImage=img;drawPlan();}resolve();};img.onerror=()=>{toast('Подложка не найдена. Выберите изображение плана заново.');resolve();};img.src='/api/plan-image?id='+encodeURIComponent(id);});}
async function choosePlan(id){const asset=planAssets.find(p=>p.id===id);if(state.cameras.some(c=>c.calibration||c.plan_points.length)&&!confirm('Смена подложки сбросит точки плана и привязки всех камер. Продолжить?')){renderPlanAssets();return;}
 for(const c of state.cameras){c.plan_points=[];c.calibration=null;}state.plan_background=asset?{image_id:id,opacity:.85}:null;
 if(asset){const old=state.room,next={width_m:old.width_m,height_m:old.width_m*asset.height/asset.width};const sy=next.height_m/old.height_m;for(const c of state.cameras)c.position[1]*=sy;for(const wall of state.walls){wall.a[1]*=sy;wall.b[1]*=sy;}state.room=next;state.coordinate_units='relative';}
 activePair=0;pairSide='plan';testPoint=null;testVideoPoint=null;syncRoom();await loadPlanBackground();renderCameras();if(camera())updatePairs();changed();}
$('planImageSelect').onchange=()=>choosePlan($('planImageSelect').value);$('planOpacity').oninput=()=>{if(state.plan_background){state.plan_background.opacity=Number($('planOpacity').value);drawPlan();changed();}};$('coordinateUnits').onchange=()=>{state.coordinate_units=$('coordinateUnits').value;syncRoom();drawPlan();changed();};
$('uploadPlan').onclick=()=>$('planFile').click();$('planFile').onchange=async()=>{const file=$('planFile').files[0];if(!file)return;try{const result=await api('/api/plan-upload',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file});planAssets=await api('/api/plans');renderPlanAssets();$('planImageSelect').value=result.id;await choosePlan(result.id);}catch(e){toast(e.message);}finally{$('planFile').value='';}};
function attachViewport(canvas,zoom,redraw,prefix){
 let drag=null;canvas.addEventListener('pointerdown',e=>{canvas._didPan=false;if(e.altKey||e.button===2){drag=[e.clientX,e.clientY,zoom.panX,zoom.panY];canvas.setPointerCapture(e.pointerId);e.preventDefault();}});
 canvas.addEventListener('pointermove',e=>{if(drag){zoom.panX=drag[2]+e.clientX-drag[0];zoom.panY=drag[3]+e.clientY-drag[1];canvas._didPan=true;redraw();}});canvas.addEventListener('pointerup',()=>{drag=null;});canvas.addEventListener('pointercancel',()=>{drag=null;});canvas.addEventListener('contextmenu',e=>e.preventDefault());
 function rescale(factor){const next=Math.max(1,Math.min(10,zoom.zoom*factor));const ratio=next/zoom.zoom;zoom.zoom=next;zoom.panX*=ratio;zoom.panY*=ratio;redraw();}
 canvas.addEventListener('wheel',e=>{e.preventDefault();rescale(e.deltaY<0?1.12:1/1.12);},{passive:false});$(prefix+'ZoomIn').onclick=()=>rescale(1.5);$(prefix+'ZoomOut').onclick=()=>rescale(1/1.5);$(prefix+'Fit').onclick=()=>{Object.assign(zoom,{zoom:1,panX:0,panY:0});redraw();};new ResizeObserver(redraw).observe(canvas.parentElement);
}
attachViewport(frame,frameZoom,drawFrame,'frame');attachViewport(calibrationPlan,planZoom,drawCalibrationPlan,'plan');
