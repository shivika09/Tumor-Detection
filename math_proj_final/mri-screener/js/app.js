const $=id=>document.getElementById(id);
const MODEL_JSON=window.MODEL_DATA;
const M=makeModel(MODEL_JSON);
const mt=MODEL_JSON.metrics;
$('kk').textContent=M.k;$('nte').textContent=mt.n;
$('sAcc').textContent=(mt.acc*100).toFixed(1)+'%';
$('sRec').textContent=(mt.tp/(mt.tp+mt.fn)*100).toFixed(1)+'%';
$('sPre').textContent=(mt.tp/(mt.tp+mt.fp)*100).toFixed(1)+'%';
$('tn').textContent=mt.tn;$('fp').textContent=mt.fp;$('fn').textContent=mt.fn;$('tp').textContent=mt.tp;

let counts={t:0,c:0,n:0};
function refresh(){
  const any=counts.n>0;
  $('sum').hidden=!any;$('empty').hidden=any;
  $('nAll').textContent=counts.n;$('nT').textContent=counts.t;$('nC').textContent=counts.c;
}

async function decode(blob){
  let bmp;
  try{bmp=await createImageBitmap(blob,{imageOrientation:'none'});}
  catch(e){
    const url=URL.createObjectURL(blob);
    bmp=await new Promise((res,rej)=>{const im=new Image();im.onload=()=>res(im);im.onerror=()=>rej(new Error('decode'));im.src=url;});
  }
  const w=bmp.width||bmp.naturalWidth,h=bmp.height||bmp.naturalHeight;
  const cv=document.createElement('canvas');cv.width=w;cv.height=h;
  const cx=cv.getContext('2d',{willReadFrequently:true});cx.drawImage(bmp,0,0);
  return {data:cx.getImageData(0,0,w,h).data,w,h};
}

function card(name,blob,truth){
  const el=document.createElement('article');el.className='card';
  const url=URL.createObjectURL(blob);
  el.innerHTML='<div class="views"><div class="main"><img alt="" src="'+url+'"></div><div class="px"><canvas width="64" height="64"></canvas><span>model view 64×64</span></div></div><div class="body"><div class="name"></div><div class="status conf">Analysing…</div></div>';
  el.querySelector('.name').textContent=name;el.querySelector('img').alt='MRI scan '+name;
  $('grid').appendChild(el);
  return el;
}

async function classify(name,blob,truth){
  const el=card(name,blob,truth);
  try{
    const im=await decode(blob);
    const pix=preprocess(im.data,im.w,im.h);
    const s=decide(M,pix);
    // model view
    const c=el.querySelector('canvas'),cx=c.getContext('2d'),id=cx.createImageData(64,64);
    for(let i=0;i<4096;i++){id.data[i*4]=id.data[i*4+1]=id.data[i*4+2]=pix[i];id.data[i*4+3]=255;}
    cx.putImageData(id,0,0);
    const tumor=s>0,low=Math.abs(s)<0.3;
    counts.n++;tumor?counts.t++:counts.c++;refresh();
    const pos=Math.max(2,Math.min(98,50+s/2.5*50));
    let t='';
    if(truth!==undefined){const ok=(truth===1)===tumor;t='<div class="truth">Dataset label: <b>'+(truth===1?'tumor':'no tumor')+'</b> · model is <b>'+(ok?'correct':'wrong')+'</b></div>';}
    el.querySelector('.body').innerHTML='<div class="name"></div><div class="verdict"><span class="pill '+(tumor?'t':'c')+'">'+(tumor?'Tumor detected':'No tumor')+'</span><span class="conf">'+(low?'low confidence':Math.abs(s)>1?'high confidence':'moderate')+'</span></div><div><div class="meter"><i style="left:'+pos+'%"></i></div></div><div class="scale"><span>no tumor</span><span>tumor</span></div><div class="score">SVM score '+(s>=0?'+':'')+s.toFixed(2)+'</div>'+t;
    el.querySelector('.name').textContent=name;
  }catch(e){
    el.querySelector('.views').remove();
    el.querySelector('.body').innerHTML='<div class="err">Could not read <b></b> as an image. Use JPG, PNG or WebP (DICOM files need exporting to an image first).</div>';
    el.querySelector('.err b').textContent=name;
  }
}

async function addFiles(files){for(const f of files){await classify(f.name,f);}}

function b64blob(b64){const bin=atob(b64),u=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)u[i]=bin.charCodeAt(i);return new Blob([u],{type:'image/jpeg'});}
async function loadSamples(){
  for(const s of window.SAMPLES_DATA) await classify(s.name,b64blob(s.b64),s.label);
}
function clearAll(){$('grid').innerHTML='';counts={t:0,c:0,n:0};refresh();}

$('file').addEventListener('change',e=>{addFiles([...e.target.files]);e.target.value='';});
$('pick').addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();$('file').click();}});
$('samples').addEventListener('click',loadSamples);
$('clear').addEventListener('click',clearAll);
const drop=$('drop');
['dragenter','dragover'].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.add('over');}));
['dragleave','drop'].forEach(t=>drop.addEventListener(t,e=>{e.preventDefault();drop.classList.remove('over');}));
drop.addEventListener('drop',e=>addFiles([...e.dataTransfer.files].filter(f=>f.type.startsWith('image/'))));
refresh();loadSamples();
