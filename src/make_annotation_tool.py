"""Build a self-contained, offline labelling page. Output contains note text: keep it local."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from mds_pilot.items import ITEMS

ROOT = Path(__file__).resolve().parents[1]
PRIVATE = ROOT / "pilot_private"
SECTION_NAMES = {"I": "Section I: diagnoses", "N": "Section N: medications", "O": "Section O: treatments"}
STATUSES = [("present", "Present / given"), ("absent", "Explicitly absent"), ("possible", "Possible / uncertain"),
            ("past", "Past or resolved"), ("planned", "Planned or conditional"), ("other", "Another person")]

TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>MDS pilot labelling</title><style>
*{box-sizing:border-box}body{margin:0;font:14px/1.45 -apple-system,Segoe UI,sans-serif;color:#1b1f24;background:#f6f7f8}
header{position:sticky;top:0;z-index:2;background:#fff;border-bottom:1px solid #d5d9de;padding:8px 14px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
main{display:grid;grid-template-columns:minmax(0,1fr) 380px;gap:12px;padding:12px;max-width:1500px;margin:auto}
#note{background:#fff;border:1px solid #d5d9de;border-radius:6px;padding:10px 14px;min-width:0}
#panel{position:sticky;top:60px;align-self:start;background:#fff;border:1px solid #d5d9de;border-radius:6px;padding:10px;max-height:calc(100vh - 75px);overflow:auto}
h2{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:#596370;margin:14px 0 4px}
.s{padding:2px 6px;border-radius:4px;cursor:pointer;border-left:3px solid transparent;overflow-wrap:anywhere}.s:hover{background:#eef2f6}
.s.sel{background:#dcebff;border-left-color:#1668c7}.s.tag{background:#e9f6ec;border-left-color:#23803d}.s.sel.tag{background:#d6ead9}
.chips{font-size:11px;color:#23803d;font-weight:600}button,select,input{font:inherit;padding:4px 8px;border:1px solid #aab2bb;border-radius:4px;background:#fff;color:inherit}
button{cursor:pointer}button.on{background:#1668c7;border-color:#1668c7;color:#fff}.it{display:grid;grid-template-columns:1fr;gap:3px}.it button{text-align:left}
.done{background:#23803d;border-color:#23803d;color:#fff}.muted{color:#596370;font-size:12px}#progress{font-weight:600}
@media(max-width:900px){main{grid-template-columns:1fr}#panel{position:static;max-height:none}}
</style></head><body><header>
<label>Annotator <input id="who" size="10"></label><button id="prev">Previous</button><span id="pos"></span><button id="next">Next</button>
<label>Jump <select id="jump"></select></label><button id="doneBtn">Mark note done</button><span id="progress"></span>
<button id="exportBtn">Export labels</button><label>Import <input type="file" id="imp" accept=".json"></label></header>
<main><div id="note"></div><aside id="panel"></aside></main><script>
const DATA=__DATA__;const KEY='mds_pilot_labels_v1';let st={who:'',tags:{},done:{}};
try{st=Object.assign(st,JSON.parse(localStorage.getItem(KEY)||'{}'))}catch(e){}
let cur=0,sel=null;const $=i=>document.getElementById(i);
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){alert('Could not autosave. Export your labels now.')}}
function k(n,s){return n+'|'+s}
function renderNote(){const n=DATA.notes[cur];const box=$('note');box.textContent='';let last=null;
 n.sentences.forEach(s=>{if(s.section!==last){const h=document.createElement('h2');h.textContent=s.section;box.appendChild(h);last=s.section}
  const d=document.createElement('div');d.className='s';d.id='s'+s.id;const tags=st.tags[k(n.key,s.id)]||{};
  if(Object.keys(tags).length)d.classList.add('tag');if(sel===s.id)d.classList.add('sel');d.textContent=s.text;
  if(Object.keys(tags).length){const c=document.createElement('div');c.className='chips';
   c.textContent=Object.entries(tags).map(([i,v])=>DATA.itemNames[i]+' ('+v+')').join(', ');d.appendChild(c)}
  d.onclick=()=>{const y=scrollY;sel=s.id;render();scrollTo(0,y)};box.appendChild(d)})}
function renderPanel(){const n=DATA.notes[cur],p=$('panel');p.textContent='';
 if(sel===null){const m=document.createElement('p');m.className='muted';m.textContent='Click a sentence, then tag every item it gives evidence for. Leave unrelated sentences untagged.';p.appendChild(m);return}
 const tags=st.tags[k(n.key,sel)]=st.tags[k(n.key,sel)]||{};const lab=document.createElement('label');lab.textContent='Status for next tag ';
 const ss=document.createElement('select');ss.id='status';DATA.statuses.forEach(([v,t])=>{const o=document.createElement('option');o.value=v;o.textContent=t;ss.appendChild(o)});
 ss.value=window._status||'present';ss.onchange=()=>window._status=ss.value;lab.appendChild(ss);p.appendChild(lab);
 DATA.groups.forEach(g=>{const h=document.createElement('h2');h.textContent=g.name;p.appendChild(h);const w=document.createElement('div');w.className='it';
  g.items.forEach(i=>{const b=document.createElement('button');b.textContent=DATA.itemNames[i]+(tags[i]?' ('+tags[i]+')':'');if(tags[i])b.className='on';
   b.onclick=()=>{if(tags[i])delete tags[i];else tags[i]=ss.value;if(!Object.keys(tags).length)delete st.tags[k(n.key,sel)];save();render()};w.appendChild(b)});p.appendChild(w)})}
function render(){const n=DATA.notes[cur];$('pos').textContent=(cur+1)+' / '+DATA.notes.length+(n.overlap?' (overlap note)':'');$('jump').value=cur;
 $('doneBtn').textContent=st.done[n.key]?'Done: click to undo':'Mark note done';$('doneBtn').className=st.done[n.key]?'done':'';
 $('progress').textContent=Object.keys(st.done).length+' done';renderNote();renderPanel()}
function go(i){cur=Math.max(0,Math.min(DATA.notes.length-1,i));sel=null;render();scrollTo(0,0)}
DATA.notes.forEach((n,i)=>{const o=document.createElement('option');o.value=i;o.textContent=(i+1)+' '+n.key;$('jump').appendChild(o)});
$('who').value=st.who||'';$('who').oninput=e=>{st.who=e.target.value;save()};$('prev').onclick=()=>go(cur-1);$('next').onclick=()=>go(cur+1);$('jump').onchange=e=>go(+e.target.value);
$('doneBtn').onclick=()=>{const n=DATA.notes[cur];if(st.done[n.key])delete st.done[n.key];else st.done[n.key]=true;save();render()};
$('exportBtn').onclick=()=>{if(!st.who){alert('Enter your annotator initials first.');return}const tags=[];Object.entries(st.tags).forEach(([a,v])=>{const [n,s]=a.split('|');Object.entries(v).forEach(([i,x])=>tags.push({note_key:n,sent_id:+s,item:i,status:x}))});
 const blob=new Blob([JSON.stringify({annotator:st.who,done:Object.keys(st.done),tags},null,1)],{type:'application/json'});const a=document.createElement('a');a.href=URL.createObjectURL(blob);a.download='labels_'+st.who+'.json';a.click()};
$('imp').onchange=e=>{const f=e.target.files[0];if(!f)return;f.text().then(t=>{const j=JSON.parse(t);st.who=j.annotator;st.tags={};st.done={};j.done.forEach(d=>st.done[d]=true);j.tags.forEach(x=>{const q=k(x.note_key,x.sent_id);st.tags[q]=st.tags[q]||{};st.tags[q][x.item]=x.status});save();$('who').value=st.who;render()})};
render();</script></body></html>"""


def build(sentences, order, overlap, out_path, items=ITEMS):
    grouped = {k: g.sort_values("sent_id") for k, g in sentences.groupby("note_key")}
    data = {
        "statuses": STATUSES,
        "itemNames": {i.id: i.id.split("_", 1)[1].replace("_", " ") for i in items},
        "groups": [{"name": SECTION_NAMES[s], "items": [i.id for i in items if i.section == s]} for s in "INO"],
        "notes": [{"key": key, "overlap": key in overlap,
                   "sentences": [{"id": int(r.sent_id), "section": r.section, "text": r.text}
                                 for r in grouped[key].itertuples()]} for key in order],
    }
    payload = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    out_path.write_text(TEMPLATE.replace("__DATA__", payload), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlap", type=int, default=10)
    args = ap.parse_args()
    sentences = pd.read_parquet(PRIVATE / "pilot_sentences.parquet")
    order = sorted(sentences["note_key"].unique())
    build(sentences, order, set(order[:args.overlap]), PRIVATE / "annotation_tool.html")
    (PRIVATE / "annotations").mkdir(exist_ok=True)
    print(f"Wrote annotation tool with {len(order)} notes; first {args.overlap} are the overlap set.")


if __name__ == "__main__":
    main()
