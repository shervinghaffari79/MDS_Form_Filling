"""Build a blinded, offline review page for measuring accuracy on MIMIC (output contains note text: it stays in /home/user/mimic).

Rules set before any labelling:
Admissions: the 20 that every method was run on (reader baseline and graph pilot).
Candidates per admission: every MDS 2.0 item filled by at least one method (graph notes, graph notes+v, graph records, full graph,
  reader), plus 5 items no method filled (seed 41), so misses shared by all methods can be estimated.
Blinding: the page never says which method filled an item; candidates are shuffled per admission (seed 41). Note evidence shown
  is the union of sentences cited by any method, highlighted in the full note; record evidence is listed with source and date.
Question per candidate: does this MDS 2.0 item apply to the patient, judged from the note and records against the item wording and
  look-back period? Present / Absent / Can't tell, optional comment. Labels export as JSON keyed by note key and item code only.
"""
import html
import json
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from mds_pilot.reader import PLAIN_SCHEMA, SYSTEM, build_prompt  # noqa: E402
from sweetspot import admission_graph as A  # noqa: E402
from sweetspot.llm import LLM  # noqa: E402

PRIVATE = Path("/home/user/mimic/pilot_private")
WORK = Path("/home/user/mimic/sweetspot")
OUT_DIR = Path("/home/user/mimic/review")
N_NEG = 5
METHODS = ("notes", "notes+v", "records", "graph", "reader")


def reader_decisions(keys, sents):
    """Reader answers and citations for the 20 admissions, read from the reply cache (same requests as the baseline run)."""
    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cpu")
    items = A.ITEMS["items"]
    q = enc.encode(["Represent this sentence for searching relevant passages: " + i["wording"] for i in items], normalize_embeddings=True)
    llm = LLM("p2_reader")
    out = {}
    for k in keys:
        rows = sents[sents["note_key"] == k].sort_values("sent_id").reset_index(drop=True)
        e = enc.encode(rows["text"].tolist(), normalize_embeddings=True, batch_size=128)
        sim = e @ q.T
        reqs = []
        for j, it in enumerate(items):
            top = np.sort(np.argsort(-sim[:, j])[:20])
            block = [(int(rows.loc[t, "sent_id"]), rows.loc[t, "section"], rows.loc[t, "text"]) for t in top]
            reqs.append({"name": "plain", "system": SYSTEM, "prompt": build_prompt("plain", it["wording"], block), "schema": PLAIN_SCHEMA})
        for it, r in zip(items, llm.run(reqs)):
            if r and r["parsed"]["answer"] == "yes":
                out[(k, it["code"])] = sorted(set(r["parsed"]["evidence"]))
    return out


def main():
    state = pickle.loads((WORK / "p2_graphs.pkl").read_bytes())
    arms = state["arms"]
    sents = pd.read_parquet(PRIVATE / "proxy_sentences.parquet")
    keys = sorted(arms["notes+v"])[:20]
    reader = reader_decisions(keys, sents)
    items = {i["code"]: i for i in A.ITEMS["items"]}
    rng = random.Random(41)
    data, truth = [], []
    for k in keys:
        rows = sents[sents["note_key"] == k].sort_values("sent_id")
        filled = {m: set(arms[m][k]) if m != "reader" else {c for (kk, c) in reader if kk == k} for m in METHODS}
        cand = sorted(set().union(*filled.values()))
        neg = rng.sample([c for c in items if c not in cand], N_NEG)
        allc = cand + neg
        rng.shuffle(allc)
        entries = []
        for c in allc:
            note_ev, rec_ev = set(), []
            for m in ("notes", "notes+v", "graph"):
                for p in arms[m][k].get(c, []):
                    if p["source"] == "note":
                        note_ev.add(int(p["sentence"]))
                    elif m == "graph":
                        rec_ev.append(f"{p['source']}: {p['concept']}" + (f" ({p['time'][:10]})" if p.get("time") not in (None, "None") else ""))
            note_ev |= set(reader.get((k, c), []))
            entries.append({"code": c, "label": items[c]["wording"], "look_back": A.window_days(items[c]),
                            "sentences": sorted(note_ev), "records": sorted(set(rec_ev))})
            truth.append({"note_key": k, "item": c, "sampled_negative": c in neg, **{m: c in filled[m] for m in METHODS}})
        data.append({"key": k, "sentences": [{"id": int(s), "section": sec, "text": t}
                                             for s, sec, t in zip(rows["sent_id"], rows["section"], rows["text"])],
                     "items": entries})
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "method_outputs.json").write_text(json.dumps(truth))  # kept apart from the page so reviewers stay blind
    page = TEMPLATE.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    (OUT_DIR / "mds_review.html").write_text(page)
    n = sum(len(d["items"]) for d in data)
    print(json.dumps({"admissions": len(data), "candidates": n, "per_admission": round(n / len(data), 1),
                      "sampled_negatives": N_NEG * len(data), "page": str(OUT_DIR / "mds_review.html")}))


TEMPLATE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'">
<title>MDS 2.0 review</title><style>
:root{--bg:#f6f7f8;--card:#fff;--ink:#1b1f24;--muted:#5b6470;--line:#d5d9de;--hl:#fff3c4;--hlb:#c99a00;--acc:#1668c7;--ok:#23803d;--no:#b42318}
@media(prefers-color-scheme:dark){:root{--bg:#14171a;--card:#1d2126;--ink:#e6e8eb;--muted:#9aa3ad;--line:#33393f;--hl:#4a3f12;--hlb:#e0b830;--acc:#5aa2f0;--ok:#4cc36b;--no:#f2766b}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 -apple-system,Segoe UI,sans-serif;color:var(--ink);background:var(--bg)}
header{position:sticky;top:0;z-index:2;background:var(--card);border-bottom:1px solid var(--line);padding:8px 16px;display:flex;gap:10px;flex-wrap:wrap;align-items:center}
main{display:grid;grid-template-columns:minmax(0,1fr) 420px;gap:12px;padding:12px 16px;max-width:1500px;margin:auto}
#note,#panel{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:0}
#panel{position:sticky;top:60px;align-self:start;max-height:calc(100vh - 75px);overflow:auto}
h2{font-size:12px;letter-spacing:.04em;text-transform:uppercase;color:var(--muted);margin:14px 0 4px}
.s{padding:2px 6px;border-left:3px solid transparent;overflow-wrap:anywhere}.s.ev{background:var(--hl);border-left-color:var(--hlb)}
.item{border:1px solid var(--line);border-radius:6px;padding:8px;margin:8px 0;cursor:pointer}.item.cur{border-color:var(--acc);box-shadow:0 0 0 1px var(--acc)}
.item .t{font-weight:600}.muted{color:var(--muted);font-size:12px}.rec{font-size:12px;margin:2px 0 0 10px}
.btns{display:flex;gap:6px;margin-top:6px;flex-wrap:wrap}button,input,select{font:inherit;padding:4px 9px;border:1px solid var(--line);border-radius:5px;background:var(--card);color:var(--ink)}
button{cursor:pointer}button.p.on{background:var(--ok);border-color:var(--ok);color:#fff}button.a.on{background:var(--no);border-color:var(--no);color:#fff}
button.u.on{background:var(--muted);border-color:var(--muted);color:#fff}input.c{width:100%;margin-top:6px}#progress{font-weight:600}
@media(max-width:900px){main{grid-template-columns:1fr}#panel{position:static;max-height:none}}
</style></head><body><header>
<label>Reviewer <input id="who" size="8"></label><button id="prev">Previous admission</button><span id="pos"></span>
<button id="next">Next admission</button><input id="find" placeholder="Search this note" size="18"><span id="progress"></span>
<button id="exp">Export labels</button><label class="muted">Import <input type="file" id="imp" accept=".json"></label></header>
<main><div id="note"></div><aside id="panel"></aside></main><script>
const DATA=__DATA__;const KEY='mds20_review_v1';let st={who:'',labels:{}};
try{st=Object.assign(st,JSON.parse(localStorage.getItem(KEY)||'{}'))}catch(e){}
let cur=0,curItem=0;const $=i=>document.getElementById(i);
function save(){try{localStorage.setItem(KEY,JSON.stringify(st))}catch(e){alert('Autosave failed. Export your labels now.')}}
function key(a,c){return a+'|'+c}
function renderNote(){const a=DATA[cur],box=$('note'),it=a.items[curItem],ev=new Set(it?it.sentences:[]),q=$('find').value.toLowerCase();box.textContent='';let last=null,first=null;
 a.sentences.forEach(s=>{if(s.section!==last){const h=document.createElement('h2');h.textContent=s.section;box.appendChild(h);last=s.section}
  if(q&&!s.text.toLowerCase().includes(q))return;const d=document.createElement('div');d.className='s'+(ev.has(s.id)?' ev':'');d.textContent=s.text;
  if(ev.has(s.id)&&!first)first=d;box.appendChild(d)});return first}
function renderPanel(){const a=DATA[cur],p=$('panel');p.textContent='';
 const m=document.createElement('p');m.className='muted';m.textContent='For each item: does it apply to this patient, judged from the note and records against the MDS 2.0 wording and look-back period? Highlighted sentences are evidence some method used; check the whole note.';p.appendChild(m);
 a.items.forEach((it,i)=>{const d=document.createElement('div');d.className='item'+(i===curItem?' cur':'');const lab=st.labels[key(a.key,it.code)]||{};
  const t=document.createElement('div');t.className='t';t.textContent=it.code+'  '+it.label;d.appendChild(t);
  const w=document.createElement('div');w.className='muted';w.textContent=(it.look_back?'Look-back: last '+it.look_back+' days. ':'Look-back: active during the stay. ')+it.sentences.length+' highlighted sentence(s), '+it.records.length+' record(s).';d.appendChild(w);
  it.records.slice(0,12).forEach(r=>{const x=document.createElement('div');x.className='rec';x.textContent=r;d.appendChild(x)});
  if(it.records.length>12){const x=document.createElement('div');x.className='rec muted';x.textContent='+'+(it.records.length-12)+' more records';d.appendChild(x)}
  const b=document.createElement('div');b.className='btns';[['present','Present','p'],['absent','Absent','a'],['unclear',"Can't tell",'u']].forEach(([v,txt,c])=>{
   const bt=document.createElement('button');bt.textContent=txt;bt.className=c+(lab.v===v?' on':'');bt.onclick=e=>{e.stopPropagation();
    st.labels[key(a.key,it.code)]=Object.assign(lab,{v});save();if(i<a.items.length-1)curItem=i+1;render()};b.appendChild(bt)});d.appendChild(b);
  const c=document.createElement('input');c.className='c';c.placeholder='Comment (optional)';c.value=lab.c||'';c.onclick=e=>e.stopPropagation();
  c.onchange=()=>{st.labels[key(a.key,it.code)]=Object.assign(st.labels[key(a.key,it.code)]||{},{c:c.value});save()};d.appendChild(c);
  d.onclick=()=>{curItem=i;render(true)};p.appendChild(d)})}
function render(scroll){const a=DATA[cur];$('pos').textContent='Admission '+(cur+1)+' / '+DATA.length;
 const total=DATA.reduce((n,x)=>n+x.items.length,0),done=Object.values(st.labels).filter(l=>l.v).length;$('progress').textContent=done+' / '+total+' labelled';
 const f=renderNote();renderPanel();if(scroll&&f)f.scrollIntoView({block:'center'})}
$('who').value=st.who;$('who').onchange=()=>{st.who=$('who').value.trim();save()};
$('prev').onclick=()=>{if(cur>0){cur--;curItem=0;render();scrollTo(0,0)}};$('next').onclick=()=>{if(cur<DATA.length-1){cur++;curItem=0;render();scrollTo(0,0)}};
$('find').oninput=()=>render();
$('exp').onclick=()=>{if(!st.who){alert('Enter your reviewer initials first.');return}const rows=[];Object.entries(st.labels).forEach(([k,l])=>{if(!l.v)return;const [n,c]=k.split('|');rows.push({note_key:n,item:c,label:l.v,comment:l.c||''})});
 const blob=new Blob([JSON.stringify({reviewer:st.who,form:'MDS 2.0',labels:rows},null,1)],{type:'application/json'});const u=URL.createObjectURL(blob);
 const x=document.createElement('a');x.href=u;x.download='mds_review_labels_'+st.who+'.json';x.click();setTimeout(()=>URL.revokeObjectURL(u),1000)};
$('imp').onchange=e=>{const f=e.target.files[0];if(!f)return;f.text().then(t=>{const j=JSON.parse(t);j.labels.forEach(r=>{st.labels[key(r.note_key,r.item)]={v:r.label,c:r.comment}});if(j.reviewer)st.who=j.reviewer;$('who').value=st.who;save();render()})};
render(true);</script></body></html>"""

if __name__ == "__main__":
    main()
