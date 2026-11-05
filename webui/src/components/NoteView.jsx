import React, { useEffect, useMemo, useRef, useState } from "react";

export const visible = (mode, label) => mode === "shown" || (mode === "after" && !!label);

export default function NoteView({ adm, item, label, mode }) {
  const [q, setQ] = useState("");
  const box = useRef(null);
  const show = item && visible(mode, label);
  const graphEv = useMemo(() => new Set(show ? item.graph.note_evidence.map((e) => e.sentence) : []), [item, show]);
  const readerEv = useMemo(() => new Set(show && item.reader ? item.reader.evidence : []), [item, show]);
  const directEv = useMemo(() => new Set(show && item.direct ? item.direct.evidence : []), [item, show]);

  useEffect(() => {
    const first = box.current?.querySelector(".ev-graph, .ev-reader, .ev-direct");
    if (first) first.scrollIntoView({ block: "center", behavior: "smooth" });
  }, [item?.code, adm?.key, show]);

  if (!adm) return <main className="note card muted pad">Loading…</main>;
  let last = null;
  const needle = q.trim().toLowerCase();
  return (
    <main className="note card" ref={box} aria-label="Discharge summary">
      <div className="note-head">
        <strong>Admission {adm.key}</strong>
        <input placeholder="Search this note" value={q} onChange={(e) => setQ(e.target.value)} />
        {show && (
          <span className="legend small">
            <span className="sw ev-direct" /> direct <span className="sw ev-graph" /> graph {item.reader && (<><span className="sw ev-reader" /> reader</>)}
          </span>
        )}
      </div>
      {adm.sentences.map((s) => {
        const head = s.section !== last ? <h3 key={`h${s.id}`}>{s.section}</h3> : null;
        last = s.section;
        if (needle && !s.text.toLowerCase().includes(needle)) return head;
        const cls = [graphEv.has(s.id) && "ev-graph", readerEv.has(s.id) && "ev-reader", directEv.has(s.id) && "ev-direct"].filter(Boolean).join(" ");
        return (
          <React.Fragment key={s.id}>
            {head}
            <p id={`s-${s.id}`} className={`s ${cls}`}>{s.text}</p>
          </React.Fragment>
        );
      })}
    </main>
  );
}
