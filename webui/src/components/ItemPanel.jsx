import React, { useEffect, useRef } from "react";
import { visible } from "./NoteView.jsx";

const LABELS = [
  ["present", "Present", "1"],
  ["absent", "Absent", "2"],
  ["unclear", "Can't tell", "3"],
];
const AGREE_TEXT = { both_yes: "both say yes", graph_only: "graph only: check", direct_only: "direct only: check" };
const SECTION_SHORT = { I: "I · Diagnoses", O: "O · Medications", P: "P · Special treatments" };

function Answer({ name, value }) {
  if (value === null || value === undefined) return <span className="ans none">{name}: not run</span>;
  return <span className={`ans ${value ? "yes" : "no"}`}>{name}: {value ? "yes" : "no"}</span>;
}

const pretty = (status) => status.map((x) => x.replaceAll("_", " ")).join(", ");
const jump = (id) => document.getElementById(`s-${id}`)?.scrollIntoView({ block: "center", behavior: "smooth" });

function Detail({ item, label, mode, onComment }) {
  const show = visible(mode, label);
  return (
    <div className="detail">
      <p className="small">
        {item.wording}
        <br />
        <span className="muted">
          Look-back: {item.look_back_days ? `last ${item.look_back_days} days before discharge` : "active during this stay"}
        </span>
      </p>
      {show ? (
        <>
          {item.direct && item.direct.evidence.length > 0 && (
            <div className="ev ev-d small">
              <strong>Direct, evidence sentences</strong>
              <div>
                {item.direct.evidence.map((id) => (
                  <button key={id} className="evline" onClick={() => jump(id)}>
                    sentence {id}
                  </button>
                ))}
              </div>
            </div>
          )}
          {item.graph.note_evidence.length > 0 && (
            <div className="ev small">
              <strong>Graph, note evidence</strong>
              {item.graph.note_evidence.map((e, i) => (
                <button key={i} className="evline" onClick={() => jump(e.sentence)} title="Show this sentence in the note">
                  “{e.concept}” · sentence {e.sentence} · {pretty(e.status)}
                </button>
              ))}
            </div>
          )}
          {item.graph.record_evidence.length > 0 && (
            <div className="ev small">
              <strong>Graph, records</strong>
              {item.graph.record_evidence.slice(0, 15).map((e, i) => (
                <div key={i}>
                  {e.source}: {e.concept}
                  {e.time ? ` (${e.time})` : ""}
                </div>
              ))}
              {item.graph.record_evidence.length > 15 && <div className="muted">+{item.graph.record_evidence.length - 15} more</div>}
            </div>
          )}
        </>
      ) : (
        <p className="muted small">System answers are hidden{mode === "after" ? " until you answer" : ""}.</p>
      )}
      <input
        className="comment"
        placeholder="Comment (optional)"
        defaultValue={label?.comment || ""}
        key={item.code}
        disabled={!label}
        title={label ? "" : "Label the item first"}
        onBlur={(e) => e.target.value !== (label?.comment || "") && onComment(item.code, e.target.value)}
      />
    </div>
  );
}

export default function ItemPanel(p) {
  const { meta, adm, section, onSection, items, labels, selected, onSelect, onLabel, onComment, onRestAbsent, mode } = p;
  const listRef = useRef(null);
  useEffect(() => {
    listRef.current?.querySelector(".item.cur")?.scrollIntoView({ block: "nearest" });
  }, [selected]);
  useEffect(() => {
    const onKey = (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
      const order = ["I", "O", "P"];
      const i = order.indexOf(section);
      if (e.key === "]" && i < 2) onSection(order[i + 1]);
      if (e.key === "[" && i > 0) onSection(order[i - 1]);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [section, onSection]);
  useEffect(() => {
    if (items.length && !items.some((i) => i.code === selected)) onSelect(items[0].code);
  }, [section, items]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!adm || !meta) return <aside className="panel card muted pad">Loading…</aside>;
  const count = (s) => adm.items.filter((i) => i.section === s && labels[i.code]).length;
  const total = (s) => adm.items.filter((i) => i.section === s).length;
  const groups = meta.groups.filter((g) => g.section === section);
  return (
    <aside className="panel card" aria-label="MDS items">
      <div className="tabs" role="tablist">
        {["I", "O", "P"].map((s) => (
          <button key={s} role="tab" aria-selected={s === section} className={s === section ? "on" : ""} onClick={() => onSection(s)}>
            {SECTION_SHORT[s]} <span className="muted small">{count(s)}/{total(s)}</span>
          </button>
        ))}
      </div>
      {p.canFilter && (
        <label className="filter-line small">
          <input type="checkbox" checked={p.reviewOnly} onChange={(e) => p.onReviewOnly(e.target.checked)} /> Only items to check (direct and graph disagree):{" "}
          {(p.sectionItems || []).filter((i) => ["graph_only", "direct_only"].includes(i.agreement)).length} in this section
        </label>
      )}
      <div className="items" ref={listRef}>
        {groups.map((g) => (
          <section key={g.prefix}>
            <h4>{g.title}</h4>
            {items
              .filter((i) => i.code.startsWith(g.prefix))
              .map((i) => {
                const l = labels[i.code];
                const show = visible(mode, l);
                return (
                  <div key={i.code} className={`item ${i.code === selected ? "cur" : ""}`} onClick={() => onSelect(i.code)}>
                    <div className="item-row">
                      <span className="code">{i.code}</span>
                      <span className="name">{i.label}</span>
                      <span className="btns" onClick={(e) => e.stopPropagation()}>
                        {LABELS.map(([v, t, k]) => (
                          <button key={v} className={`lab ${v} ${l?.label === v ? "on" : ""}`} title={`${t} (key ${k})`} onClick={() => onLabel(i.code, v)}>
                            {t}
                          </button>
                        ))}
                      </span>
                    </div>
                    {show && (
                      <div className="answers">
                        <Answer name="Direct" value={i.direct ? i.direct.answer : null} />
                        <Answer name="Graph" value={i.graph.answer} />
                        <Answer name="Reader" value={i.reader ? i.reader.answer : null} />
                        {i.agreement && i.agreement !== "both_no" && <span className={`agree ${i.agreement}`}>{AGREE_TEXT[i.agreement]}</span>}
                        {i.graph.answer && (
                          <span className="muted small">
                            {i.graph.note_evidence.length ? "note" : ""}
                            {i.graph.note_evidence.length && i.graph.record_evidence.length ? " + " : ""}
                            {i.graph.record_evidence.length ? "records" : ""}
                          </span>
                        )}
                      </div>
                    )}
                    {i.code === selected && <Detail item={i} label={l} mode={mode} onComment={onComment} />}
                  </div>
                );
              })}
          </section>
        ))}
      </div>
      <div className="panel-foot">
        <button className="wide" onClick={onRestAbsent}>Mark rest of this section Absent</button>
        <button onClick={p.onPrev}>Previous</button>
        <button
          className={p.done ? "done" : "primary"}
          disabled={!p.done && !p.allLabelled}
          title={p.allLabelled || p.done ? "" : "Label all 78 items first"}
          onClick={p.onDone}
        >
          {p.done ? "Done ✓ (undo)" : "Mark admission done"}
        </button>
        <button onClick={p.onNext}>Next</button>
      </div>
    </aside>
  );
}
