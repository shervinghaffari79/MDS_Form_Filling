import React, { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "./api.js";
import TopBar from "./components/TopBar.jsx";
import AdmissionList from "./components/AdmissionList.jsx";
import NoteView from "./components/NoteView.jsx";
import ItemPanel from "./components/ItemPanel.jsx";

const store = {
  get: (k, d) => {
    try {
      return localStorage.getItem(k) ?? d;
    } catch {
      return d;
    }
  },
  set: (k, v) => {
    try {
      localStorage.setItem(k, v);
    } catch {
      /* the page still works without browser storage */
    }
  },
};

export default function App() {
  const [meta, setMeta] = useState(null);
  const [reviewer, setReviewer] = useState(store.get("mds.reviewer", ""));
  const [mode, setMode] = useState(store.get("mds.mode", "shown"));
  const [list, setList] = useState([]);
  const [key, setKey] = useState(null);
  const [adm, setAdm] = useState(null);
  const [labels, setLabels] = useState({});
  const [section, setSection] = useState("I");
  const [reviewOnly, setReviewOnly] = useState(store.get("mds.reviewOnly", "0") === "1");
  const [selected, setSelected] = useState(null);
  const [status, setStatus] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    api.meta().then(setMeta).catch((e) => setError(e.message));
  }, []);

  const refreshList = useCallback(() => {
    if (reviewer) api.admissions(reviewer).then(setList).catch((e) => setError(e.message));
  }, [reviewer]);
  useEffect(refreshList, [refreshList]);

  useEffect(() => {
    if (!key && list.length) setKey((list.find((a) => !a.done) || list[0]).key);
  }, [list, key]);

  useEffect(() => {
    if (!key || !reviewer) return;
    setAdm(null);
    Promise.all([api.admission(key), api.labels(reviewer, key)])
      .then(([a, l]) => {
        setAdm(a);
        setLabels(l);
        setSelected(a.items.find((i) => i.section === section)?.code ?? null);
      })
      .catch((e) => setError(e.message));
  }, [key, reviewer]); // eslint-disable-line react-hooks/exhaustive-deps

  const sectionItems = useMemo(() => (adm ? adm.items.filter((i) => i.section === section) : []), [adm, section]);
  // The filter shows only items where the systems disagree; it is offered only while system answers are shown, so it cannot leak them.
  const filtering = reviewOnly && mode === "shown";
  const items = useMemo(
    () => (filtering ? sectionItems.filter((i) => ["graph_only", "direct_only"].includes(i.agreement)) : sectionItems),
    [sectionItems, filtering],
  );
  const selectedItem = adm?.items.find((i) => i.code === selected) ?? null;

  const save = useCallback(
    async (entries) => {
      const next = { ...labels };
      entries.forEach(({ code, label, comment }) => {
        if (label === null) delete next[code];
        else next[code] = { ...(labels[code] || {}), label, comment: comment ?? labels[code]?.comment ?? "", mode };
      });
      setLabels(next);
      setStatus("Saving…");
      try {
        await api.saveLabels(
          reviewer,
          key,
          entries.map((e) => ({ code: e.code, label: e.label, comment: e.comment ?? next[e.code]?.comment ?? "" })),
          mode,
        );
        setStatus("Saved");
        setList((l) => l.map((a) => (a.key === key ? { ...a, labelled: Object.keys(next).length } : a)));
      } catch (e) {
        setStatus("");
        setError(`Not saved: ${e.message}`);
      }
    },
    [labels, reviewer, key, mode],
  );

  const setLabel = (code, label) => {
    save([{ code, label: labels[code]?.label === label ? null : label }]);
    const idx = items.findIndex((i) => i.code === code);
    if (idx >= 0 && idx < items.length - 1 && labels[code]?.label !== label) setSelected(items[idx + 1].code);
  };
  const setComment = (code, comment) => {
    if (labels[code]) save([{ code, label: labels[code].label, comment }]);
  };
  const markRestAbsent = () => {
    const rest = sectionItems.filter((i) => !labels[i.code]);
    if (!rest.length) return;
    if (window.confirm(`Mark the ${rest.length} unlabelled items in this section as Absent?`))
      save(rest.map((i) => ({ code: i.code, label: "absent" })));
  };
  const allLabelled = adm && adm.items.every((i) => labels[i.code]);
  const current = list.find((a) => a.key === key);
  const toggleDone = async () => {
    const done = !current?.done;
    await api.setDone(reviewer, key, done).catch((e) => setError(e.message));
    const updated = list.map((a) => (a.key === key ? { ...a, done } : a));
    setList(updated);
    if (done) {
      const next = updated.find((a) => !a.done && a.key !== key);
      if (next) setKey(next.key);
    }
  };
  const go = (step) => {
    const i = list.findIndex((a) => a.key === key);
    const n = list[i + step];
    if (n) setKey(n.key);
  };
  const exportAll = async () => {
    const data = await api.exportAll();
    const blob = new Blob([JSON.stringify(data, null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `mds_labels_${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  };

  useEffect(() => {
    const onKey = (e) => {
      if (["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName) || !adm) return;
      const idx = items.findIndex((i) => i.code === selected);
      if (e.key === "ArrowDown" || e.key === "j") {
        e.preventDefault();
        if (idx < items.length - 1) setSelected(items[idx + 1].code);
      } else if (e.key === "ArrowUp" || e.key === "k") {
        e.preventDefault();
        if (idx > 0) setSelected(items[idx - 1].code);
      } else if (selected && ["1", "2", "3"].includes(e.key)) {
        setLabel(selected, { 1: "present", 2: "absent", 3: "unclear" }[e.key]);
      } else if (e.key === "n") go(1);
      else if (e.key === "p") go(-1);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (!reviewer)
    return (
      <div className="welcome">
        <form
          className="card"
          onSubmit={(e) => {
            e.preventDefault();
            const v = new FormData(e.target).get("who").trim();
            if (v) {
              store.set("mds.reviewer", v);
              setReviewer(v);
            }
          }}
        >
          <h1>MDS 2.0 labelling</h1>
          <p className="muted">
            Label Sections I, O and P for each admission. Your labels become the reference for measuring the systems, so judge each
            item from the note and records yourself.
          </p>
          <label>
            Your initials <input name="who" autoFocus maxLength={12} required />
          </label>
          <button className="primary">Start</button>
        </form>
      </div>
    );

  return (
    <div className="app">
      <TopBar
        reviewer={reviewer}
        onReviewer={() => {
          store.set("mds.reviewer", "");
          setReviewer("");
          setKey(null);
        }}
        mode={mode}
        onMode={(m) => {
          store.set("mds.mode", m);
          setMode(m);
        }}
        list={list}
        status={status}
        onExport={exportAll}
      />
      {error && (
        <div className="error" role="alert">
          {error} <button onClick={() => setError("")}>Dismiss</button>
        </div>
      )}
      <div className="layout">
        <AdmissionList list={list} current={key} onPick={setKey} mode={mode} />
        <NoteView adm={adm} item={selectedItem} label={labels[selected]} mode={mode} />
        <ItemPanel
          meta={meta}
          adm={adm}
          section={section}
          onSection={setSection}
          items={items}
          sectionItems={sectionItems}
          reviewOnly={filtering}
          canFilter={mode === "shown"}
          onReviewOnly={(v) => {
            store.set("mds.reviewOnly", v ? "1" : "0");
            setReviewOnly(v);
          }}
          labels={labels}
          selected={selected}
          onSelect={setSelected}
          onLabel={setLabel}
          onComment={setComment}
          onRestAbsent={markRestAbsent}
          mode={mode}
          done={!!current?.done}
          allLabelled={!!allLabelled}
          onDone={toggleDone}
          onPrev={() => go(-1)}
          onNext={() => go(1)}
        />
      </div>
    </div>
  );
}
