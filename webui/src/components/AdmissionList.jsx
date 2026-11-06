import React, { useMemo, useState } from "react";

const FILTERS = [
  ["todo", "To do"],
  ["progress", "In progress"],
  ["done", "Done"],
  ["all", "All"],
];

export default function AdmissionList({ list, current, onPick, mode }) {
  const [filter, setFilter] = useState("todo");
  const [byReview, setByReview] = useState(false);
  const showReview = mode === "shown";
  const [q, setQ] = useState("");
  const shown = useMemo(
    () =>
      list.filter(
        (a) =>
          a.key.toLowerCase().includes(q.toLowerCase()) &&
          (a.key === current ||
            filter === "all" ||
            (filter === "done" && a.done) ||
            (filter === "todo" && !a.done && a.labelled === 0) ||
            (filter === "progress" && !a.done && a.labelled > 0)),
      ),
    [list, filter, q, current],
  );
  const ordered = useMemo(
    () => (byReview && showReview ? [...shown].sort((a, b) => b.needs_review - a.needs_review) : shown),
    [shown, byReview, showReview],
  );
  return (
    <nav className="adm-list" aria-label="Admissions">
      <div className="filters" role="tablist">
        {FILTERS.map(([v, t]) => (
          <button key={v} role="tab" aria-selected={filter === v} className={filter === v ? "on" : ""} onClick={() => setFilter(v)}>
            {t}
          </button>
        ))}
      </div>
      <input className="search" placeholder="Find admission" value={q} onChange={(e) => setQ(e.target.value)} />
      {showReview && (
        <label className="small muted">
          <input type="checkbox" checked={byReview} onChange={(e) => setByReview(e.target.checked)} /> Most items to check first
        </label>
      )}
      <ul>
        {ordered.map((a) => (
          <li key={a.key}>
            <button className={`adm ${a.key === current ? "cur" : ""}`} onClick={() => onPick(a.key)}>
              <span className="adm-key">
                {a.key} {a.done && <span className="tick" aria-label="done">✓</span>}
              </span>
              <span className="bar" aria-hidden="true">
                <span style={{ width: `${(100 * a.labelled) / a.items}%` }} />
              </span>
              <span className="muted small">
                {a.labelled}/{a.items} labelled{showReview ? ` · ${a.needs_review} to check` : ""}{a.done_by > 1 ? ` · ${a.done_by} reviewers` : ""}
              </span>
            </button>
          </li>
        ))}
        {!ordered.length && <li className="muted small pad">Nothing here.</li>}
      </ul>
    </nav>
  );
}
