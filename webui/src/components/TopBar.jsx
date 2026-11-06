import React, { useState } from "react";

const MODES = [
  ["shown", "Show system answers"],
  ["after", "Show after I answer"],
  ["hidden", "Hide system answers"],
];

export default function TopBar({ reviewer, onReviewer, mode, onMode, list, status, onExport }) {
  const [help, setHelp] = useState(false);
  const done = list.filter((a) => a.done).length;
  const labelled = list.reduce((n, a) => n + a.labelled, 0);
  const total = list.reduce((n, a) => n + a.items, 0);
  return (
    <header className="topbar">
      <strong className="brand">MDS 2.0 labelling</strong>
      <span className="muted">
        {done} / {list.length} admissions done · {labelled} / {total} items labelled
      </span>
      <span className="spacer" />
      <label className="mode" title="Seeing system answers first can bias labels; your choice is stored with every label.">
        <select value={mode} onChange={(e) => onMode(e.target.value)}>
          {MODES.map(([v, t]) => (
            <option key={v} value={v}>
              {t}
            </option>
          ))}
        </select>
      </label>
      <span className="status" aria-live="polite">
        {status}
      </span>
      <button onClick={() => setHelp((h) => !h)} aria-expanded={help}>
        Shortcuts
      </button>
      <button onClick={onExport}>Export labels</button>
      <button className="chip" onClick={onReviewer} title="Switch reviewer">
        {reviewer}
      </button>
      {help && (
        <div className="help card" role="dialog">
          <p>
            <kbd>↑</kbd>/<kbd>↓</kbd> or <kbd>k</kbd>/<kbd>j</kbd> move between items · <kbd>1</kbd> Present · <kbd>2</kbd> Absent ·{" "}
            <kbd>3</kbd> Can't tell (pressing the same key again clears it)
          </p>
          <p>
            <kbd>[</kbd>/<kbd>]</kbd> previous / next section · <kbd>p</kbd>/<kbd>n</kbd> previous / next admission
          </p>
          <button onClick={() => setHelp(false)}>Close</button>
        </div>
      )}
    </header>
  );
}
