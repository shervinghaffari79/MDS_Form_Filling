async function call(path, options = {}) {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.error || `Request failed (${res.status})`);
  return body;
}

export const api = {
  meta: () => call("/api/meta"),
  admissions: (reviewer) => call(`/api/admissions?reviewer=${encodeURIComponent(reviewer)}`),
  admission: (key) => call(`/api/admissions/${encodeURIComponent(key)}`),
  labels: (reviewer, key) => call(`/api/labels?reviewer=${encodeURIComponent(reviewer)}&key=${encodeURIComponent(key)}`),
  saveLabels: (reviewer, key, items, mode) =>
    call("/api/labels", { method: "POST", body: JSON.stringify({ reviewer, key, items, mode }) }),
  setDone: (reviewer, key, done) => call("/api/done", { method: "POST", body: JSON.stringify({ reviewer, key, done }) }),
  exportAll: () => call("/api/export"),
};
