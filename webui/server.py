"""Local server for the MDS labelling web UI: serves the built app and stores labels in SQLite. Python standard library only.

Run (after `npm run build` in webui/):
  python webui/server.py --data /home/user/mimic/review/review_data.json --db /home/user/mimic/review/labels.sqlite
Binds to 127.0.0.1 by default; pass --host 0.0.0.0 to let a labelling crew on the same network use it. The data file and the
database hold MIMIC text and labels: keep both off the repository and inside your data agreement.
"""
import argparse
import json
import sqlite3
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

DIST = Path(__file__).resolve().parent / "dist"
LABELS = {"present", "absent", "unclear"}
MODES = {"shown", "after", "hidden"}


class Store:
    def __init__(self, data_path, db_path):
        self.data = json.loads(Path(data_path).read_text())
        self.by_key = {a["key"]: a for a in self.data["admissions"]}
        self.codes = {i["code"] for a in self.data["admissions"] for i in a["items"]}
        self.lock = threading.Lock()
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS labels (reviewer TEXT, note_key TEXT, item TEXT, label TEXT, comment TEXT,
                suggestions_mode TEXT, updated_at TEXT, PRIMARY KEY (reviewer, note_key, item));
            CREATE TABLE IF NOT EXISTS done (reviewer TEXT, note_key TEXT, done_at TEXT, PRIMARY KEY (reviewer, note_key));""")
        self.db.commit()

    def query(self, sql, args=()):
        with self.lock:
            return self.db.execute(sql, args).fetchall()

    def write(self, sql, args=()):
        with self.lock:
            self.db.execute(sql, args)
            self.db.commit()


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def make_handler(store):
    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(DIST), **k)

        def log_message(self, *a):
            pass

        def send_json(self, obj, status=200):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def body(self):
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path == "/api/meta":
                d = store.data
                return self.send_json({k: d[k] for k in ("form", "sections", "groups", "systems")} |
                                      {"reviewers": [r[0] for r in store.query("SELECT DISTINCT reviewer FROM labels ORDER BY 1")]})
            if u.path == "/api/admissions":
                rev = q.get("reviewer", "")
                counts = dict(store.query("SELECT note_key, COUNT(*) FROM labels WHERE reviewer=? GROUP BY note_key", (rev,)))
                done = {r[0] for r in store.query("SELECT note_key FROM done WHERE reviewer=?", (rev,))}
                anyone = dict(store.query("SELECT note_key, COUNT(DISTINCT reviewer) FROM done GROUP BY note_key"))
                return self.send_json([{"key": a["key"], "items": len(a["items"]), "labelled": counts.get(a["key"], 0),
                                        "done": a["key"] in done, "done_by": anyone.get(a["key"], 0),
                                        "suggested": sum(i["graph"]["answer"] for i in a["items"])} for a in store.data["admissions"]])
            if u.path.startswith("/api/admissions/"):
                a = store.by_key.get(u.path.rsplit("/", 1)[-1])
                return self.send_json(a) if a else self.send_json({"error": "not found"}, 404)
            if u.path == "/api/labels":
                rows = store.query("SELECT item, label, comment, suggestions_mode, updated_at FROM labels WHERE reviewer=? AND note_key=?",
                                   (q.get("reviewer", ""), q.get("key", "")))
                return self.send_json({r[0]: {"label": r[1], "comment": r[2], "mode": r[3], "updated_at": r[4]} for r in rows})
            if u.path == "/api/export":
                rows = store.query("SELECT reviewer, note_key, item, label, comment, suggestions_mode, updated_at FROM labels ORDER BY 1, 2, 3")
                done = store.query("SELECT reviewer, note_key, done_at FROM done ORDER BY 1, 2")
                return self.send_json({"form": store.data["form"], "exported_at": now(),
                                       "labels": [dict(zip(("reviewer", "note_key", "item", "label", "comment", "suggestions_mode", "updated_at"), r)) for r in rows],
                                       "done": [dict(zip(("reviewer", "note_key", "done_at"), r)) for r in done]})
            if u.path.startswith("/api/"):
                return self.send_json({"error": "unknown endpoint"}, 404)
            if not (DIST / u.path.lstrip("/")).is_file():
                self.path = "/index.html"
            return super().do_GET()

        def do_POST(self):
            u = urlparse(self.path)
            b = self.body()
            rev, key = (b.get("reviewer") or "").strip(), b.get("key")
            if not rev or key not in store.by_key:
                return self.send_json({"error": "reviewer and a known admission key are required"}, 400)
            if u.path == "/api/labels":
                items = b.get("items") or [{"code": b.get("code"), "label": b.get("label"), "comment": b.get("comment", "")}]
                mode = b.get("mode") if b.get("mode") in MODES else "shown"
                for it in items:
                    if it.get("code") not in store.codes:
                        return self.send_json({"error": f"unknown item {it.get('code')}"}, 400)
                    if it.get("label") is None:
                        store.write("DELETE FROM labels WHERE reviewer=? AND note_key=? AND item=?", (rev, key, it["code"]))
                    elif it["label"] in LABELS:
                        store.write("INSERT INTO labels VALUES (?,?,?,?,?,?,?) ON CONFLICT(reviewer, note_key, item) DO UPDATE SET "
                                    "label=excluded.label, comment=excluded.comment, suggestions_mode=excluded.suggestions_mode, "
                                    "updated_at=excluded.updated_at",
                                    (rev, key, it["code"], it["label"], it.get("comment", ""), mode, now()))
                    else:
                        return self.send_json({"error": "label must be present, absent, unclear or null"}, 400)
                return self.send_json({"ok": True, "saved": len(items)})
            if u.path == "/api/done":
                if b.get("done"):
                    store.write("INSERT OR REPLACE INTO done VALUES (?,?,?)", (rev, key, now()))
                else:
                    store.write("DELETE FROM done WHERE reviewer=? AND note_key=?", (rev, key))
                return self.send_json({"ok": True})
            return self.send_json({"error": "unknown endpoint"}, 404)

    return Handler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="/home/user/mimic/review/review_data.json")
    p.add_argument("--db", default="/home/user/mimic/review/labels.sqlite")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8765)
    a = p.parse_args()
    store = Store(a.data, a.db)
    print(f"MDS labelling UI on http://{a.host}:{a.port} ({len(store.by_key)} admissions)", flush=True)
    ThreadingHTTPServer((a.host, a.port), make_handler(store)).serve_forever()


if __name__ == "__main__":
    main()
