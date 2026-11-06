# MDS 2.0 labelling web UI

A React app for nurses or a labelling crew to label MIMIC-IV admissions on MDS 2.0 Sections I (diagnoses), O (medications) and
P (special treatments), next to the systems' answers and evidence. The labels become the reference for scoring the systems
(`src/sweetspot/p2_score_labels.py`).

The app and server code live here. The data they load holds MIMIC note text and is **not** in this repository: build it locally
and keep it, and the labels database, inside your PhysioNet data use agreement.

## Run
```bash
# 1. Build the labelling data (needs the pilot outputs in /home/user/mimic/sweetspot)
python src/sweetspot/export_review_data.py /home/user/mimic/review/review_data.json

# 2. Build the app
cd webui && npm install && npm run build && cd ..

# 3. Serve it (Python standard library only); add --host 0.0.0.0 to share it on a local network
python webui/server.py --data /home/user/mimic/review/review_data.json --db /home/user/mimic/review/labels.sqlite
# open http://127.0.0.1:8765
```
For development, run the server and `npm run dev` side by side; Vite proxies `/api` to port 8765.

## For reviewers
- Enter your initials. Each reviewer's labels are stored separately, so several people can label the same admissions.
- Left: admissions to do, in progress and done. Middle: the discharge summary, with a search box. Right: the 78 items in three
  tabs (I, O, P), each with Present, Absent or Can't tell and an optional comment.
- Judge every item yourself from the note and the item's look-back period, shown under the item.
- System answers: choose "Show", "Show after I answer" or "Hide" at the top. Seeing answers first can pull labels towards them, so
  the choice is stored with every label and the scores are reported for each mode. "Show after I answer" or "Hide" gives the
  cleanest reference.
- With answers visible, highlighted sentences are the evidence the systems used (amber: knowledge graph, blue: note reader);
  click an evidence line to jump to its sentence. Records evidence (diagnosis codes, prescriptions, eMAR, ICU events) is listed
  with dates.
- "Mark rest of this section Absent" fills the remaining unlabelled items of the open tab; "Mark admission done" becomes available
  when all 78 items are labelled.
- Keys: up/down or j/k move between items; 1 Present, 2 Absent, 3 Can't tell (press again to clear); [ and ] change section;
  p and n change admission.
- "Export labels" downloads every reviewer's labels as JSON.

## Scoring
```bash
python src/sweetspot/p2_score_labels.py --db /home/user/mimic/review/labels.sqlite
```
Precision, recall and F1 per system and section over admissions marked done, with 95% intervals; agreement between reviewers
(Cohen's kappa) where admissions are double-labelled; and the same scores split by how system answers were shown.
