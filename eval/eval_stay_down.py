"""'Stays down for 2 seconds' alarm on CAUCAFall subjects 6-10 (Test 1).

A deployed alarm fires only when the person has looked like lying for 2 s in a row; bending down to pick something up
lasts about a second. Persistent score at frame t = lowest single-frame score over the last 2 s. Every 5th raw frame
(25 fps) is scored, as in the CAUCAFall release.
  frames: negatives as in eval_caucafall.py; positives = lying frames at least 2 s after the person is first seen lying.
  events: each fall clip's lying part is a positive event; the 25 everyday clips and the standing part of each fall clip
          are negative events. Reports the share of falls caught when 0 / 1 / 2 negative events are allowed to alarm,
          and the highest-scoring negative events.
Small test (24 falls, 34 negative events, one home): one event is about 4 points. Report next to the single-frame numbers.

  python eval/eval_stay_down.py --data "/path/to/CAUCAFall" --variant small
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image
from common import add_model_args, ap, load_detector
from eval_caucafall import TEST_SUBJECTS, TRANSITION

WINDOW = 50  # raw frames = 2 s at 25 fps


def clips(root: Path):
    for clip in sorted(root.glob("Subject.*/*")):
        if not clip.is_dir() or int(clip.parent.name.split(".")[1]) not in TEST_SUBJECTS: continue
        txts = sorted(clip.glob("*.txt")); rows = []
        for i, t in enumerate(txts):
            v = open(t).read().split(); png = t.with_suffix(".png")
            if png.exists(): rows.append({"i": i, "png": str(png), "cls": int(v[0]) if len(v) >= 5 else None})
        labels = [open(t).read().split() for t in txts]
        onset = next((i for i, v in enumerate(labels) if len(v) >= 5 and v[0] == "1"), None)
        yield clip.parent.name + "/" + clip.name, clip.name.startswith("Fall"), onset, rows


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, required=True); add_model_args(p)
    args = p.parse_args(); det = load_detector(args); data = []
    for cid, is_fall, onset, rows in clips(args.data):
        for r in rows: r["score"] = det.frame_score(Image.open(r["png"]))
        data.append({"clip": cid, "fall": is_fall, "onset": onset, "rows": rows})
    ss, sl, ev_pos, ev_neg = [], [], [], []
    for c in data:
        rows, onset = c["rows"], c["onset"]
        for r in rows:
            win = [q["score"] for q in rows if r["i"] - WINDOW < q["i"] <= r["i"]]
            r["persist"] = min(win) if r["i"] - rows[0]["i"] >= WINDOW - 5 else None
        first_lying = next((r["i"] for r in rows if c["fall"] and onset is not None and r["i"] >= onset + TRANSITION and r["cls"] == 1), None)
        pos, neg = [], []
        for r in rows:
            if r["cls"] is None or r["persist"] is None: continue
            if c["fall"] and onset is not None:
                if r["i"] < onset - TRANSITION and r["cls"] == 0: lab = 0
                elif first_lying is not None and r["i"] >= first_lying + WINDOW and r["cls"] == 1: lab = 1
                else: continue
            else: lab = 0
            (pos if lab else neg).append(r["persist"]); ss.append(r["persist"]); sl.append(lab)
        if pos: ev_pos.append(max(pos))
        if neg: ev_neg.append(max(neg))
    ss, sl = np.array(ss), np.array(sl); negs = np.sort(ss[sl == 0])[::-1]; en = np.sort(ev_neg)[::-1]
    rec = lambda fpr: round(float((ss[sl == 1] > negs[int(fpr * len(negs))]).mean()), 4)
    ev = lambda k: round(float((np.array(ev_pos) > en[k]).mean()), 4)
    top = sorted(((round(max([r["persist"] for r in c["rows"] if r["persist"] is not None and r["cls"] == 0] or [0]), 3), c["clip"])
                  for c in data if not c["fall"]), reverse=True)[:4]
    print(json.dumps({"name": args.name, "frames": int(len(ss)), "lying_frames": int(sl.sum()), "persistent_AP": round(ap(ss, sl), 4),
                      "recall@5%FA": rec(0.05), "recall@1%FA": rec(0.01), "recall@0FA": round(float((ss[sl == 1] > negs[0]).mean()), 4),
                      "fall_events": len(ev_pos), "negative_events": len(ev_neg),
                      "falls_caught@0_false_alarm_events": ev(0), "falls_caught@1_false_alarm_event": ev(1), "falls_caught@2_false_alarm_events": ev(2),
                      "top_negative_events": top}, indent=1))


if __name__ == "__main__":
    main()
