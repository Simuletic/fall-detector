"""Test 1: CAUCAFall (CC BY 4.0), subjects 6-10, single frames.

Ground truth is 'is the person lying on the floor', not 'is this the falling motion':
  positive: fall-labelled frames from 1 s after the onset;
  negative: all everyday-activity frames (walk, hop, sit down, kneel, pick up an object) and fall-clip frames before the onset;
  left out: the second either side of the onset (mid-fall) and post-onset frames the dataset labels no-fall.
Frame score = highest fallen_person score. Reports AP, recall at 5% / 1% of negative frames alarming, the end-on
subset (box taller than wide: person lying with head or feet towards the camera), and per-activity false alarms at
the threshold that keeps 80% recall.

  python eval/eval_caucafall.py --data "/path/to/CAUCAFall" --variant small
"""
import argparse, json, re
from pathlib import Path
import numpy as np
from PIL import Image
from common import add_model_args, ap, load_detector

TEST_SUBJECTS = {6, 7, 8, 9, 10}
TRANSITION = 25  # raw frames (~1 s at 25 fps)


def frames(root: Path):
    out = []
    for clip in sorted(root.glob("Subject.*/*")):
        subj = int(re.search(r"Subject\.(\d+)", str(clip)).group(1))
        if subj not in TEST_SUBJECTS or not clip.is_dir():
            continue
        txts = sorted(clip.glob("*.txt")); labels = []
        for t in txts:
            v = open(t).read().split(); labels.append(v if len(v) >= 5 else None)
        onset = next((i for i, v in enumerate(labels) if v and v[0] == "1"), None)
        is_fall = clip.name.startswith("Fall")
        for i, (t, v) in enumerate(zip(txts, labels)):
            png = t.with_suffix(".png")
            if v is None or not png.exists():
                continue
            c = int(v[0])
            if is_fall and onset is not None:
                if abs(i - onset) < TRANSITION: continue
                if i > onset and c == 0: continue
                if i < onset and c == 1: continue
            out.append({"png": png, "activity": clip.name, "cls": c, "box": [float(x) for x in v[1:5]]})
    return out


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, required=True, help="folder that contains Subject.1 ... Subject.10"); add_model_args(p)
    args = p.parse_args(); det = load_detector(args); fs = frames(args.data)
    score = np.array([det.frame_score(Image.open(f["png"])) for f in fs]); lab = np.array([f["cls"] for f in fs])
    neg = np.sort(score[lab == 0])[::-1]
    rec = lambda fpr, m=(lab == 1): float((score[m] > neg[int(fpr * len(neg))]).mean())
    endon = np.array([f["cls"] == 1 and f["box"][2] * Image.open(f["png"]).width <= f["box"][3] * Image.open(f["png"]).height for f in fs])
    thr80 = np.sort(score[lab == 1])[::-1][int(0.8 * (lab == 1).sum())]
    acts = {}
    for f, s in zip(fs, score):
        if f["cls"] == 0: acts.setdefault(f["activity"], []).append(s > thr80)
    res = {"name": args.name, "frames": len(fs), "lying_frames": int(lab.sum()), "frame_AP": round(ap(score, lab), 4),
           "recall@5%FA": round(rec(0.05), 4), "recall@1%FA": round(rec(0.01), 4),
           "endon_frames": int(endon.sum()), "endon_recall@5%FA": round(rec(0.05, endon), 4) if endon.any() else None,
           "false_alarm_rate_by_activity@80%recall": {a: round(float(np.mean(v)), 3) for a, v in sorted(acts.items())}}
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
