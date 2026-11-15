"""Test 2: GMDCSA-24 (MIT), a different home, clip level. Bed and sofa clips are left out (out of scope).

Frames sampled at 2 fps; clip score = 2nd-highest frame score (robust to a one-frame blip). Reports clip AP and the
share of fall clips caught when 5% / 10% of everyday-activity clips are allowed to alarm.

  python eval/eval_gmdcsa.py --data /path/to/GMDCSA24 --variant small
"""
import argparse, csv, json
from pathlib import Path
import cv2, numpy as np
from PIL import Image
from common import add_model_args, ap, load_detector

LYING = ["bed", "sleep", "lying", "lie ", "lies", "sofa", "couch"]


def main():
    p = argparse.ArgumentParser(); p.add_argument("--data", type=Path, required=True, help="folder that contains 'Subject 1' ... 'Subject 4'"); add_model_args(p)
    args = p.parse_args(); det = load_detector(args); rows = []
    for subj in sorted(args.data.glob("Subject *")):
        for kind in ["Fall", "ADL"]:
            desc = {r[0].strip(): r[-1].strip() for r in csv.reader(open(subj / f"{kind}.csv")) if r and r[0].strip().endswith(".mp4")}
            for vid in sorted((subj / kind).glob("*.mp4")):
                cap = cv2.VideoCapture(str(vid)); fps = cap.get(cv2.CAP_PROP_FPS) or 25; step = max(1, round(fps / 2)); i = 0; sc = []
                while True:
                    ok, fr = cap.read()
                    if not ok: break
                    if i % step == 0: sc.append(det.frame_score(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))))
                    i += 1
                s = sorted(sc)[-2] if len(sc) > 1 else (sc[0] if sc else 0.0)
                d = desc.get(vid.name, "")
                rows.append({"kind": kind, "clip": f"{subj.name}/{kind}/{vid.name}", "score": s,
                             "bed_or_sofa": kind == "ADL" and any(k in d.lower() for k in LYING)})
    keep = [r for r in rows if not r["bed_or_sofa"]]
    sc = np.array([r["score"] for r in keep]); lab = np.array([r["kind"] == "Fall" for r in keep]).astype(int)
    neg = np.sort(sc[lab == 0])[::-1]
    rec = lambda fpr: round(float((sc[lab == 1] > neg[min(len(neg) - 1, int(fpr * len(neg)))]).mean()), 4)
    print(json.dumps({"name": args.name, "clips_scored": len(keep), "fall_clips": int(lab.sum()), "bed_sofa_clips_left_out": len(rows) - len(keep),
                      "clip_AP": round(ap(sc, lab), 4), "falls_caught@5%clipFA": rec(0.05), "falls_caught@10%clipFA": rec(0.10)}, indent=1))


if __name__ == "__main__":
    main()
