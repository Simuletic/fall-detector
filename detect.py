"""Run the Simuletic Fall Detector on an image or a video.

  python detect.py photo.jpg                          # writes photo_fall_detector.jpg with boxes
  python detect.py clip.mp4 --fps 5 --seconds 2       # per-frame scores + 'stays down' alarm, annotated video
  python detect.py clip.mp4 --variant nano --onnx     # Nano weights on ONNX Runtime

Thresholds: the default 0.5 is a starting point, not a tuned operating point. Pick yours on footage from your own
cameras (see the model card for how the published numbers were measured).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from fall_detector import FallDetector, StaysDownAlarm

COLORS = {"person": (80, 160, 255), "fallen_person": (255, 60, 60)}


def draw(image: Image.Image, dets, threshold: float, banner: str | None = None) -> Image.Image:
    im = image.convert("RGB").copy()
    d = ImageDraw.Draw(im)
    for det in dets:
        if det.score < threshold:
            continue
        d.rectangle(det.box, outline=COLORS[det.label], width=3)
        d.text((det.box[0] + 4, det.box[1] + 2), f"{det.label} {det.score:.2f}", fill=COLORS[det.label])
    if banner:
        d.rectangle((0, 0, im.width, 28), fill=(200, 0, 0))
        d.text((8, 7), banner, fill=(255, 255, 255))
    return im


def run_image(det, path: Path, args):
    image = Image.open(path)
    dets = det(image)
    out = path.with_name(path.stem + "_fall_detector.jpg")
    draw(image, dets, args.threshold).save(out)
    score = max((x.score for x in dets if x.label == "fallen_person"), default=0.0)
    print(json.dumps({"image": str(path), "on_floor_score": round(score, 3), "person_on_floor": score >= args.threshold,
                      "annotated": str(out)}))


def run_video(det, path: Path, args):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, round(fps / args.fps))
    alarm = StaysDownAlarm(threshold=args.threshold, seconds=args.seconds)
    out_path = path.with_name(path.stem + "_fall_detector.mp4")
    writer, rows, i = None, [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            t = i / fps
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            dets = det(image)
            score = max((x.score for x in dets if x.label == "fallen_person"), default=0.0)
            fired, persistent = alarm.update(t, score)
            rows.append({"t": round(t, 2), "score": round(score, 3), "persistent": None if persistent is None else round(persistent, 3),
                         "alarm": alarm.active})
            if fired:
                print(f"ALARM at {t:.1f}s: on the floor for {args.seconds:.0f}s")
            vis = draw(image, dets, args.threshold, "FALL ALARM: PERSON ON THE FLOOR" if alarm.active else None)
            bgr = cv2.cvtColor(np.asarray(vis), cv2.COLOR_RGB2BGR)
            if writer is None:
                writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps / step, (bgr.shape[1], bgr.shape[0]))
            writer.write(bgr)
        i += 1
    if writer:
        writer.release()
    json.dump(rows, open(path.with_name(path.stem + "_fall_detector.json"), "w"), indent=1)
    print(json.dumps({"video": str(path), "frames_scored": len(rows), "alarm_frames": sum(r["alarm"] for r in rows),
                      "annotated": str(out_path)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("--variant", default="nano", choices=["small", "nano"])
    ap.add_argument("--weights", help="local model folder instead of the Hub")
    ap.add_argument("--onnx", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--fps", type=float, default=5, help="frames per second to score in videos")
    ap.add_argument("--seconds", type=float, default=2.0, help="how long someone must stay down before the alarm")
    args = ap.parse_args()
    det = FallDetector(args.variant, args.weights, onnx=args.onnx)
    if args.source.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}:
        run_video(det, args.source, args)
    else:
        run_image(det, args.source, args)


if __name__ == "__main__":
    main()
