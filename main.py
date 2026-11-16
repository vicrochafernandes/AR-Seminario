import argparse
from collections import Counter
from pathlib import Path

import dataset
import train

import cv2
import mediapipe as mp
from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_ground_truth(img_path, w, h, names):
    """Read YOLO-format labels (detection datasets) for an image."""
    label = img_path.parent.parent / "labels" / f"{img_path.stem}.txt"
    boxes = []
    if not label.exists():
        return boxes
    for line in label.read_text().splitlines():
        p = line.split()
        if len(p) != 5:  # skip segmentation/polygon rows
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
        boxes.append((int((cx - bw / 2) * w), int((cy - bh / 2) * h),
                      int((cx + bw / 2) * w), int((cy + bh / 2) * h), names[c]))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Roboflow download folder (contains data.yaml)")
    ap.add_argument("--split", default="valid", choices=["train", "valid", "test"])
    ap.add_argument("--weights", default="runs/detect/train/weights/best.pt")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--hands", action="store_true", help="also run MediaPipe hands")
    ap.add_argument("--gt", action="store_true", help="draw ground-truth boxes from the dataset labels")
    ap.add_argument("--show", action="store_true", help="open a window (needs a display)")
    ap.add_argument("--limit", type=int, default=0, help="max images (0 = all)")
    args = ap.parse_args()

    model = YOLO(args.weights)
    hands = (mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2,
                                      min_detection_confidence=0.5)
             if args.hands else None)
    draw = mp.solutions.drawing_utils

    img_dir = Path(args.dataset) / args.split / "images"
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    if args.limit:
        images = images[: args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    totals = Counter()

    for img_path in images:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        h, w = frame.shape[:2]

        # Ground truth (blue)
        if args.gt:
            for x1, y1, x2, y2, name in load_ground_truth(img_path, w, h, model.names):
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 120, 0), 2)
                cv2.putText(frame, f"GT {name}", (x1, y2 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 120, 0), 2)

        # Detection (green)
        boxes = []
        for r in model(frame, verbose=False, conf=args.conf):
            for b in r.boxes:
                x1, y1, x2, y2 = map(int, b.xyxy[0])
                name = model.names[int(b.cls[0])]
                conf = float(b.conf[0])
                boxes.append((x1, y1, x2, y2, name))
                totals[name] += 1

                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 120), -1)
                frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 120), 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(y1 - 10, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 120), 2)

        # Optional hands
        if hands:
            res = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            for lm in (res.multi_hand_landmarks or []):
                draw.draw_landmarks(frame, lm, mp.solutions.hands.HAND_CONNECTIONS)
                tip = lm.landmark[8]
                px, py = int(tip.x * w), int(tip.y * h)
                for x1, y1, x2, y2, name in boxes:
                    if x1 < px < x2 and y1 < py < y2:
                        cv2.putText(frame, f"Selected: {name}", (x1, y2 + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

        cv2.imwrite(str(out_dir / img_path.name), frame)
        print(f"{img_path.name}: {[b[4] for b in boxes] or 'nothing detected'}")

        if args.show:
            cv2.imshow("AR dataset", frame)
            if cv2.waitKey(0) & 0xFF == 27:  # any key = next, Esc = quit
                break

    cv2.destroyAllWindows()
    print("\nDetections per class:", dict(totals))
    print(f"Annotated images saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()