import cv2
import json
import os
from pathlib import Path

INPUT_ROOT = "data/inputs"
OUT_IMG = "data/yolo/images/train"
OUT_LBL = "data/yolo/labels/train"

os.makedirs(OUT_IMG, exist_ok=True)
os.makedirs(OUT_LBL, exist_ok=True)

img_counter = 0

sets = sorted(os.listdir(INPUT_ROOT))
print("Found sets:", sets)

for set_name in sets:
    set_path = os.path.join(INPUT_ROOT, set_name)

    video_path = os.path.join(set_path, "visible.mp4")
    json_path  = os.path.join(set_path, "visible.json")

    if not os.path.exists(video_path) or not os.path.exists(json_path):
        print(f"Skipping {set_name} (missing files)")
        continue

    print(f"Processing {set_name}")

    with open(json_path, "r") as f:
        labels = json.load(f)

    cap = cv2.VideoCapture(video_path)
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w, _ = frame.shape

        img_name = f"img_{img_counter:06d}.jpg"
        cv2.imwrite(os.path.join(OUT_IMG, img_name), frame)

        if labels["exist"][frame_id] == 1:
            x, y, bw, bh = labels["gt_rect"][frame_id]

            xc = (x + bw / 2) / w
            yc = (y + bh / 2) / h
            bw /= w
            bh /= h

            lbl_path = os.path.join(OUT_LBL, img_name.replace(".jpg", ".txt"))
            with open(lbl_path, "w") as lf:
                lf.write(f"0 {xc} {yc} {bw} {bh}")

        img_counter += 1
        frame_id += 1

    cap.release()

print("YOLO dataset generation complete.")
print("Total images:", img_counter)
