import cv2
import os
import glob

INPUT_DIR = "../data/inputs"
OUT_RGB = "../data/samples_rgb"
OUT_IR  = "../data/samples_ir"

os.makedirs(OUT_RGB, exist_ok=True)
os.makedirs(OUT_IR, exist_ok=True)

sets = sorted(glob.glob(f"{INPUT_DIR}/*"))
print("Found input sets:", len(sets))

roi_count = 0

for s in sets:
    print("Processing set:", s)

    rgb_files = glob.glob(f"{s}/visible.mp4")
    ir_files  = glob.glob(f"{s}/infrared.mp4")

    if len(rgb_files) == 0 or len(ir_files) == 0:
        print("Skipping (missing RGB/IR video):", s)
        continue

    RGB_VIDEO = rgb_files[0]
    IR_VIDEO  = ir_files[0]

    print("  RGB:", RGB_VIDEO)
    print("  IR :", IR_VIDEO)

    rgb_cap = cv2.VideoCapture(RGB_VIDEO)
    ir_cap  = cv2.VideoCapture(IR_VIDEO)

    frame_count = 0
    set_roi_count = 0

    while True:
        ok1, rgb = rgb_cap.read()
        ok2, ir  = ir_cap.read()

        if not ok1 or not ok2:
            break

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        ir_gray = cv2.cvtColor(ir, cv2.COLOR_BGR2GRAY)

        if frame_count == 0:
            prev = gray.copy()
            frame_count += 1
            continue

        diff = cv2.absdiff(prev, gray)
        prev = gray.copy()

        _, mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
        mask = cv2.medianBlur(mask, 5)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            if w * h < 15:
                continue

            rgb_roi = gray[y:y+h, x:x+w]
            ir_roi  = ir_gray[y:y+h, x:x+w]

            if rgb_roi.size == 0 or ir_roi.size == 0:
                continue

            cv2.imwrite(f"{OUT_RGB}/rgb_{roi_count}.png", rgb_roi)
            cv2.imwrite(f"{OUT_IR}/ir_{roi_count}.png", ir_roi)

            roi_count += 1
            set_roi_count += 1

        frame_count += 1

    rgb_cap.release()
    ir_cap.release()

    print(f"ROIs extracted from {s}: {set_roi_count}")

print("\nROI extraction complete.")
print("Total ROIs saved:", roi_count)
