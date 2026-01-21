import os
import json

GT_DIR = "data/test"
PRED_DIR = "results"
IOU_THRESH = 0.3

def iou(a, b):
    xA = max(a[0], b[0])
    yA = max(a[1], b[1])
    xB = min(a[0]+a[2], b[0]+b[2])
    yB = min(a[1]+a[3], b[1]+b[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = a[2] * a[3]
    areaB = b[2] * b[3]

    return inter / (areaA + areaB - inter + 1e-6)


TP = FP = FN = 0

for pred_file in os.listdir(PRED_DIR):
    if not pred_file.endswith("_pred.json"):
        continue

    set_name = pred_file.replace("_pred.json", "")
    pred_path = os.path.join(PRED_DIR, pred_file)
    gt_path = os.path.join(GT_DIR, set_name, "visible.json")

    if not os.path.exists(gt_path):
        continue

    with open(pred_path) as f:
        preds = json.load(f)

    with open(gt_path) as f:
        gts = json.load(f)

    for p in preds:
        frame = p["frame"]
        dets = p["detections"]

        gt = [x for x in gts if x["frame"] == frame]
        if not gt:
            continue

        gt = gt[0]

        if not gt["drone_present"]:
            FP += len(dets)
            continue

        if not dets:
            FN += 1
            continue

        best = max(iou(d["bbox"], gt["bbox"]) for d in dets)

        if best > IOU_THRESH:
            TP += 1
        else:
            FP += 1
            FN += 1


precision = TP / (TP + FP + 1e-6)
recall    = TP / (TP + FN + 1e-6)
f1        = 2 * precision * recall / (precision + recall + 1e-6)

print("\n--- OVERALL MOTION DETECTION PERFORMANCE ---")
print("Precision:", round(precision, 3))
print("Recall   :", round(recall, 3))
print("F1-score :", round(f1, 3))
