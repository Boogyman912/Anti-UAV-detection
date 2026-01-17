import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import cv2
import glob
import numpy as np
from pipeline.roi_features import extract_features

files = glob.glob("data/samples_ir/*")

print("Found IR ROI files:", len(files))

feats = []

for f in files:
    img = cv2.imread(f, 0)
    if img is None:
        print("Could not read:", f)
        continue

    feat = extract_features(img)
    feats.append(feat)

feats = np.array(feats)

print("Feature array shape:", feats.shape)

np.save("training/ir_feats.npy", feats)
print("IR feature dataset saved.")
