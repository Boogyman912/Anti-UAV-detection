import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

X = np.load("training/rgb_feats.npy", allow_pickle=True)

print("Loaded RGB features:", X.shape)

if len(X) == 0:
    raise ValueError("RGB feature array is empty. Training cannot continue.")

model = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    random_state=42
)

model.fit(X)

joblib.dump(model, "models/iforest_rgb.pkl")

print("RGB Isolation Forest trained successfully.")
