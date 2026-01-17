import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

X = np.load("training/ir_feats.npy", allow_pickle=True)

print("Loaded IR features:", X.shape)

if len(X) == 0:
    raise ValueError("IR feature array is empty. Training cannot continue.")

model = IsolationForest(
    n_estimators=120,
    contamination=0.08,
    random_state=42
)

model.fit(X)

joblib.dump(model, "models/iforest_ir.pkl")

print("IR Isolation Forest trained successfully.")
