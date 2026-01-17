import joblib

class IForestRGB:
    def __init__(self, path):
        self.model = joblib.load(path)

    def score(self, feat):
        return float(-self.model.decision_function([feat])[0])

class IForestIR:
    def __init__(self, path):
        self.model = joblib.load(path)

    def score(self, feat):
        return float(-self.model.decision_function([feat])[0])
