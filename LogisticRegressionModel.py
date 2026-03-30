import numpy as np
import joblib

class LogisticRegressionModel:

    def __init__(self):
        self.model = None

    def load_model(self, filename):
        self.model = joblib.load(filename)

    def predict_productivity(self, study_hours, sleep_hours, phone_usage, social_media, focus_score, attendance):
        data = np.array([[study_hours, sleep_hours, phone_usage, social_media, focus_score, attendance]])
        prediction = self.model.predict(data)[0]
        probability = self.model.predict_proba(data)[0][1]
        return prediction, probability