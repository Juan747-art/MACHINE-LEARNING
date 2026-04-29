import pandas as pd
from KMeansModel import KMeansModel

df = pd.read_csv('student_productivity.csv')

features = df[['study_hours_per_day', 'sleep_hours', 'social_media_hours', 'exercise_minutes']]

model = KMeansModel()
model.train(features, k=3)

model.save_model('kmeans_model.pkl')