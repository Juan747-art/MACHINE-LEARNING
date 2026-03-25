from LogisticRegressionModel import LogisticRegressionModel

dataset_path = "datos.csv"

model = LogisticRegressionModel(dataset_path)

print("🔄 Training model...")
model.train()

print("💾 Saving model...")
model.save_model("logistic_model.pkl")

print("✅ Model ready!")