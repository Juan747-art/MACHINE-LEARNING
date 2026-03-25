import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib


class LogisticRegressionModel:

    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.model = None
        self.scaler = None

    
    def load_data(self):
        data = pd.read_csv(self.csv_path)

        X = data[[
            "edad",
            "ingreso_mensual",
            "visitas_web_mes",
            "tiempo_sitio_min",
            "compras_previas",
            "descuento_usado"
        ]]

        y = data["target"]

        return train_test_split(X, y, test_size=0.2, random_state=42)

    def train(self):
        X_train, X_test, y_train, y_test = self.load_data()

        
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X_train, y_train)

        
        y_pred = self.model.predict(X_test)

        print("🔹 Accuracy:", accuracy_score(y_test, y_pred))
        print("\n🔹 Classification Report:\n")
        print(classification_report(y_test, y_pred))

    
    def save_model(self, filename="logistic_model.pkl"):
        if self.model is not None and self.scaler is not None:
            joblib.dump((self.model, self.scaler), filename)
            print(f"✅ Model saved as {filename}")
        else:
            print("❌ No trained model to save")

    
    def load_model(self, filename="logistic_model.pkl"):
        self.model, self.scaler = joblib.load(filename)
        print("✅ Model loaded successfully")

    
    def predict(self, data):
        if self.model is None:
            raise Exception("Model not trained or loaded")

        data = self.scaler.transform(data)
        return self.model.predict(data)[0]

    
    def predict_probability(self, data):
        if self.model is None:
            raise Exception("Model not trained or loaded")

        data = self.scaler.transform(data)
        return self.model.predict_proba(data)[0][1]