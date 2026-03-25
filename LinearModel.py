import os
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.read_csv("nvidia.csv")
data = data.dropna()

X = data.index.values.reshape(-1,1)
y = data["Close"]

model = LinearRegression()
model.fit(X, y)

def predict_price(day):
    return round(model.predict([[day]])[0], 2)

def generate_graph():
    os.makedirs("static/images", exist_ok=True)

    plt.figure()
    plt.scatter(X, y)
    plt.plot(X, model.predict(X))
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("NVIDIA Stock Prediction")
    plt.savefig("static/images/graph.png")
    plt.close()