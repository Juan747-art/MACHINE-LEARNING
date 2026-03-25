import pandas as pd
from sklearn.linear_model import LinearRegression

data = pd.read_csv("nvidia.csv")
data = data.dropna()

X = data.index.values.reshape(-1,1)
y = data["Close"]

model = LinearRegression()
model.fit(X, y)

def predict_price(day):
    return round(model.predict([[day]])[0], 2)