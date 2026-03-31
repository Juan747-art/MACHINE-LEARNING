import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

df = pd.read_csv("student_productivity.csv")

df.columns = df.columns.str.strip()

def classify_productivity(score):
    if score < 40:
        return 0
    elif score < 70:
        return 1
    else:
        return 2

df["productivity_class"] = df["productivity_score"].apply(classify_productivity)

X = df[[
    "study_hours_per_day",
    "sleep_hours",
    "phone_usage_hours",
    "social_media_hours",
    "focus_score",
    "attendance_percentage"
]]

y = df["productivity_class"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier(max_depth=5)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred,average="weighted")
rec = recall_score(y_test,y_pred,average="weighted")
f1 = f1_score(y_test,y_pred,average="weighted")

plt.figure()
plt.imshow(cm)
plt.title("Decision Tree Confusion Matrix")
plt.savefig("static/images/dt_confusion_matrix.png")

joblib.dump(model,"decision_tree_model.pkl")

print("Accuracy:",acc)
print("Precision:",prec)
print("Recall:",rec)
print("F1-score:",f1)