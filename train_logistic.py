import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

df = pd.read_csv("student_productivity.csv")

df.columns = df.columns.str.strip()

df["productive"] = df["productivity_score"].apply(lambda x: 1 if x >= 70 else 0)

X = df[["study_hours_per_day","sleep_hours","phone_usage_hours","social_media_hours","focus_score","attendance_percentage"]]
y = df["productive"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

cm = confusion_matrix(y_test,y_pred)
acc = accuracy_score(y_test,y_pred)
prec = precision_score(y_test,y_pred)
rec = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
auc = roc_auc_score(y_test,y_prob)

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.savefig("static/images/confusion_matrix.png")

fpr,tpr,_ = roc_curve(y_test,y_prob)
plt.figure()
plt.plot(fpr,tpr)
plt.title("ROC Curve")
plt.savefig("static/images/roc_curve.png")

pickle.dump(model,open("logistic_model.pkl","wb"))

print("Accuracy:",acc)
print("Precision:",prec)
print("Recall:",rec)
print("F1-score:",f1)
print("AUC:",auc)