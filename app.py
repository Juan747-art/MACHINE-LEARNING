from flask import Flask, render_template, request, redirect, url_for
import random
import LinearRegression
from LogisticRegressionModel import LogisticRegressionModel
import LinearModel
from DecisionTreeModel import DecisionTreeModel
import matplotlib.pyplot as plt
import pandas as pd
from KMeansModel import KMeansModel

app = Flask(__name__)

USERNAME = "admin"
PASSWORD = "1234"

logistic_model = LogisticRegressionModel()
logistic_model.load_model("logistic_model.pkl")
decision_model = DecisionTreeModel()
decision_model.load_model("decision_tree_model.pkl")

@app.route("/decision_tree_application")
def decision_tree_application():
    return render_template("decision_tree_application.html")

@app.route("/predict_decision_tree", methods=["POST"])
def predict_decision_tree():

    study = float(request.form["study_hours"])
    sleep = float(request.form["sleep_hours"])
    phone = float(request.form["phone_usage"])
    social = float(request.form["social_media"])
    focus = float(request.form["focus_score"])
    attendance = float(request.form["attendance"])

    result = decision_model.predict_productivity(study, sleep, phone, social, focus, attendance)

    return render_template("decision_tree_application.html", prediction=result)

@app.route('/')
def login_page():
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    if username == USERNAME and password == PASSWORD:
        return redirect(url_for('dashboard'))
    else:
        return "Invalid username or password"
@app.route('/application', methods=["GET", "POST"])
def application():
    result = None

    if request.method == "POST":
        day = float(request.form["day"])
        result = LinearModel.predict_price(day)

    return render_template("application.html", result=result)
@app.route('/linear-regression')
def linear_regression():
    LinearModel.generate_graph()
    return render_template("linear_regression.html")

@app.route('/dashboard')
def dashboard():
    prediction = random.randint(60, 95)
    return render_template("dashboard.html", prediction=prediction)

@app.route('/ml-use-cases')
def ml_use_cases():
    return render_template('use_cases.html')

@app.route('/use-case-1')
def use_case_1():
    return render_template('use_case_1.html')

@app.route('/use-case-2')
def use_case_2():
    return render_template('use_case_2.html')

@app.route('/use-case-3')
def use_case_3():
    return render_template('use_case_3.html')

@app.route('/use-case-4')
def use_case_4():
    return render_template('use_case_4.html')
@app.route('/linear-concepts')
def linear_concepts():
    return render_template('linear_concepts.html')

@app.route("/logistic_application")
def logistic_application():
    return render_template("logistic_application.html")
import pickle
import numpy as np

model = pickle.load(open("logistic_model.pkl","rb"))

@app.route("/predict_logistic", methods=["POST"])
def predict_logistic():

    study = float(request.form["study_hours"])
    sleep = float(request.form["sleep_hours"])
    phone = float(request.form["phone_usage"])
    social = float(request.form["social_media"])
    focus = float(request.form["focus_score"])
    attendance = float(request.form["attendance"])

    prediction, probability = logistic_model.predict_productivity(study, sleep, phone, social, focus, attendance)

    if prediction == 1:
        result = "Productive student"
    else:
        result = "Low productivity student"

    return render_template("logistic_application.html", prediction=result, probability=round(probability,2))

@app.route('/kmeans_application')
def kmeans_application():
    df = pd.read_csv('student_productivity.csv')

    features = df[['study_hours_per_day', 'sleep_hours', 'social_media_hours', 'exercise_minutes']]

    model = KMeansModel()
    model.load_model('kmeans_model.pkl')

    clusters = model.predict(features)
    df['cluster'] = clusters

    centroids = model.get_centroids()

    plt.figure()
    plt.scatter(df['study_hours_per_day'], df['sleep_hours'], c=clusters)
    plt.scatter(centroids[:,0], centroids[:,1], marker='x')

    plt.xlabel('Study Hours Per Day')
    plt.ylabel('Sleep Hours')

    plt.savefig('static/images/kmeans_plot.png')

    summary = df.groupby('cluster')[[
        'study_hours_per_day',
        'sleep_hours',
        'social_media_hours',
        'exercise_minutes'
    ]].mean()

    return render_template(
        'kmeans_application.html',
        tables=[df.head(20).to_html(classes='table table-striped')],
        summary=summary.to_html(classes='table table-bordered'),
        centroids=centroids
    )

@app.route('/predict-exercise', methods=["GET", "POST"])
def predict_exercise():
    result = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        result = LinearRegression.calculateGrade(hours)

    return render_template("predict.html", result=result)

@app.route("/logistic_concepts")
def logistic_concepts():
    return render_template("logistic_concepts.html")

@app.route('/unsupervised')
def unsupervised():
    return render_template('unsupervised.html')

@app.route('/kmeans_concepts')
def kmeans_concepts():
    return render_template('kmeans_concepts.html')

@app.route('/kmeans_manual')
def kmeans_manual():
    return render_template('kmeans_manual.html')

@app.route("/classification_model_concepts")
def classification_model_concepts():
    return render_template("classification_model_concepts.html")



    return render_template(
        'predict_logistic.html',
        result=result,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)