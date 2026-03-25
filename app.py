from flask import Flask, render_template, request, redirect, url_for
import random
import LinearRegression
from LogisticRegressionModel import LogisticRegressionModel
import LinearModel

app = Flask(__name__)

USERNAME = "admin"
PASSWORD = "1234"

logistic_model = LogisticRegressionModel("datos.csv")
logistic_model.load_model("logistic_model.pkl")

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

@app.route('/linear-exercise', methods=["GET", "POST"])
def linear_exercise():
    result = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        result = LinearRegression.calculateGrade(hours)

    return render_template("LinearRegressionGrade.html", result=result)

@app.route('/predict-exercise', methods=["GET", "POST"])
def predict_exercise():
    result = None

    if request.method == "POST":
        hours = float(request.form["hours"])
        result = LinearRegression.calculateGrade(hours)

    return render_template("predict.html", result=result)

@app.route('/predict-logistic', methods=['GET', 'POST'])
def predict_logistic():
    result = None
    probability = None

    if request.method == "POST":
        age = float(request.form['edad'])
        income = float(request.form['ingreso'])
        visits = float(request.form['visitas'])
        time = float(request.form['tiempo'])
        purchases = float(request.form['compras'])
        discount = float(request.form['descuento'])

        data = [[age, income, visits, time, purchases, discount]]

        prediction = logistic_model.predict(data)
        prob = logistic_model.predict_probability(data)

        result = "Will Buy" if prediction == 1 else "Will Not Buy"
        probability = f"{prob*100:.2f}%"

    return render_template(
        'predict_logistic.html',
        result=result,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)