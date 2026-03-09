from flask import Flask, render_template, request, redirect, url_for
import random
import LinearRegression

app = Flask(__name__)

USERNAME = "admin"
PASSWORD = "1234"


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
    
@app.route('/LinearRegression', methods=["GET","POST"])
def calculateGrade():
   if request.method=="POST":
    hours = float(request.form["hours"]) 
       
   result = LinearRegression.calculateGrade(20)
   return render_template("LinearRegressionGrade",result=result)


@app.route('/dashboard')
def dashboard():

    # Simulated ML prediction
    prediction = random.randint(60,95)

    return render_template("dashboard.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
