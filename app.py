from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = joblib.load("model_resampled_xgb.pkl")

@app.route('/')
def home():
    # Render the input form for prediction
    return '''
    <!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictify - Churn Prediction</title>
    <style>
        /* Global styles */
        body {
            margin: 0;
            font-family: 'Arial', sans-serif;
            color: #333;
            background-color: #f9f9f9;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 1rem 2rem;
            background-color: #1a1a1a;
            color: #fff;
            position: sticky;
            top: 0;
            z-index: 1000;
        }

        .logo {
            font-size: 1.5rem;
            font-weight: bold;
            color: #fff;
        }

        .nav-links {
            list-style: none;
            display: flex;
            gap: 1.5rem;
        }

        .nav-links li a {
            text-decoration: none;
            color: #fff;
            font-size: 1rem;
            transition: color 0.3s;
        }

        .nav-links li a:hover {
            color: #ff4081;
        }

        .cta-button {
            background-color: #ff4081;
            color: #fff;
            border: none;
            padding: 0.5rem 1rem;
            border-radius: 5px;
            font-size: 1rem;
            cursor: pointer;
            transition: background-color 0.3s;
        }

        .cta-button:hover {
            background-color: #e0356d;
        }

        /* Hero Section */
        .hero-section {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 2rem;
            background: linear-gradient(90deg, #ff4081, #ff80ab);
            color: #fff;
        }

        .hero-text h1 {
            font-size: 2.5rem;
            margin-bottom: 1rem;
        }

        .hero-text p {
            font-size: 1.25rem;
        }

        .hero-image img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
        }

        /* Form Section */
        .form-section {
            padding: 2rem;
            text-align: center;
            background-color: #fff;
            margin: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }

        .form-section h2 {
            margin-bottom: 1rem;
        }

        .form-container {
            display: flex;
            flex-direction: column;
            gap: 1rem;
            align-items: center;
        }

        .form-container input {
            width: 90%;
            max-width: 400px;
            padding: 0.75rem;
            border: 1px solid #ccc;
            border-radius: 5px;
            font-size: 1rem;
        }

        footer {
            text-align: center;
            padding: 1rem;
            background-color: #1a1a1a;
            color: #fff;
            margin-top: 2rem;
        }
    </style>
</head>
<body>
    <header class="navbar">
        <div class="logo">Predictify</div>
        <nav>
            <ul class="nav-links">
                <li><a href="#">Home</a></li>
                <li><a href="#">About</a></li>
                <li><a href="#">Features</a></li>
                <li><a href="#">Contact</a></li>
            </ul>
        </nav>
        <button class="cta-button">Get Started</button>
    </header>

    <main>
        <section class="hero-section">
            <div class="hero-text">
                <h1>Welcome to Predictify</h1>
                <p>Empowering businesses to predict customer churn with ease.</p>
            </div>
            <div class="hero-image">
                <img src="hero-image.png" alt="Data Analysis">
            </div>
        </section>

        <section class="form-section">
            <h2>Predict Customer Churn</h2>
            <form method="POST" action="/predict" class="form-container">
                <input type="number" name="credit_score" placeholder="Credit Score" required>
                <input type="text" name="gender" placeholder="Gender (1 for Male/ 0for Female)" required>
                <input type="number" name="age" placeholder="Age" required>
                <input type="number" name="tenure" placeholder="Tenure" required>
                <input type="number" name="balance" placeholder="Balance" required>
                <input type="number" name="products_number" placeholder="Number of Products" required>
                <input type="number" name="credit_card" placeholder="Has Credit Card (1 for yes/0 for no)" required>
                <input type="number" name="active_member" placeholder="Active Member (1 for yes/0 for no)" required>
                <input type="number" name="estimated_salary" placeholder="Estimated Salary" required>
                <input type="number" name="country_Germany" placeholder="are you German? (1 for yes/0 for no)" required>
                <input type="number" name="country_Spain" placeholder="are you spanish? (1 for yes/0 for no)" required>
                <input type="text" name="high_risk_age" placeholder="are you above age 40?(1 for yes/0 for no)" required>
                <input type="text" name="high_balance_risk" placeholder="Is your balance below 50000?(1 for yes/0 for no)" required>
                <input type="text" name="high_credit_score" placeholder="Is your credit score below 650?(1 for yes/ 0 for no)" required>
                <button type="submit" class="cta-button">Predict</button>
            </form>
        </section>
    </main>

    <footer>
        <p>© 2025 Predictify. All rights reserved.</p>
    </footer>
</body>
</html>
'''

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect all the input features from the form
        credit_score = float(request.form.get("credit_score"))
        gender = int(request.form.get("gender"))
        age = float(request.form.get("age"))
        tenure = int(request.form.get("tenure"))
        balance = float(request.form.get("balance"))
        products_number = int(request.form.get("products_number"))
        credit_card = int(request.form.get("credit_card"))
        active_member = int(request.form.get("active_member"))
        estimated_salary = float(request.form.get("estimated_salary"))
        country_Germany = int(request.form.get("country_Germany"))
        country_Spain = int(request.form.get("country_Spain"))
        high_risk_age = request.form.get("high_risk_age") == "True"
        high_balance_risk = request.form.get("high_balance_risk") == "True"
        high_credit_score = int(request.form.get("high_credit_score"))

        # Convert boolean values to 1 and 0
        high_risk_age = 1 if high_risk_age else 0
        high_balance_risk = 1 if high_balance_risk else 0

        # Create a feature array
        features = np.array([[credit_score, gender, age, tenure, balance, products_number,
                              credit_card, active_member, estimated_salary, country_Germany,
                              country_Spain, high_risk_age, high_balance_risk, high_credit_score]])

        # Make predictions
        prediction = model.predict(features)
        prediction_proba = model.predict_proba(features)[:, 1]

        # Interpret the prediction
        result = "Churn" if prediction[0] == 1 else "No Churn"
        probability = round(prediction_proba[0] * 100, 2)

        # Return the results
        return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Prediction Result</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    text-align: center;
                    background-color: #f4f4f4;
                    margin: 0;
                    padding: 20px;
                }}
                .result-container {{
                    background: #fff;
                    padding: 20px;
                    margin: 50px auto;
                    border-radius: 8px;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
                    width: 50%;
                }}
                h1 {{
                    color: #007BFF;
                }}
                p {{
                    font-size: 18px;
                }}
                a {{
                    color: #007BFF;
                    text-decoration: none;
                }}
            </style>
        </head>
        <body>
            <div class="result-container">
                <h1>Prediction Result</h1>
                <p>The predicted result is: <strong>{result}</strong></p>
                <p>Probability of Churn: <strong>{round(probability, 2)}%</strong></p>
                <a href="/">Go Back</a>
            </div>
        </body>
        </html>
        '''
    except Exception as e:
        # Handle errors
        return f"<h2>An error occurred:</h2><p>{str(e)}</p>"

if __name__ == "__main__":
    app.run(debug=True)
