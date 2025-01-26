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
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-100 font-sans">
    <!-- Header -->
    <header class="bg-gradient-to-r from-pink-500 to-red-500 text-white shadow-lg">
        <div class="container mx-auto flex justify-between items-center py-4 px-6">
            <div class="text-2xl font-bold">Predictify</div>
            <nav>
                <ul class="flex space-x-6 text-lg">
                    <li><a href="#" class="hover:text-gray-200">Home</a></li>
                    <li><a href="#" class="hover:text-gray-200">About</a></li>
                    <li><a href="#" class="hover:text-gray-200">Features</a></li>
                    <li><a href="#" class="hover:text-gray-200">Contact</a></li>
                </ul>
            </nav>
            <button class="bg-white text-pink-500 px-4 py-2 rounded-full shadow-md hover:bg-pink-100">
                Get Started
            </button>
        </div>
    </header>

    <!-- Hero Section -->
    <section class="hero-section bg-gradient-to-r from-purple-400 to-blue-500 text-white py-20">
        <div class="container mx-auto flex flex-col lg:flex-row items-center px-6 space-y-6 lg:space-y-0 lg:space-x-12">
            <div class="hero-text text-center lg:text-left max-w-lg">
                <h1 class="text-4xl font-bold mb-4">Welcome to Predictify</h1>
                <p class="text-lg">Empowering businesses to predict customer churn with ease.</p>
                <div class="hero-image">
                    <img src="predtify_icon.png" alt="">
                </div>
            </div>
        </div>
    </section>

    <!-- Form Section -->
    <section class="form-section bg-white py-12">
        <div class="container mx-auto text-center px-6">
            <h2 class="text-3xl font-bold mb-6">Predict Customer Churn</h2>
            <form method="POST" action="/predict" class="form-container flex flex-col items-center gap-4 max-w-lg mx-auto">
                <input type="number" name="credit_score" placeholder="Credit Score" required class="w-full p-3 border border-gray-300 rounded-md">
                <input type="text" name="gender" placeholder="Gender (1 for Male/ 0 for Female)" required class="w-full p-3 border border-gray-300 rounded-md">
                <input type="number" name="age" placeholder="Age" required class="w-full p-3 border border-gray-300 rounded-md">
                <input type="number" name="tenure" placeholder="Tenure" required class="w-full p-3 border border-gray-300 rounded-md">
                <input type="number" name="balance" placeholder="Balance" required class="w-full p-3 border border-gray-300 rounded-md">
                <input type="number" name="products_number" placeholder="Number of Products" required class="w-full p-3 border border-gray-300 rounded-md">
                <input type="number" name="credit_card" placeholder="Has Credit Card (1 for yes/0 for no)" required class="w-full p-3 border border-gray-300 rounded-md">
                <input type="number" name="active_member" placeholder="Active Member (1 for yes/0 for no)" required class="w-full p-3 border border-gray-300 rounded-md">
                <input type="number" name="estimated_salary" placeholder="Estimated Salary" required class="w-full p-3 border border-gray-300 rounded-md">
                <input type="number" name="country_Germany" placeholder="Are you German? (1 for yes/0 for no)" required class="w-full p-3 border border-gray-300 rounded-md">
                <input type="number" name="country_Spain" placeholder="Are you Spanish? (1 for yes/0 for no)" required class="w-full p-3 border border-gray-300 rounded-md">
                <input type="text" name="high_risk_age" placeholder="Are you above age 40? (1 for yes/0 for no)" required class="w-full p-3 border border-gray-300 rounded-md">
                <input type="text" name="high_balance_risk" placeholder="Is your balance below 50,000? (1 for yes/0 for no)" required class="w-full p-3 border border-gray-300 rounded-md">
                <input type="text" name="high_credit_score" placeholder="Is your credit score below 650? (1 for yes/0 for no)" required class="w-full p-3 border border-gray-300 rounded-md">
                <button type="submit" class="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-3 rounded-md hover:shadow-lg hover:scale-105 transition">
                    Predict
                </button>
            </form>
        </div>
    </section>

    <!-- Footer -->
    <footer class="bg-gray-800 text-white py-6">
        <div class="container mx-auto text-center">
            <p class="text-sm">© 2025 Predictify. All rights reserved.</p>
        </div>
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
