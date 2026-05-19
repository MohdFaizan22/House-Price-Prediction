
from flask import Flask, request, jsonify, render_template_string
import joblib
import numpy as np


model = joblib.load('house_price_model.pkl')

app = Flask(__name__)


HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Predictor</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #121212;
            color: white;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }

        .container {
            background: #1e1e1e;
            padding: 30px;
            border-radius: 12px;
            width: 350px;
            box-shadow: 0 0 15px rgba(0,0,0,0.4);
        }

        h1 {
            text-align: center;
            margin-bottom: 25px;
        }

        input {
            width: 100%;
            padding: 10px;
            margin-top: 10px;
            border-radius: 6px;
            border: none;
            font-size: 16px;
        }

        button {
            width: 100%;
            padding: 12px;
            margin-top: 20px;
            background: #00b894;
            color: white;
            border: none;
            border-radius: 6px;
            font-size: 16px;
            cursor: pointer;
        }

        button:hover {
            background: #00a383;
        }

        .result {
            margin-top: 20px;
            text-align: center;
            font-size: 20px;
            font-weight: bold;
            color: #00ff99;
        }
    </style>
</head>
<body>

<div class="container">
    <h1>House Price Predictor</h1>

    <form method="POST">
        <input type="number" name="area" placeholder="Area in sqft" required>

        <input type="number" name="bath" placeholder="Bathrooms" required>

        <input type="number" name="bhk" placeholder="BHK" required>

        <input type="number" step="0.01" name="location" placeholder="Location Avg Price" required>

        <button type="submit">Predict Price</button>
    </form>

    {% if prediction %}
        <div class="result">
            Predicted Price: ₹ {{ prediction }} Lakhs
        </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def home():

    prediction = None

    if request.method == 'POST':

        area = float(request.form['area'])
        bath = float(request.form['bath'])
        bhk = float(request.form['bhk'])
        location = float(request.form['location'])

        features = np.array([[area, bath, bhk, location]])

        pred = model.predict(features)

        prediction = round(np.exp(pred[0]), 2)

    return render_template_string(HTML_PAGE, prediction=prediction)


if __name__ == '__main__':
    app.run(debug=True)

