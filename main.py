from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# Load the trained XGBoost model
model = joblib.load('attrition.sav')  # Ensure the model is saved as a .pkl file

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract input features from the form
    try:
        features = [
            float(request.form['satisfaction_level']),
            float(request.form['last_evaluation']),
            int(request.form['number_project']),
            int(request.form['average_monthly_hours']),
            int(request.form['time_spend_company']),
            int(request.form['work_accident']),
            int(request.form['promotion_last_5years']),
            int(request.form['low']),
            int(request.form['medium'])
        ]
        # Convert to numpy array and reshape for prediction
        features_array = np.array(features).reshape(1, -1)
        prediction = model.predict(features_array)
        result = 'Employee will leave' if prediction[0] == 1 else 'Employee will stay'
    except Exception as e:
        result = f"Error: {e}"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
