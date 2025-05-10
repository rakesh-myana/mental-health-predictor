from flask import Flask, render_template, request
import pickle
import numpy as np

# Create the Flask app instance
app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values
        role = request.form['role']
        diet = request.form['diet']
        suicidal = request.form['suicidal']
        age = float(request.form['age'])
        hours = float(request.form['hours'])
        finance = float(request.form['finance'])
        pressure = float(request.form['pressure'])
        satisfaction = float(request.form['satisfaction'])

        # Encode categorical values (same as training time)
        role_encoded = 0 if role == 'Student' else 1
        diet_map = {'Healthy': 0, 'Moderate': 1, 'Unhealthy': 2}
        suicidal_encoded = 1 if suicidal == 'Yes' else 0

        # Prepare feature list
        features = [
            age,
            role_encoded,
            diet_map[diet],
            suicidal_encoded,
            hours,
            finance,
            pressure,
            satisfaction
        ]

        # Convert features to numpy array for prediction
        input_data = np.array(features).reshape(1, -1)

        # Predict depression status
        prediction = model.predict(input_data)[0]
        result = "Depressed 😔" if prediction == 1 else "Not Depressed 🙂"

    except Exception as e:
        result = f"Error: {str(e)}"

    # Return result to HTML template
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
