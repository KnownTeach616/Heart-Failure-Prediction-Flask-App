# Import necessary libraries for the Flask web application.
import os
from flask import Flask, request, jsonify, render_template
import pickle # Used for loading scikit-learn models and scalers.
import numpy as np
import pandas as pd # Used for creating DataFrame from input for consistent scaling.

# Initialize the Flask application.
app = Flask(__name__)

# --- Model and Scaler Loading ---
# This section loads the pre-trained machine learning model and the StandardScaler.
# These objects are essential for making predictions on new, unscaled data.

try:
    # Define the paths for the model and scaler files.
    # Ensure these paths are correct relative to where app.py is executed.
    model_path = os.path.join(os.path.dirname(__file__), 'gradient_boosting_model.pkl')
    scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')

    # Load the trained Gradient Boosting model.
    # The 'rb' mode opens the file in binary read mode.
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    print(f"Model loaded successfully from {model_path}")

    # Load the fitted StandardScaler.
    # This scaler must be the same one used during model training for consistent preprocessing.
    with open(scaler_path, 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    print(f"Scaler loaded successfully from {scaler_path}")

except Exception as e:
    # Log any errors encountered during model or scaler loading.
    # This is critical for debugging deployment issues.
    print(f"Error loading model or scaler: {e}")
    model = None # Set model to None to indicate loading failure.
    scaler = None # Set scaler to None to indicate loading failure.

# --- Routes and API Endpoints ---
# This section defines the web routes for the Flask application.

@app.route('/')
def home():
    """
    Renders the main home page of the application.
    This typically serves the index.html file to the user.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles POST requests for making predictions.
    It expects input data from a web form, processes it, and returns a prediction.
    """
    if model is None or scaler is None:
        # Return an error if the model or scaler failed to load at startup.
        return jsonify({'error': 'Model or scaler not loaded. Please check server logs.'}), 500

    try:
        # Extract features from the incoming form data.
        # Ensure that the names of the form fields in index.html match these keys.
        features = [
            float(request.form['age']),
            float(request.form['anaemia']),
            float(request.form['creatinine_phosphokinase']),
            float(request.form['diabetes']),
            float(request.form['ejection_fraction']),
            float(request.form['high_blood_pressure']),
            float(request.form['platelets']),
            float(request.form['serum_creatinine']),
            float(request.form['serum_sodium']),
            float(request.form['sex']),
            float(request.form['smoking']),
            float(request.form['time'])
        ]

        # Convert the list of features into a NumPy array.
        # Reshape to (1, -1) to represent a single sample with multiple features.
        input_array = np.array(features).reshape(1, -1)

        # Create a Pandas DataFrame from the input array.
        # This is crucial because the scaler was fitted on a DataFrame and expects feature names or a consistent order.
        # Ensure these column names match the order of features used during model training.
        feature_names = [
            'age', 'anaemia', 'creatinine_phosphokinase', 'diabetes',
            'ejection_fraction', 'high_blood_pressure', 'platelets',
            'serum_creatinine', 'serum_sodium', 'sex', 'smoking', 'time'
        ]
        input_df = pd.DataFrame(input_array, columns=feature_names)


        # Scale the input features using the loaded StandardScaler.
        # This step is vital for consistent preprocessing, as the model was trained on scaled data.
        scaled_features = scaler.transform(input_df)

        # Make a prediction using the loaded Gradient Boosting model.
        # .predict() returns the predicted class label (0 or 1).
        prediction_class = model.predict(scaled_features)[0]

        # Optionally, get the probability of the positive class (DEATH_EVENT = 1).
        # .predict_proba() returns probabilities for both classes [prob_class_0, prob_class_1].
        prediction_proba = model.predict_proba(scaled_features)[0][1]

        # Determine the human-readable result.
        if prediction_class == 1:
            result = 'High risk of Heart Failure Event'
        else:
            result = 'Low risk of Heart Failure Event'

        # Return the prediction and probability as a JSON response.
        # This makes it easy for the frontend (JavaScript) to process the result.
        return jsonify({'prediction': result, 'probability': float(prediction_proba)})

    except Exception as e:
        # Catch and log any errors during the prediction process.
        # Provide a user-friendly error message without exposing sensitive details.
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'An error occurred during prediction. Please check your input.'}), 400

# --- Application Entry Point ---
# This ensures the Flask development server runs when the script is executed directly.
if __name__ == '__main__':
    # app.run(debug=True) enables debug mode, which provides detailed error messages and
    # automatically reloads the server on code changes. Set to False in production.
    app.run(debug=True)