from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load model and preprocessors once at startup
model = pickle.load(open('model.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    gender = request.form['gender']
    region_category = request.form['region_category']
    membership_category = request.form['membership_category']
    age = int(request.form['age'])
    days_since_joining = int(request.form['days_since_joining'])
    days_since_last_visit = int(request.form['days_since_last_visit'])

    # Prepare data in the format your model expects
    # Example: create a dataframe or numpy array, then preprocess
    
    # Here is a generic placeholder; adapt to your preprocessing pipeline:
    input_data = np.array([[gender, region_category, membership_category, age, days_since_joining, days_since_last_visit]])
    
    # Apply preprocessor & scaler if needed, e.g.:
    input_processed = preprocessor.transform(input_data)
    input_scaled = scaler.transform(input_processed)

    # Predict
    prediction = model.predict(input_scaled)

    # Create output message
    prediction_text = "Customer is likely to churn." if prediction[0] == 1 else "Customer is unlikely to churn."

    return render_template('index.html', prediction_text=prediction_text)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
