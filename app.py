from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open('cancer_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 7 features from user input
        user_inputs = [
            float(request.form['radius_mean']),
            float(request.form['texture_mean']),
            float(request.form['perimeter_mean']),
            float(request.form['area_mean']),
            float(request.form['smoothness_mean']),
            float(request.form['compactness_mean']),
            float(request.form['concavity_mean']),
        ]

        # 23 default values (hardcoded)
        prefix_values = [
            0.05, 0.18, 0.06,  # concave points_mean, symmetry_mean, fractal_dimension_mean
            0.5, 1.0, 2.5, 25.0, 0.007, 0.05, 0.06,  # radius_se to concavity_se
            0.02, 0.03, 0.005, 0.8, 0.6, 8.0, 130.0,  # concave points_se to area_worst
            0.007, 0.05, 0.06, 0.02, 0.03, 0.006  # remaining worst feature estimates
        ]

        # Combine user + prefix values = 30 features
        final_features = [np.array(user_inputs + prefix_values[:23])]  # ensure total = 30

        # Make prediction
        prediction = model.predict(final_features)[0]
        result = 'Malignant (Cancerous)' if prediction == 1 else 'Benign (Non-cancerous)'


        return render_template('index.html', prediction_text=f'Tumor is likely: {result}')

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")



if __name__ == '__main__':
    app.run(debug=True)
