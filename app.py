from flask import Flask, request, jsonify, render_template
import numpy as np
import joblib
from flask_cors import CORS
from train_model import load_and_preprocess_data, main as train_model_main

app = Flask(__name__)
CORS(app)

# Load the trained model
model = joblib.load('model/heart_disease_model.pkl')

# Define valid ranges for numerical fields
valid_ranges = {
    'age': (20, 100),
    'sex': (0, 1),
    'cp': (0, 3),
    'trestbps': (90, 200),
    'chol': (100, 600),
    'fbs': (0, 1),
    'restecg': (0, 2),
    'thalach': (60, 220),
    'exang': (0, 1),
    'oldpeak': (0, 6.2),
    'slope': (0, 2),
    'ca': (0, 4),
    'thal': (0, 3)
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        
        # Validate all required fields are present
        missing_fields = [field for field in valid_ranges.keys() if field not in data]
        if missing_fields:
            return jsonify({'error': f'Missing fields: {", ".join(missing_fields)}'}), 400
        
        # Validate numerical ranges
        for field, (min_val, max_val) in valid_ranges.items():
            value = float(data[field])
            if not (min_val <= value <= max_val):
                return jsonify({'error': f'{field} must be between {min_val} and {max_val}'}), 400
        
        # Prepare features for prediction
        # Add feature order validation
        features = [float(data[field]) for field in valid_ranges.keys()]
        
        # Must match training order: ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
        # 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        assert len(features) == 13, "Feature count mismatch"
        
        # Make prediction
        prediction = int(model.predict([features])[0])
        
        return jsonify({'prediction': prediction})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/train', methods=['POST'])
def train():
    try:
        # Train the model
        train_model_main()
        return jsonify({'message': 'Model trained successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)

 
 