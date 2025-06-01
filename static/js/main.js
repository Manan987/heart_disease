document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const formData = new FormData(e.target);
    const data = {};
    
    try {
        // Convert form data to appropriate types
        const fields = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 
            'restecg', 'thalach', 'exang', 'oldpeak', 
            'slope', 'ca', 'thal'
        ];

        // Update the type conversion logic
        fields.forEach(field => {
            const value = formData.get(field);
            if (!value && value !== '0') throw new Error(`Missing field: ${field}`);
            
            // Correct type mapping
            data[field] = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak'].includes(field)
                ? parseFloat(value)
                : parseInt(value);
        });

        const response = await fetch('http://localhost:5001/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Prediction failed');
        }

        const result = await response.json();
        
        const resultDiv = document.getElementById('result');
        const predictionText = document.getElementById('predictionText');
        
        resultDiv.classList.remove('hidden');
        
        if (result.prediction === 1) {
            predictionText.className = 'text-lg font-medium text-red-600';
            predictionText.textContent = 'High Risk of Heart Disease Detected';
        } else {
            predictionText.className = 'text-lg font-medium text-green-600';
            predictionText.textContent = 'Low Risk of Heart Disease Detected';
        }
        
        resultDiv.scrollIntoView({ behavior: 'smooth', block: 'center' });
        
    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Prediction error:', error);
    }
});

// Add training functionality
document.getElementById('trainButton').addEventListener('click', async () => {
    const trainButton = document.getElementById('trainButton');
    const trainingStatus = document.getElementById('trainingStatus');
    
    try {
        // Disable button and show status
        trainButton.disabled = true;
        trainingStatus.classList.remove('hidden');
        
        const response = await fetch('http://localhost:5001/train', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Training failed');
        }

        const result = await response.json();
        alert('Model trained successfully!');
        
    } catch (error) {
        alert('Error: ' + error.message);
        console.error('Training error:', error);
    } finally {
        // Re-enable button and hide status
        trainButton.disabled = false;
        trainingStatus.classList.add('hidden');
    }
});
