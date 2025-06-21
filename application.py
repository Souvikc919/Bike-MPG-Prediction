from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        weight = float(request.form['weight'])
        horsepower = float(request.form['horsepower'])
        displacement = float(request.form['displacement'])
        acceleration = float(request.form['acceleration'])
        model_year = float(request.form['model_year'])
        origin = float(request.form['origin'])

        input_data = np.array([[weight, horsepower, displacement, acceleration, model_year, origin]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        mpg = round(prediction[0], 2)

        return render_template('result.html', mpg=mpg)
    
    except Exception as e:
        return render_template('result.html', mpg=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True)
