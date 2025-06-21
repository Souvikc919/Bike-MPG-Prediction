# Auto MPG Predictor 🚗

A simple web application that predicts the fuel efficiency (MPG - miles per gallon) of a car based on its specifications.

🔮 Built with:
- Python
- Flask
- Scikit-learn
- Random Forest Regressor
- HTML / CSS frontend
- 
---

## 🚀 How it works

The app takes in key attributes of a car:
- Weight (lbs)
- Horsepower
- Displacement (cubic inches)
- Acceleration (0-60 mph in seconds)
- Model Year
- Origin (1 = USA, 2 = Europe, 3 = Japan)

It then uses a trained **Random Forest Regression model** to predict the MPG (miles per gallon).

---

## 💻 Running locally

1️⃣ Clone the repo:

```bash
git clone https://github.com/your-username/auto-mpg-predictor.git
cd auto-mpg-predictor
