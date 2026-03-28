# Student Performance Prediction System
### Final Year Academic Project | Educational Data Mining with Machine Learning

---

## Project Overview

A complete ML web application that predicts student final scores based on 7 academic features using trained scikit-learn models, with a Flask backend and a dark-themed Bootstrap frontend.

---

## Technology Stack

| Layer      | Technology                          |
|------------|--------------------------------------|
| Language   | Python 3.10+                         |
| ML         | scikit-learn, pandas, numpy          |
| Viz        | matplotlib, seaborn                  |
| Backend    | Flask                                |
| Frontend   | HTML5, CSS3, Bootstrap 5, Chart.js   |
| Model Save | joblib                               |

---

## Project Structure

```
StudentPerformancePrediction/
│
├── app.py                  ← Flask web application
├── train_model.py          ← Dataset generation + model training
├── student_performance.csv ← Synthetic dataset (1200 rows)
├── model.pkl               ← Saved best ML model
├── requirements.txt
│
├── static/
│   ├── css/style.css
│   └── images/             ← EDA graphs (auto-generated)
│
└── templates/
    ├── base.html
    ├── index.html
    ├── dashboard.html
    ├── predict.html
    └── result.html
```

---

## Setup Instructions

### Step 1 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Train the Model (generates dataset + graphs)
```bash
python train_model.py
```

### Step 3 — Run the Web App
```bash
python app.py
```

### Step 4 — Open in Browser
```
http://127.0.0.1:5000
```

---

## Dataset Features

| Feature                | Range    | Description                        |
|------------------------|----------|------------------------------------|
| study_hours            | 1–10     | Daily study hours                  |
| attendance             | 50–100   | Class attendance (%)               |
| assignments_completed  | 0–10     | Assignments completed out of 10    |
| previous_grade         | 40–100   | Previous semester grade            |
| participation          | 1–10     | Classroom participation score      |
| sleep_hours            | 4–10     | Average nightly sleep              |
| internet_usage         | 1–8      | Recreational internet hrs/day      |
| **final_score**        | 0–100    | **Target variable**                |

---

## Model Performance (Test Set)

| Model             | R² Score | MAE   | MSE    |
|-------------------|----------|-------|--------|
| Linear Regression | 0.8957   | 3.31  | 18.07  |
| Decision Tree     | 0.6959   | 5.64  | 52.70  |
| Random Forest     | 0.8592   | 3.88  | 24.39  |

✅ **Best model: Linear Regression** (saved as `model.pkl`)

---

## Web App Routes

| Route       | Description            |
|-------------|------------------------|
| `/`         | Home page + statistics |
| `/dashboard`| EDA graphs + metrics   |
| `/predict`  | Prediction input form  |
| `/result`   | Prediction output page |

---

*Built for academic submission — Python · Flask · scikit-learn*
