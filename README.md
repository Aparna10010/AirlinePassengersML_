
# ✈️ Airline Customer Satisfaction Prediction

A machine learning project focused on predicting customer satisfaction based on flight service features. This project deals with an **imbalanced classification problem**, applies **SMOTE**, performs **feature selection**, and uses **model tuning** to improve performance.

---

## 📌 Problem Statement

The goal is to predict whether a customer is **satisfied or not** based on various service and flight-related features. The dataset is imbalanced — the number of satisfied vs unsatisfied customers is skewed.

---

## 🧪 Dataset Overview

- Source: [Kaggle / Airline Satisfaction Dataset](https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction)
- Total rows: ~100k+
- Features include: `Inflight_wifi_service`, `Flight_Distance`, `Online_boarding`, `Seat_comfort`, etc.
- Target variable: `Satisfaction (0 = Not Satisfied, 1 = Satisfied)`

---

## 📊 Summary Slides

📄 [View Project Summary Slides (PDF)](./Project_Summary.pdf)  

---

## 🧠 Key Steps

- ✔️ Data Cleaning & Feature Engineering
- ✔️ Exploratory Data Analysis & Visualization
- ✔️ One-Hot Encoding
- ✔️ Feature Selection using RFE
- ✔️ Handling Imbalance with **SMOTE**
- ✔️ Model Training: Logistic Regression & Random Forest
- ✔️ Hyperparameter Tuning using **RandomizedSearchCV**
- ✔️ Evaluation with Accuracy, F1, AUPRC
- ✔️ Pipeline creation & model serialization

---

## ⚙️ Model Performance

### ✅ Before SMOTE (Logistic Regression):
- Accuracy: `93%`
- F1 (Class 1): `0.73`
- AUPRC: `~0.79`

### ✅ After SMOTE:
- Accuracy: `95%`
- F1 (Class 1): `0.84`
- AUPRC: `0.926`

### 🌲 Final Random Forest Model (after SMOTE + Tuning):
- Accuracy: `95.2%`
- F1 (Class 1): `0.84`
- AUPRC: `0.91`

---

## 🔥 Visual Insights

| Feature Importance | Confusion Matrix | AUPRC Curve |
|--------------------|------------------|-------------|
| ✅ Online Boarding | ✅ Clear Class Separation | ✅ High precision under recall |

---

## 🧾 Tech Stack

- Python, Pandas, NumPy, Matplotlib, Seaborn
- scikit-learn, imblearn (SMOTE)
- RandomForest, LogisticRegression
- Pickle (for saving model pipeline)

---

## 💾 Model Deployment

You can load the saved pipeline directly and use it for predictions:

```python
import pickle
with open("final_pipeline_rf.pkl", "rb") as f:
    model = pickle.load(f)
    
predictions = model.predict(X_test)---


##   Future Work
Deploy using Streamlit or Flask

Try ensemble models or XGBoost

Test on real-time data


## Contact
Email : aparnasharma10010@gmail.com
