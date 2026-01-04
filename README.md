# 🌸 Iris Flower Intelligence  
### 🚀 A Production-Style Machine Learning Dashboard

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange)
![Status](https://img.shields.io/badge/Status-Completed-success)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🔗 Live Demo
📽 **Demo Video:** *(Optional – YouTube / Drive link)*

---

## 📌 Project Overview

**Iris Flower Intelligence** is a **production-style machine learning dashboard** built to classify Iris flower species based on sepal and petal measurements.

Unlike basic ML demos, this project focuses on **end-to-end ML product development**, including:
- model comparison
- analytics
- explainability
- modern UI/UX
- user-friendly reporting

This project demonstrates how a **classic ML problem** can be transformed into a **real-world analytical application**.

---

## ✨ Key Features

### 🤖 Machine Learning
- Multi-class **classification**
- Models implemented:
  - **K-Nearest Neighbors (KNN)**
  - **Logistic Regression**
- Probability-based predictions
- Model accuracy comparison
- Confusion matrix evaluation

---

### 📊 Analytics Dashboard
- Prediction probability distribution
- Confusion matrix visualization
- Model performance metrics
- Accuracy-based model recommendation

---

### 🧠 Explainability (Stable & Interpretable)
- Feature importance using **Logistic Regression coefficients**
- Explains **which input features influence predictions**
- No unstable SHAP dependencies (production-safe approach)

---

### 🎨 User Experience
- Modern **tab-based UI**
  - 🏠 Overview  
  - 📊 Analytics  
  - 🧠 Explainability  
  - 📸 Gallery
- Clean control-dock sidebar
- Mobile-friendly layout
- Visual feedback with flower images
- Prediction history tracking
- CSV export functionality

---

### 📥 Export & Reporting
- Download prediction history as **CSV**
- Useful for:
  - analysis
  - reporting
  - audit trails

---
## 🗂 Project Structure
```
iris-flower-intelligence/
│
├── app.py # Streamlit dashboard
├── model.py # ML training & loading logic
├── README.md # Project documentation
├── requirements.txt # Dependencies
├── images/ # Flower images
│ ├── setosa.jpg
│ ├── versicolor.jpg
│ └── virginica.jpg
└── saved_models/ # Trained models (gitignored)
```

---

## 🛠 Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python" />
  <img src="https://img.shields.io/badge/Streamlit-App-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn" />
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-purple?logo=pandas" />
  <img src="https://img.shields.io/badge/NumPy-Numerical%20Computing-blue?logo=numpy" />
  <img src="https://img.shields.io/badge/Joblib-Model%20Persistence-green" />
</p>


---

## ⚙️ How to Run Locally

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/your-username/iris-flower-intelligence.git
cd iris-flower-intelligence
```
### 2️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the Application
```bash
streamlit run app.py
```
### 📈 Model Performance
| Model | Accuracy |
|---------|---------|
| KNN | ~95% |
| Logistic Regression | ~96–97% |

---

### 🧪 Example Workflow
- **1.** Select a machine learning model
- **2.** Adjust flower measurements
- **3.** Click Analyze Flower
- **4.** View:
  - predicted species
  - confidence score
  - analytics & explainability
- **5.** Export results if needed

### 🙌 Acknowledgements
```
Scikit-learn Iris Dataset
Streamlit for rapid ML application development
```
## 🙌 Contribution
```
Feel free to fork this repo, suggest features, or raise issues.
```
## 📧 Contact

Made by Hitesh Parmar · Reach out on [LinkedIn](https://www.linkedin.com/in/hiteshparmar18/)
