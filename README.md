# medical-cost-prediction
# 🧮 Medical Insurance Cost Prediction

This project uses linear regression in R to predict individual medical insurance costs based on personal and lifestyle features such as age, BMI, smoking status, and region.

## 📊 Project Overview

- **Goal**: Predict `charges` (medical insurance cost) using customer attributes
- **Dataset**: [Insurance dataset from Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Model**: Linear regression using the `caret` package in R
- **Language**: R

## 📁 Dataset Features

| Column     | Description                             |
| ---------- | --------------------------------------- |
| `age`      | Age of primary beneficiary              |
| `sex`      | Gender                                  |
| `bmi`      | Body Mass Index                         |
| `children` | Number of children covered by insurance |
| `smoker`   | Smoking status                          |
| `region`   | Residential region in the US            |
| `charges`  | Medical insurance cost (target)         |

## 🧠 Methodology

- Split data into training (80%) and testing (20%) sets
- Preprocessed numeric features (scaled + centered)
- Encoded categorical variables as factors
- Trained linear regression model using `caret::train`
- Evaluated model using RMSE and R-squared
- Visualized predicted vs. actual values

## 📈 Results

### 🔵 Predicted vs. Actual (Training Set)

![Training Plot](images/Rplot.png)

### 🔵 Predicted vs. Actual (Test Set)

![Testing Plot](images/Rplot01.png)

### 📌 Model Performance

| Metric    | Training Set   | Testing Set    |
| --------- | -------------- | -------------- |
| RMSE      | *6126.3533715* | *5712.9682177* |
| R-squared | *0.7430295*    | *0.7827933*    |

## 🔍 Feature Importance

![Feature Importance](images/Rplot02.png)

---

## ✅ Key Takeaways

- Smoking status and age were the most influential predictors of insurance cost
- A simple, real-world use case for business/healthcare analytics
