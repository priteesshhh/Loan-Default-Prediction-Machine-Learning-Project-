# 📊 Loan Default Prediction – Machine Learning Project
 

This project builds machine learning models to predict whether borrowers will become **90+ days delinquent** within two years using the *Give Me Some Credit* dataset. Accurate risk prediction helps financial institutions reduce losses while maintaining fairness and transparency.

---

# 📁 Dataset Summary
**Dataset:** Give Me Some Credit (Kaggle)  
**Target Variable:** `SeriousDlqin2yrs`  
**Training Samples:** 150,000  
**Features:** 10 numerical variables  

### **Dataset Challenges**
- Class imbalance (**6.7%** default rate)  
- Missing values (`MonthlyIncome` ~20%, `NumberOfDependents` ~3%)  
- Outliers in financial behavior features (Debt Ratio, Utilization)  
- No categorical encoding needed (all features numeric)

---

# 🧹 Data Preprocessing Pipeline

### ✔ Missing Value Handling
- Median imputation → `MonthlyIncome`  
- Mode imputation → `NumberOfDependents`

### ✔ Outlier Treatment
- Clipped at **99th percentile**  
- Enforced **minimum age = 21**

### ✔ Scaling
- `StandardScaler` applied to Logistic Regression only

### ✔ Train/Test Split
- **80/20 stratified split** to preserve imbalance ratio

### ✔ Class Imbalance Correction
- **SMOTE** oversampling applied to minority class during training

---

# 📊 Flowchart of the Pipeline
<img width="1591" height="104" alt="Screenshot 2025-12-03 at 11 19 13 AM" src="https://github.com/user-attachments/assets/72ad7f68-6454-463b-8786-b1bac5f1d6fa" />


---

# 🤖 Machine Learning Models

We implemented one baseline model and two advanced models:

### **Baseline Model**
- **Logistic Regression**  
  Simple, interpretable, strong calibration.

### **Advanced Models**
- **Random Forest** — robust, ensemble-based  
- **Gradient Boosting** — best for tabular data, strong recall  

---

# 📈 Model Performance (Validation Set)

## **Logistic Regression**
- **Accuracy:** 0.8034  
- **AUC:** 0.8602  
- **Confusion Matrix:**  
  - TN: 22601  
  - FP: 5394  
  - FN: 503  
  - TP: 1502  

📌 * Logistic Regression ROC Curve *  
 <img width="735" height="482" alt="Screenshot 2025-12-03 at 11 22 33 AM" src="https://github.com/user-attachments/assets/71b9d1ab-d432-40ef-92a4-9eb33ca12c2a" />


---

## **Random Forest**
- **Accuracy:** 0.8942  
- **AUC:** 0.8306  
- **Confusion Matrix:**  
  - TN: 25904  
  - FP: 2091  
  - FN: 1083  
  - TP: 922  

📌 * Random Forest ROC Curve *  
 <img width="724" height="482" alt="Screenshot 2025-12-03 at 11 23 49 AM" src="https://github.com/user-attachments/assets/f2f75562-a270-482c-b009-27487ba588a9" />


---

## **Gradient Boosting**
- **Accuracy:** 0.8716  
- **AUC:** 0.8337  
- **Confusion Matrix:**  
  - TN: 25055  
  - FP: 2940  
  - FN: 913  
  - TP: 1092  

📌 * Gradient Boosting + All Models ROC Curve *  
 <img width="750" height="483" alt="Screenshot 2025-12-03 at 11 24 45 AM" src="https://github.com/user-attachments/assets/675737ce-2ea2-4dec-9e86-414da1b7392b" />


---

# 🏆 Model Comparison Summary

| Model | Accuracy | AUC | Notes |
|-------|----------|------|------|
| Logistic Regression | 0.8034 | **0.8602** | Best AUC |
| Random Forest | **0.8942** | 0.8306 | Best accuracy |
| Gradient Boosting | 0.8716 | 0.8337 | Best recall |

### 🔍 Key Insights
- Logistic Regression ranks risk best (highest AUC).  
- Random Forest gives highest accuracy but weaker recall.  
- **Gradient Boosting achieves best recall for defaulters** — crucial for financial risk.  

---

# 🏅 Final Model Selection: Gradient Boosting

We selected **Gradient Boosting Classifier** as the final model because:

- It provides the **best recall** for minority default cases  
- Achieves strong accuracy and AUC  
- Minimizes **costly false negatives**  
- More stable and reliable for real-world risk prediction  


---

# 📂 Final Output: Predictions File

Generated predictions for the `cs-test.csv` dataset:

➡️ **`final_predictions.csv`**

Columns:
- `Id` — row index  
- `ProbabilityOfDefault` — predicted default risk  

---

# 🧠 Ethical Considerations

### ⚠ Risks
- **Proxy discrimination** via income, debt history, or other correlated features  
- **Unbalanced false positive/false negative rates**  
- Lack of transparency in automated credit scoring  

### ✔ Mitigation Strategies
- Fairness audits (compare FPR/FNR across groups)  
- Explainability tools (SHAP, LIME)  
- Model cards for documentation  
- Compliance with **Equal Credit Opportunity Act (ECOA)**  

---

# 🧪 Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Matplotlib / Seaborn  
- Imbalanced-learn (SMOTE)  
- Jupyter Notebook / Google Colab  

---

# 📘 How to Run the Notebook

Clone the repo:

```bash
git clone https://github.com/priteesshhh/Loan-Default-Prediction-Machine-Learning-Project-.git
cd loan-default-prediction

