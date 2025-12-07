# 🏥 Health Insurance Fraud Detection & Customer Segmentation  
### End‑to‑End Machine Learning Project (EDA → Modeling → Clustering → Deployment)

This project demonstrates a complete, production‑style machine learning workflow using a **synthetic health insurance dataset**.  
It includes **fraud detection**, **customer segmentation**, **visual analytics**, and **deployment** using **Flask API** and **Streamlit dashboard**.

---

## 📌 Project Overview

Insurance fraud is a major challenge for insurers, leading to billions in losses every year.  
This project builds an end‑to‑end ML pipeline to:

✅ Detect fraudulent insurance claims (Supervised ML)  
✅ Segment customers into meaningful groups (Unsupervised ML)  
✅ Visualize insights for business decision‑making  
✅ Deploy models for real‑time predictions  

The goal is to showcase a **full-stack data science workflow** suitable for real-world applications.

---

## 📁 Dataset Description

The dataset used in this project is a **synthetic insurance dataset** from Kaggle.  
It simulates:

- Customer demographics  
- Policy details  
- Claim history  
- Fraud indicators  

Although synthetic, it closely resembles real insurance data and is ideal for demonstrating ML workflows.

---

## 🆚 Synthetic vs Real Insurance Dataset

### ✅ Synthetic Dataset
- Artificially generated  
- Cleaner and easier to model  
- Balanced fraud labels  
- No privacy issues  
- Great for learning and demos  

### ✅ Real Insurance Dataset
- Collected from actual claims  
- Highly imbalanced (fraud < 2%)  
- Messy, noisy, inconsistent  
- Requires domain knowledge  
- Harder but more realistic  

### ✅ In This Project
We use a **synthetic dataset**, making the project ideal for demonstrating:

- Fraud detection  
- Clustering  
- Visual analytics  
- Deployment  

---

## 🧪 Exploratory Data Analysis (EDA)

The EDA notebook includes:

- Dataset overview  
- Missing value analysis  
- Fraud distribution  
- Correlation heatmap  
- Numerical & categorical insights  
- Outlier detection  
- Feature relationships  

Visuals are stored in the `visuals/` folder.

---

## 🧹 Data Cleaning & Feature Engineering

Key steps:

- Handling missing values  
- Encoding categorical variables  
- Scaling numerical features  
- Feature selection  
- Outlier treatment  
- Train-test split  

---

## 🤖 Fraud Detection Models (Supervised ML)

The following models are implemented and compared:

- ✅ Random Forest  
- ✅ Support Vector Machine (SVM)  
- ✅ K-Nearest Neighbors (KNN)  
- ✅ Logistic Regression (baseline)  

### Evaluation Metrics:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC-AUC  
- Confusion Matrix  

Imbalanced learning techniques like **SMOTE** are applied when needed.

---

## 🧩 Customer Segmentation (Unsupervised ML)

Clustering algorithms used:

- ✅ K-Means  