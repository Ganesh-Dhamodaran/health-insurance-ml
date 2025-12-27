## 📚 Table of Contents

- [🏥 Project Overview](#-overview)
- [🎯 Problem Statement](#-problem-statement)
  - [✅ Fraud Detection](#-fraud-detection)
  - [✅ Customer Segmentation](#-customer-segmentation)
- [📁 Dataset Description](#-dataset-description)
- [🆚 Synthetic vs Real Insurance Dataset](#-synthetic-vs-real-insurance-dataset)
- [🧪 Exploratory Data Analysis (EDA)](#-exploratory-data-analysis-eda)
- [🧹 Data Cleaning & Feature Engineering](#-data-cleaning--feature-engineering)
- [🚨 Fraud Detection (Supervised ML)](#-fraud-detection-supervised-ml)
- [🧩 Customer Segmentation (Unsupervised ML)](#-customer-segmentation-unsupervised-ml)
- [🏗️ System Architecture](#️-system-architecture)
- [🚀 Deployment](#-deployment)
- [📂 Project Structure](#-project-structure)
- [🛠️ Installation & Setup](#️-installation--setup)
- [📊 Dashboard Preview](#-dashboard-preview)
- [📈 Business Impact](#-business-impact)
- [🔮 Future Enhancements](#-future-enhancements)
- [🙌 Author](#-author)

# Health Insurance Fraud Detection & Customer Segmentation  
## 🏥 Overview

This project demonstrates a complete, production‑style machine learning workflow using a **synthetic health insurance dataset**.  
It includes **fraud detection**, **customer segmentation**, **visual analytics**, and **deployment** using **Flask API** and a **Streamlit dashboard**.

---

## 🎯 Problem Statement

Insurance companies face significant financial losses due to fraudulent claims.  
This project aims to:

### ✅ Fraud Detection
Identify **suspicious claim activities** using:

- Demographic data  
- Policy and financial data  
- Risk indicators  
- Claim history  
- Behavioral and interaction data  

A synthetic fraud label (`Fraud_Flag`) is generated using realistic business rules to simulate real‑world fraud patterns.

### ✅ Customer Segmentation
Group customers into meaningful clusters based on:

- Demographics  
- Behavior  
- Policy details  
- Risk profile  

This helps insurers personalize offerings, optimize pricing, and identify high‑risk groups.

---

## 📁 Dataset Description

The dataset contains **customer‑centric insurance data** with the following fields:

### ✅ Demographics
- Age  
- Gender  
- Marital Status  
- Education Level  
- Occupation  
- Income Level  

### ✅ Geographic
- Geographic Information  
- Location  

### ✅ Behavioral
- Behavioral Data  
- Purchase History  
- Interactions with Customer Service  

### ✅ Policy & Financial
- Policy Start Date  
- Policy Renewal Date  
- Policy Type  
- Coverage Amount  
- Premium Amount  
- Deductible  

### ✅ Risk & Claims
- Risk Profile  
- Previous Claims History  
- Claim History  
- Driving Record  
- Credit Score  

### ✅ Preferences
- Customer Preferences  
- Preferred Communication Channel  
- Preferred Contact Time  
- Preferred Language  

### ✅ Segmentation
- Segmentation Group (used for clustering)

---

## 🆚 Synthetic vs Real Insurance Dataset

### ✅ Synthetic Dataset
- Artificially generated and privacy‑safe  
- Clean, consistent, and easier to model  
- Ideal for learning, experimentation, and demonstrating ML workflows  

### ✅ Real Insurance Dataset
- Collected from actual customer claims  
- Highly imbalanced, noisy, and complex  
- Requires heavy cleaning and domain expertise  

### ✅ In This Project
A **synthetic Kaggle dataset** is used to simulate real insurance behavior.  
It enables end‑to‑end fraud detection and segmentation without privacy concerns.

---

## 🧪 Exploratory Data Analysis (EDA)

The EDA notebook includes:

- Dataset overview  
- Missing value analysis  
- Numerical & categorical distributions  
- Correlation heatmap  
- Segmentation group analysis  
- Behavioral and risk insights  

📁 Visuals are stored in the `visuals/` folder.

---

## 🧹 Data Cleaning & Feature Engineering

Key steps:

- Handling missing values  
- Encoding categorical variables  
- Scaling numerical features  
- Date feature extraction  
- Outlier detection  
- Feature selection  
- Train‑test split  

---

## 🚨 Fraud Detection (Supervised ML)

A synthetic fraud label (`Fraud_Flag`) is generated using realistic rules based on:

- Low credit score  
- Poor driving record  
- High risk profile  
- Multiple previous claims  
- High coverage + low deductible  
- High premium relative to income  

### ✅ Models Implemented
- Logistic Regression  
- Random Forest  
- Support Vector Machine (SVM)  
- K‑Nearest Neighbors (KNN)  

### ✅ Evaluation Metrics
- Accuracy  
- Precision  
- Recall  
- F1‑score  
- ROC‑AUC  
- Confusion Matrix  

---

## 🧩 Customer Segmentation (Unsupervised ML)

Clustering algorithms used:

- K‑Means  
- DBSCAN  
- Hierarchical Clustering  

### ✅ Visualizations
- Elbow method  
- Silhouette score  
- PCA / t‑SNE  
- Dendrogram  

Cluster profiles are created to interpret customer groups.

---

## 🏗️ System Architecture  
## 🔄 End‑to‑End Project Flow

                Raw Data  
                   ↓  
                Preprocessing  
                   ↓  
                ==============================  
                 Fraud Detection Pipeline  
                ==============================  
                   ↓  
                Train Supervised Model  
                   ↓  
                Fraud Inference  
                   ↓  
                Dashboard / API Output  
                
                ==============================  
                 Customer Segmentation Pipeline  
                ==============================  
                   ↓  
                Train Clustering Model  
                   ↓  
                Segmentation Inference  
                   ↓  
                Dashboard Visualization  
                
                ==============================  
                 Final Outputs  
                ==============================  
                - Predictions  
                  - Probabilities  
                  - Clusters  
                  - PCA Plots  
                  - Downloadable Results   


---

## 🚀 Deployment

### ✅ Flask API (`/predict`)
- Accepts JSON input  
- Returns fraud prediction  
- Can be integrated with applications  

### ✅ Streamlit Dashboard
- Interactive fraud prediction  
- Customer segmentation visualization  
- Cluster insights  
- Model performance charts  

---

## 📂 Project Structure (Tree View)
            
            health-insurance-ml/
            ├── app/
            │   ├── app.py
            │   ├── flask_api.py
            │   ├── preprocess.py
            │   └── dashboard/
            │       ├── fraud_tab.py
            │       └── segmentation_tab.py
            │
            ├── fraud_detection/
            │   ├── inference.py
            │   ├── preprocessing.py
            │   └── model.py
            │
            ├── segmentation/
            │   ├── inference.py
            │   ├── preprocessing.py
            │   ├── pipeline.py
            │   └── profiling.py
            │
            ├── models/
            │   ├── fraud_detection_model.pkl
            │   ├── scaler.pkl
            │   ├── numerical_cols.pkl
            │   ├── categorical_cols.pkl
            │   ├── dummy_columns.pkl
            │   ├── segmentation_preprocessor.pkl
            │   ├── segmentation_kmeans.pkl
            │   ├── segmentation_pca.pkl
            │   └── segmentation_features.json
            │
            ├── notebooks/
            │   ├── EDA.ipynb
            │   ├── feature_engineering.ipynb
            │   └── model_experiments.ipynb
            │
            ├── visuals/
            │   ├── heatmap.png
            │   ├── pca_clusters.png
            │   └── fraud_probability.png
            │
            ├── train_segmentation.py
            ├── requirements.txt
            └── README.md

---

## 🛠️ Installation & Setup

### ✅ Install dependencies

pip install -r requirements.txt

### ✅ Train models

python train_segmentation.py

### ✅ Run Streamlit Dashboard

streamlit run app/app.py

### ✅ Run Flask API

python app.py

---

## 📊 Dashboard Preview

(Add screenshots here)

- Fraud Prediction Tab  
- Segmentation Tab  
- PCA Clusters  
- Fraud Probability Distribution  

---

## 📈 Business Impact

✅ Detect fraudulent claims early → reduce financial losses  
✅ Segment customers → personalized pricing & retention  
✅ Provide actionable insights to insurance analysts  
✅ Deployable as a real‑time decision support tool  

---

## 🔮 Future Enhancements

- SHAP explainability  
- AutoML for model selection  
- Real‑time API deployment (FastAPI)  
- Database integration  
- CI/CD with GitHub Actions  
- Docker containerization  

---

## 🙌 Author

**Ganesh**  
Senior Data Engineering & Data Science Specialist  
Chennai, India  