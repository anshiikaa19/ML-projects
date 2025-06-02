# ⚡ ML System for Real-Time Threat Detection

A cutting-edge machine learning initiative focused on **User and Entity Behavior Analytics (UEBA)**. This project identifies and flags anomalous activity in real time by learning from user behavior patterns—enhancing cybersecurity through actionable, data-driven insights.

---

## 🧠 Project Summary

**Goal**: To analyze and categorize user behavior logs as either *Normal* or *Suspicious*, based on system activities such as logins, file accesses, and configuration changes.

**Dataset**: A structured dataset containing labeled user actions, event timestamps, and system interaction logs for supervised learning and anomaly detection.

---

## 🔍 Core Features

### 1. 🚦 Intelligent Data Preprocessing
- Converted and standardized timestamp fields for uniform time-based modeling.
- Encoded categorical features using `LabelEncoder`.
- Applied `StandardScaler` for feature normalization and improved model convergence.

### 2. 🤖 Machine Learning Algorithms Implemented
- **Random Forest**
- **Decision Tree**
- **CatBoost**
- **K-Nearest Neighbors (KNN)**

Each model was fine-tuned to classify behavioral patterns efficiently and accurately.

### 3. 📏 Evaluation Metrics Used
- Classification: **Accuracy**, **Precision**, **Recall**, and **F1-Score**
- Regression (for reliability scoring): **Mean Squared Error (MSE)** and **R² Score**

---

## 🔄 Project Workflow

1. **Data Ingestion**: Parsed and imported raw logs from user interaction datasets.
2. **Preprocessing & Feature Engineering**: 
   - Time features converted into usable numeric formats.
   - Encoded labels and normalized features to prepare data for model training.
3. **Model Training**: Multiple ML algorithms were trained and compared.
4. **Model Assessment**: Evaluated performance using both visual and statistical metrics.
5. **Refinements**: Hyperparameter tuning and regularization strategies applied to avoid overfitting.

---

## 📈 Model Results

| Model               | Accuracy | Precision | Recall | R² (Train) | R² (Test) |
|---------------------|----------|-----------|--------|------------|-----------|
| Random Forest       | 0.80     | 0.80      | 0.80   | 1.00       | 0.12      |
| CatBoost            | 0.75     | 0.76      | 0.75   | 0.64       | -0.09     |
| Decision Tree       | 0.70     | 0.69      | 0.70   | -0.15      | -0.31     |
| K-Nearest Neighbors | 0.80     | 0.80      | 0.80   | -0.56      | 0.12      |

**Note**: While Random Forest and KNN achieved higher accuracy, R² values suggest potential overfitting in some models—indicating room for generalization improvement.

---

## 🌟 Potential Upgrades

- Implement ensemble learning techniques (e.g., Stacking, Voting Classifiers).
- Incorporate domain-specific features through advanced feature engineering.
- Integrate with a live stream or SIEM system for real-time anomaly detection and alerting.

---

