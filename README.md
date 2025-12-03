# **Income Prediction Using the UCI Adult Dataset**

This project focuses on **predicting whether an individual's annual income exceeds $50K** using the UCI Adult dataset. The repository provides an end-to-end machine learning workflow, starting with **data exploration, preprocessing, and feature engineering**, and continuing through **model training, evaluation, and interpretability**.

### **Dataset Overview**

* **Source:** UCI Adult Dataset (via `ucimlrepo.fetch_ucirepo id=2`)
* **Rows:** 48,842
* **Features:** 14 columns including age, workclass, education, occupation, capital gain/loss, hours-per-week, race, sex, native-country
* **Target:** `income` (>50K / <=50K)

### **Key Steps and Pipeline**

1. **Data Cleaning & Preprocessing**

   * Handle missing values (`workclass`, `occupation`, `native-country`)
   * Encode categorical variables (One-Hot, Ordinal, Target encoding)
   * Scale numerical features for distance-based models

2. **Exploratory Data Analysis (EDA)**

   * Examine distributions, correlations, and imbalances
   * Identify feature importance and redundancy

3. **Modeling**

   * Baseline models: Logistic Regression, Decision Trees
   * Advanced models: Random Forest, Gradient Boosting (XGBoost, LightGBM), Neural Networks
   * Cross-validation and hyperparameter tuning

4. **Evaluation & Fairness**

   * Metrics: Accuracy, Precision, Recall, F1, ROC-AUC
   * Address class imbalance via `class_weight` or resampling (SMOTE)
   * Assess fairness across sensitive attributes (race, sex)

5. **Interpretability**

   * Feature importance, SHAP values, and partial dependence
   * Save models and preprocessing pipelines for reproducibility
   * Optional deployment: API or batch prediction scripts
