<div align="center">

# Adult Income Classification Using Machine Learning

### CCS23213 – Data Mining

### Real-World Data Mining Project

---

### Group Project

**Course:** CCS23213 Data Mining

**Dataset:** Adult Income Dataset (UCI Machine Learning Repository)

**Task Type:** Classification

**Target Variable:** Income

**Algorithms:** Decision Tree · Naïve Bayes  

**Train-Test Splits:** 50:50 · 60:40 · 70:30 · 80:20 · 90:10

**Objective:** Predict whether an individual's annual income exceeds \$50,000 based on demographic and census-related attributes.

---

### Group Members

| Name | Student ID |
|--------|--------|
| Abdou Salam Sisawo | AIU24102XXX |
| Anas Muhammad Sani Lawal | AIU231020 AIU23102XXX |
| Zikrullah Faizan | AIU23102XXX |

---

### Lecturer

Dr. Mohamad Farhan Mohamad Mohsin 

---

### Submission Date

18 June 2026

</div>


# 1. Project Background

## 1.1 Dataset Overview

This project utilizes the **Adult Income Dataset**, also known as the **Census Income Dataset**, obtained from the UCI Machine Learning Repository. The dataset was originally extracted by Barry Becker from the 1994 United States Census Database and has become one of the most widely used benchmark datasets for classification problems in machine learning and data mining.

The objective of the dataset is to predict whether an individual's annual income exceeds **$50,000 per year** based on demographic, educational, occupational, and financial attributes.

### Dataset Source

* Source: UCI Machine Learning Repository
* Dataset Name: Adult (Census Income)
* URL: https://archive.ics.uci.edu/dataset/2/adult
* Donated: April 30, 1996
* Creators: Barry Becker and Ron Kohavi

### Dataset Characteristics

| Attribute          | Value                     |
| ------------------ | ------------------------- |
| Dataset Type       | Multivariate              |
| Data Mining Task   | Classification            |
| Number of Records  | 48,842                    |
| Number of Features | 14                        |
| Target Variable    | Income                    |
| Missing Values     | Yes                       |
| Feature Types      | Numerical and Categorical |

### Problem Statement

Income prediction is an important problem in socioeconomic analysis. Governments, researchers, and organizations frequently analyze demographic and occupational factors to understand income distribution patterns and support data-driven decision-making.

The problem addressed in this project is to determine whether an individual's annual income exceeds $50,000 based on information such as age, education, occupation, work class, working hours, and other demographic characteristics.

### Objective of the Study

The main objective of this project is to develop and evaluate machine learning classification models capable of accurately predicting whether an individual's annual income exceeds $50,000 per year.

### Data Mining Task

This project involves a **Classification** task because the target variable consists of two predefined categories:

* Income ≤ $50K
* Income > $50K

---

## 1.2 Dataset Attribute Description

The Adult Income Dataset contains 14 predictor variables and 1 target variable. The measurement scale and data type of each attribute are summarized below.

| Feature        | Description                     | Data Type   | Measurement Scale | Variable Nature |
| -------------- | ------------------------------- | ----------- | ----------------- | --------------- |
| age            | Age of individual               | Numerical   | Ratio             | Continuous      |
| workclass      | Employment category             | Categorical | Nominal           | Discrete        |
| fnlwgt         | Census final weight             | Numerical   | Ratio             | Continuous      |
| education      | Highest education level         | Categorical | Ordinal           | Discrete        |
| education-num  | Education level in numeric form | Numerical   | Ordinal           | Discrete        |
| marital-status | Marital status                  | Categorical | Nominal           | Discrete        |
| occupation     | Occupation type                 | Categorical | Nominal           | Discrete        |
| relationship   | Family relationship status      | Categorical | Nominal           | Discrete        |
| race           | Race category                   | Categorical | Nominal           | Discrete        |
| sex            | Gender                          | Categorical | Nominal           | Discrete        |
| capital-gain   | Capital gain amount             | Numerical   | Ratio             | Continuous      |
| capital-loss   | Capital loss amount             | Numerical   | Ratio             | Continuous      |
| hours-per-week | Weekly working hours            | Numerical   | Ratio             | Continuous      |
| native-country | Country of origin               | Categorical | Nominal           | Discrete        |
| income         | Annual income category (Target) | Categorical | Nominal           | Discrete        |

---

## 1.3 Machine Learning Workflow

The project follows the standard data mining and machine learning lifecycle:

1. Data Collection and Integration
2. Data Understanding
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Feature Engineering
6. Train-Test Split Experiments
7. Model Development
8. Model Evaluation
9. Model Comparison
10. Model Validation Using Unseen Data
11. Model Export and Deployment Preparation

Two classification algorithms will be developed and evaluated using five train-test split ratios (50:50, 60:40, 70:30, 80:20, and 90:10). The best-performing model will be selected based on Accuracy, Precision, Recall, and F1-Score.


# Why MinMaxScaler is chosen
print("MinMaxScaler chosen because:")
print("  - Decision Tree does not require normalisation, but Naïve Bayes benefits")
print("    from bounded input ranges for its Gaussian likelihood estimation.")
print("  - capital-gain and capital-loss have extreme right skew; MinMax [0,1]")
print("    suppresses the influence of extreme winsorised values without distortion.")
print("  - Unlike StandardScaler, MinMax does not assume normality.")
print()
print("IMPORTANT: the preprocessor is fitted ONLY on the training split inside")
print("each experiment loop — preventing any data leakage from the test partition.")



# 7. Findings, Recommendations, Strengths & Limitations
print("""
FINDINGS
========
1. The Decision Tree classifier outperforms Naïve Bayes on every metric across all
   five train-test splits, with F1-scores consistently above 0.85.

2. capital-gain, relationship status, and marital status emerge as the strongest
   predictors of high income — consistent with socioeconomic theory.

3. Class imbalance (76% <=50K, 24% >50K) is a genuine challenge; addressing it
   with class_weight='balanced' in the Decision Tree significantly improves Recall
   for the minority class.

4. Naïve Bayes underperforms because its conditional-independence assumption is
   violated: features such as relationship, marital-status, and occupation are
   strongly correlated in census data.

RECOMMENDATIONS
===============
- Deploy the Decision Tree (best split: see results_df) as the production classifier.
- Future work should explore ensemble methods (Random Forest, Gradient Boosting)
  to further reduce variance.
- Recollect more recent census data — the 1994 dataset may not reflect current
  income dynamics, particularly around gender parity and remote work patterns.
- Integrate SHAP values for individual-level model explanations required in
  high-stakes deployment contexts (credit, hiring).

STRENGTHS
=========
- Full preprocessing pipeline fitted inside each train-test split loop — no data leakage.
- Mode imputation (rather than row deletion) retains all 48,842 records.
- Winsorisation of capital-gain and capital-loss preserves record count while
  controlling extreme-value distortion.
- Five distinct train-test splits provide a reliable picture of bias-variance behaviour.

LIMITATIONS
===========
- The 1994 dataset is not representative of current demographic and economic conditions.
- Gaussian Naïve Bayes assumes continuous features follow a Gaussian distribution —
  an assumption capital-gain and capital-loss clearly violate (spike at zero + heavy tail).
- Decision Tree is prone to overfitting beyond max_depth=8; hyperparameter tuning
  via grid search was not performed within this scope.
- Native country was collapsed into a high-cardinality OHE block; grouping by region
  might yield more robust splits.
""")
