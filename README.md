
# 🚀 Spaceship Titanic - ML Classification Project

This notebook is a solution for the **"Spaceship Titanic"** machine learning competition on [Kaggle](https://www.kaggle.com/competitions/spaceship-titanic). The task is to predict whether a passenger was **transported to an alternate dimension** based on various characteristics.

---

## 📁 Dataset Overview

The dataset is structured like a typical tabular classification problem:

### Main files:
- `train.csv`: Includes ~8,700 passengers with labeled target (`Transported`)
- `test.csv`: ~4,300 passengers without labels

### Key features:
- **Categorical:**  
  `HomePlanet`, `CryoSleep`, `Destination`, `VIP`, `Cabin`
- **Numerical:**  
  `Age`, `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck`
- **Target:**  
  `Transported` (Boolean: True/False)

---

## 🧠 Project Workflow

### 1. Exploratory Data Analysis (EDA)
- Checked **class balance**: the dataset is quite balanced between transported and not transported.
- Visualized **distribution of numerical features** like `Age` and `Spa` spending.
- Plotted **categorical features** using `countplot()` and `pie charts` to spot trends.
- Observed that features like `CryoSleep`, `Destination`, and `VIP` had visible correlations with the target.

### 2. Data Cleaning & Preprocessing
- **Dropped**:
  - `Name`: mostly irrelevant for prediction.
  - `Cabin`: many missing values, complex to parse initially.
- **Handled missing values**:
  - For categoricals like `HomePlanet`, filled with `"unknown"`.
  - For numerical spendings (`RoomService`, `FoodCourt`, etc.), filled with `0`.
  - For `Age`, used median imputation.
- **Converted categorical to numerical** using `LabelEncoder`.

### 3. Feature Engineering
- Simplified data types (e.g., from float64 to float32).
- Created no new features, kept baseline model clean and interpretable.

### 4. Model Training & Evaluation
- Trained and evaluated the following classifiers:
  - **Logistic Regression**
  - **Decision Tree**
  - **Random Forest**
  - **Extra Trees Classifier**
  - **LightGBM (LGBMClassifier)**
  - **XGBoost (XGBClassifier)**
- Used `train_test_split` for validation.
- Evaluated using:
  - `accuracy_score`
  - `classification_report`
  - `confusion_matrix`
- Compared each model’s performance. **Tree-based models (especially XGBoost and LGBM)** outperformed others in both accuracy and generalization.

### 5. Hyperparameter Tuning
- Applied `GridSearchCV` to:
  - **LGBMClassifier**
  - **RandomForestClassifier**
- Tuned parameters like:
  - `n_estimators`
  - `max_depth`
  - `learning_rate` (for LGBM)

---

## 🧪 Results

- **Best model**: LightGBM with tuned hyperparameters.
- Achieved **over 0.80 accuracy** on validation set.
- Saved predictions into CSV for Kaggle submission.

---

## 📦 Dependencies

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
lightgbm
```

---

## 🎯 Goal

To build a solid baseline ML pipeline and test the performance of various models on a real-world-style dataset. This project is useful for practicing:
- End-to-end ML workflows
- Data cleaning strategies
- Comparison of classic and modern classifiers
- Working with structured data from competitions

---
