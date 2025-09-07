# 🤖 Machine Learning Journey - 15 Days to ML Mastery

Welcome to my comprehensive Machine Learning learning repository! This repo documents my structured 15-day journey from ML fundamentals to building real-world predictive models.

## 🎯 Learning Objectives

By the end of this 15-day program, I will master:
- **Data Manipulation**: Pandas for data loading, cleaning, and exploration
- **Core ML Concepts**: Supervised learning, model training, and validation
- **Key Algorithms**: Decision Trees, Random Forests, Logistic Regression, KNN
- **Model Evaluation**: Train/test splits, cross-validation, performance metrics
- **Advanced Techniques**: Handling missing data, categorical encoding, pipelines
- **Real Projects**: End-to-end ML workflows on real datasets

## 📚 Repository Structure

```
ML-learning/
├── dataset/
│   └── train.csv           # Iowa Housing dataset
├── Model01.py              # First Decision Tree implementation
├── ModelValidation.py      # Model validation with train/test split
├── README.md              # This file
└── [Future files as I progress...]
```

## 🗓️ 15-Day Learning Plan

### **Week 1: Foundations & First Models**

| Day | Topic | Status | Key Learning |
|-----|-------|--------|--------------|
| **Day 1** | Pandas Fundamentals | ✅ | Data loading, `.head()`, `.describe()`, `.value_counts()` |
| **Day 2** | How Models Work | ✅ | Built first `DecisionTreeRegressor` |
| **Day 3** | Basic Data Exploration | 🔄 | Categorical vs numeric features |
| **Day 4** | First ML Model | ✅ | Trained Decision Tree on housing prices |
| **Day 5** | Model Validation | ✅ | Train/test split, MAE computation |
| **Day 6** | Underfitting & Overfitting | ✅ | Hyperparameter tuning (`max_depth`, `min_samples_leaf`) |
| **Day 7** | **Mini Project 1** | 📋 | Titanic survival prediction with Decision Tree |

### **Week 2: Better Models & Practice**

| Day | Topic | Status | Key Learning |
|-----|-------|--------|--------------|
| **Day 8** | Random Forests | 📋 | `RandomForestRegressor` vs Decision Tree |
| **Day 9** | Handling Missing Values | 📋 | `SimpleImputer`, `.fillna()` |
| **Day 10** | Categorical Data | 📋 | `pd.get_dummies()`, encoding techniques |
| **Day 11** | ML Pipelines | 📋 | Streamlined preprocessing + modeling |
| **Day 12** | Cross-Validation | 📋 | `cross_val_score` for robust evaluation |
| **Day 13** | Classification Algorithms | 📋 | KNN, Logistic Regression |
| **Day 14** | **Mini Project 2** | 📋 | End-to-end house price prediction |
| **Day 15** | **Final Project** | 📋 | Complete ML workflow on chosen dataset |

**Legend**: ✅ Completed | 🔄 In Progress | 📋 Planned

## 🚀 Current Progress

### ✅ Completed Work

#### 1. **First ML Model** (`Model01.py`)
- **Dataset**: Iowa Housing Prices
- **Algorithm**: Decision Tree Regressor
- **Features Used**: LotArea, YearBuilt, 1stFlrSF, 2ndFlrSF, FullBath, BedroomAbvGr, TotRmsAbvGrd
- **Key Achievement**: Successfully trained and made predictions on housing data

#### 2. **Model Validation** (`ModelValidation.py`)
- **Improvement**: Added proper train/test split
- **Evaluation Metric**: Mean Absolute Error (MAE)
- **Key Learning**: Importance of validating on unseen data

### 🔄 Currently Working On
- **Day 3**: Exploring categorical vs numeric features in Titanic dataset
- **Day 6**: Hyperparameter tuning for Decision Trees

## 📊 Datasets Used

| Dataset | Purpose | Source | Features |
|---------|---------|--------|----------|
| **Iowa Housing** | Regression (Price Prediction) | Kaggle | 79 features, 1460 samples |
| **Titanic** | Classification (Survival) | Kaggle | 12 features, 891 samples |

## 🛠️ Technologies & Libraries

```python
# Core Libraries
import pandas as pd                    # Data manipulation
import numpy as np                     # Numerical computing

# Scikit-learn Ecosystem
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
```

## 📈 Key Metrics & Results

### Model Performance Tracking
| Model | Dataset | Metric | Score | Notes |
|-------|---------|--------|-------|-------|
| Decision Tree | Iowa Housing | MAE | [To be updated] | Basic model |
| Decision Tree (Tuned) | Iowa Housing | MAE | [Pending] | With hyperparameter tuning |

## 🎯 Upcoming Milestones

### **Week 1 Goals**
- [ ] Complete Titanic dataset exploration
- [ ] Implement hyperparameter tuning
- [ ] Build first classification model

### **Week 2 Goals**
- [ ] Master Random Forests
- [ ] Learn data preprocessing pipelines
- [ ] Complete 3 end-to-end projects

## 🔧 How to Run

1. **Clone the repository**
   ```bash
   git clone [your-repo-url]
   cd ML-learning
   ```

2. **Install dependencies**
   ```bash
   pip install pandas scikit-learn numpy matplotlib seaborn
   ```

3. **Run individual scripts**
   ```bash
   python Model01.py           # First Decision Tree
   python ModelValidation.py   # Model with validation
   ```

## 📝 Learning Notes

### Key Concepts Mastered
- **Supervised Learning**: Learning from labeled data (features → target)
- **Regression vs Classification**: Predicting numbers vs categories
- **Overfitting**: Model memorizes training data, fails on new data
- **Validation**: Testing model performance on unseen data

### Best Practices Learned
- Always split data before training
- Use meaningful evaluation metrics (MAE for regression, accuracy for classification)
- Start simple, then add complexity
- Document and comment code thoroughly

## 🎉 Mini Projects

### Project 1: Titanic Survival Prediction (Day 7)
- **Goal**: Predict passenger survival using Decision Tree Classifier
- **Status**: Planned
- **Key Skills**: Classification, categorical data handling

### Project 2: House Price Prediction End-to-End (Day 14)
- **Goal**: Complete ML pipeline from data cleaning to model deployment
- **Status**: Planned
- **Key Skills**: Data preprocessing, feature engineering, model comparison

### Final Project: Custom Dataset Analysis (Day 15)
- **Goal**: Apply all learned skills on a chosen dataset
- **Options**: Iris, Stroke Prediction, Heart Disease
- **Status**: Planned
- **Deliverable**: Complete Kaggle notebook

## 🤝 Connect & Follow

This repository serves as both a learning log and a reference for anyone starting their ML journey. Feel free to:
- ⭐ Star this repo if you find it helpful
- 🍴 Fork it to start your own ML journey
- 📝 Open issues for questions or suggestions
- 📧 Reach out for collaboration

## 📚 Resources & References

- **Kaggle Learn**: [Machine Learning Course](https://www.kaggle.com/learn/machine-learning)
- **Scikit-learn Documentation**: [sklearn.org](https://scikit-learn.org/)
- **Pandas Documentation**: [pandas.pydata.org](https://pandas.pydata.org/)

---

**"The journey of a thousand models begins with a single decision tree."** 🌳

*Last Updated: Day 5 of 15 | Next Milestone: Hyperparameter Tuning*