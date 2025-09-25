# 🤖 Machine Learning 

Welcome to my comprehensive Machine Learning learning repository! This repo documents ML fundamentals to building real-world predictive models.

## 🎯 Learning Objectives

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
│   ├── train.csv                    # Iowa Housing dataset
│   ├── test.csv                     # Test data
│   ├── gender_submission.csv        # Sample submission format
│   └── iowatrain.csv               # Additional training data
├── Intro to ML/                     # Learning materials and notes
├── Model01.py                       # First Decision Tree implementation
├── ModelValidation.py               # Model validation with train/test split
├── overfit_underfit.py              # Overfitting and underfitting examples
├── random_forest.py                 # Random Forest implementation
├── Titanic_survival_prediction_with_Decision_Tree.py  # Classification project
├── supervised.md                    # Theory about supervised learning
└── README.md                        # This file
```


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

#### 3. **Overfitting & Underfitting** (`overfit_underfit.py`)
- **Focus**: Understanding model complexity trade-offs
- **Techniques**: Hyperparameter tuning with `max_depth` and `min_samples_leaf`
- **Key Learning**: Finding the sweet spot between bias and variance

#### 4. **Random Forest Implementation** (`random_forest.py`)
- **Algorithm**: Random Forest Regressor
- **Improvement**: Better performance through ensemble methods
- **Key Learning**: How multiple trees can outperform single decision tree

#### 5. **Titanic Classification Project** (`Titanic_survival_prediction_with_Decision_Tree.py`)
- **Project Type**: Binary Classification
- **Algorithm**: Decision Tree Classifier
- **Dataset**: Titanic passenger data
- **Goal**: Predict survival based on passenger features

#### 6. **Supervised Learning Theory** (`supervised.md`)
- **Content**: Comprehensive theory documentation
- **Topics**: Supervised learning concepts, algorithms, and best practices
- **Purpose**: Theoretical foundation for practical implementations



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

