# ================================
# 📌 House Price Prediction with Decision Tree
# Using Scikit-learn (Supervised Learning Example)
# ================================

# 1. Import Required Libraries
import pandas as pd                                 # For handling datasets (loading, cleaning, selecting columns)
from sklearn.tree import DecisionTreeRegressor     # ML model: Decision Tree Regressor (used for predicting numbers)
from sklearn.model_selection import train_test_split  # For splitting data into train & test sets
from sklearn.metrics import mean_absolute_error    # For evaluating model accuracy

# 2. Load the Dataset
path = './dataset/iowatrain.csv'   # Path to the dataset (Iowa Housing Prices dataset)
home_data = pd.read_csv(path)  # Read the CSV file into a pandas DataFrame

# 3. Define Target Variable (y)
# 👉 The target is the value we want the model to predict.
# Here, it's the "SalePrice" of the house.
y = home_data.SalePrice

# 4. Define Features (X)
# 👉 Features are the input variables that help predict the target.
# We choose a few relevant columns from the dataset.
feature_columns = [
    'LotArea',        # Size of the lot in square feet
    'YearBuilt',      # Year the house was built
    '1stFlrSF',       # First floor square feet
    '2ndFlrSF',       # Second floor square feet
    'FullBath',       # Number of full bathrooms
    'BedroomAbvGr',   # Number of bedrooms above ground
    'TotRmsAbvGrd'    # Total rooms above ground
]

# Extract these columns as our feature matrix (X)
X = home_data[feature_columns]

# 5. Define the Model
# 👉 DecisionTreeRegressor is used for regression problems
# (predicting continuous values like house prices).
iowa_model = DecisionTreeRegressor(random_state=42)  # random_state ensures reproducibility

# 6. Fit the Model (Training Step)
# 👉 "fit" means the model learns patterns from the data.
# The model studies the relationship between features (X) and target (y).
iowa_model.fit(X, y)

# 7. Make Predictions
# 👉 After fitting, the model can predict house prices for new data.
# Here we test it on the first 5 rows of our dataset.
predicted_prices = iowa_model.predict(X.head())

# 8. Compare Predictions with Actual Values
print("📌 First in-sample predictions:", predicted_prices)
print("📌 Actual target values for those homes:", y.head().tolist())
