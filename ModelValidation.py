# ================================
# 🏡 Iowa House Price Prediction
# Using Decision Tree Regressor
# ================================

# 1. Import Required Libraries
import pandas as pd
from sklearn.tree import DecisionTreeRegressor       # ML model
from sklearn.model_selection import train_test_split # Split data into train/test
from sklearn.metrics import mean_absolute_error      # Evaluation metric

# 2. Load the Dataset
path = './dataset/train.csv'    # Path to Iowa Housing dataset (Ames, Iowa)
home_data = pd.read_csv(path)   # Load CSV into a pandas DataFrame

# 3. Define Target (y)
# 👉 This is the column we want to predict: the house sale price.
y = home_data.SalePrice

# 4. Define Features (X)
# 👉 Features are the input variables that help the model predict the target.
feature_columns = [
    'LotArea',        # Lot size (in square feet)
    'YearBuilt',      # Year the house was built
    '1stFlrSF',       # First floor square footage
    '2ndFlrSF',       # Second floor square footage
    'FullBath',       # Number of full bathrooms
    'BedroomAbvGr',   # Bedrooms above ground
    'TotRmsAbvGrd'    # Total rooms above ground
]

X = home_data[feature_columns]   # Select only the chosen features

# 5. Split Data into Training and Validation Sets
# 👉 We train the model on training data, and test it on validation data.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

# 6. Define the Model
# 👉 DecisionTreeRegressor is a regression model that splits data into branches
# based on feature values to predict continuous outputs (like prices).
iowa_model = DecisionTreeRegressor(random_state=1)

# 7. Fit the Model (Training Step)
# 👉 "fit" means the model learns the relationship between X (features) and y (target).
iowa_model.fit(train_X, train_y)

# 8. Make Predictions on Validation Data
val_predictions = iowa_model.predict(val_X)

# Show a few predictions vs actual values
print("🔮 First validation predictions:", iowa_model.predict(val_X.head()))
print("✅ Actual target values for those homes:", val_y.head().tolist())

# 9. Evaluate the Model
# 👉 Mean Absolute Error (MAE) = average of |actual - predicted|
# Lower MAE = better model
mae = mean_absolute_error(val_y, val_predictions)
print("📉 Mean Absolute Error (MAE):", mae)
