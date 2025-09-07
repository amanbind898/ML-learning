# ==========================
# Step 1: Import Libraries
# ==========================
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# ==========================
# Step 2: Load Data
# ==========================
path = './dataset/iowatrain.csv'   # Iowa Housing Prices dataset
home_data = pd.read_csv(path)

y = home_data.SalePrice
# Create X
features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = home_data[features]

# Split into training and validation sets
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, random_state=1
)

# ==========================
# Step 3: Helper function
# ==========================
def get_mae(max_leaf_nodes, X_train, X_valid, y_train, y_valid):
    """Train a DecisionTreeRegressor with given max_leaf_nodes and return MAE"""
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=1)
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)
    return mae

# ==========================
# Step 4: Try different complexities
# ==========================
candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
for leaf_size in candidate_max_leaf_nodes:
    mae = get_mae(leaf_size, X_train, X_valid, y_train, y_valid)
    print(f"Max leaf nodes: {leaf_size} \t Mean Absolute Error: {mae}")

# ==========================
# Step 5: Choose Best Model
# ==========================
best_tree_size = min(candidate_max_leaf_nodes,
                     key=lambda leaf_size: get_mae(leaf_size, X_train, X_valid, y_train, y_valid))

print(f"Best tree size: {best_tree_size}")

final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=1)
final_model.fit(X, y)  # Train on all data

# ==========================
# Explanation of Overfitting / Underfitting
# ==========================

"""
Underfitting (High Bias):
-------------------------
- Happens when the model is too simple (e.g., very few leaf nodes like 5).
- The tree cannot capture patterns in the data.
- Both training and validation error are high.

Overfitting (High Variance):
----------------------------
- Happens when the model is too complex (e.g., 500+ leaf nodes).
- The tree memorizes training data and fits noise.
- Training error is very low, but validation error is high.

Good Fit (Bias-Variance Tradeoff):
----------------------------------
- Found by choosing an appropriate `max_leaf_nodes`.
- Validation error is minimized.
- The model generalizes well to unseen data.
"""
