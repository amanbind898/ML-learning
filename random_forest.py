import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt



print("Reading Data...")
data = pd.read_csv("././dataset/iowatrain.csv")
corr = data.corr(numeric_only=True)["SalePrice"].sort_values(ascending=False)
print(corr.head(15))   # top positive correlations
print(corr.tail(15))   # negative correlations
# Features that exist in Ames dataset
features = ['LotArea', 'GrLivArea', 'YearBuilt', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
X = data[features]
y = data.SalePrice  

rf_model = RandomForestRegressor(random_state=1)
rf_model.fit(X, y)

print("Model Trained.....")
