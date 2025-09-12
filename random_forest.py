import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

path= '/dataset/iowatrain.csv'

data=pd.read_csv(path)
features=['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 
                        'YearBuilt', 'Lattitude', 'Longtitude']
X=data[features]
y=data.price


rf_model= RandomForestRegressor(random_state=1)
rf_model.fit()