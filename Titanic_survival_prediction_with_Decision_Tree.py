import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
path = './dataset/Titanic-Dataset.csv'
data = pd.read_csv(path)

# Select features
features = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare']
X = data[features].copy()
y = data['Survived']

# Handle categorical data
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# Handle missing values
X['Age'] = X['Age'].fillna(X['Age'].median())
X['Fare'] = X['Fare'].fillna(X['Fare'].median())

# Split dataset
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, random_state=1
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,   # number of trees
    max_depth=7,        # prevent overfitting
    random_state=1
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_valid)

# Accuracy
accuracy = accuracy_score(y_valid, y_pred)
print(f"Validation Accuracy: {accuracy:.4f}")

# Check feature importance
feature_importances = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nFeature Importances:")
print(feature_importances)


import numpy as np

# Example: Your details
# Change these values with your actual info
my_data = pd.DataFrame([{
    "Pclass": 1,    # passenger class (1 = 1st, 2 = 2nd, 3 = 3rd)
    "Age": 25,      # your age
    "Sex": 1,       # 0 = male, 1 = female
    "SibSp": 0,     # siblings/spouses aboard
    "Parch": 0,     # parents/children aboard
    "Fare": 7    # ticket fare
}])

# Predict survival
prediction = model.predict(my_data)[0]

if prediction == 1:
    print("✅ You would have SURVIVED on Titanic")
else:
    print("❌ You would NOT have survived on Titanic")


    # Define some sample passenger profiles
examples = pd.DataFrame([
    {"Pclass": 1, "Age": 25, "Sex": 1, "SibSp": 0, "Parch": 0, "Fare": 80},   # Young 1st class female
    {"Pclass": 3, "Age": 25, "Sex": 0, "SibSp": 0, "Parch": 0, "Fare": 7},    # Young 3rd class male
    {"Pclass": 2, "Age": 40, "Sex": 1, "SibSp": 1, "Parch": 1, "Fare": 30},   # Middle-aged 2nd class female with family
    {"Pclass": 1, "Age": 60, "Sex": 0, "SibSp": 0, "Parch": 0, "Fare": 100},  # Older 1st class male
    {"Pclass": 3, "Age": 8,  "Sex": 1, "SibSp": 2, "Parch": 1, "Fare": 15},   # Child female in 3rd class with family
])

# Predict probabilities of survival (not just 0/1)
probs = model.predict_proba(examples)[:, 1]

# Combine into a table
results = examples.copy()
results["Survival_Probability"] = probs

print(results.sort_values(by="Survival_Probability", ascending=False))
