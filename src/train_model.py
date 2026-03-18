import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
import pickle

# Ensure models folder exists
os.makedirs('models', exist_ok=True)

# Load dataset
dataset = pd.read_csv('data/logit_classification.csv')

X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, -1].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

# Scaling
sc = Normalizer()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

print("Model trained successfully ✅")

# Save model properly
with open('models/logistic_model.pkl', 'wb') as f:
    pickle.dump((model, sc), f)

print("Model saved successfully ✅")
