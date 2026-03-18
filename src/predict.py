import pandas as pd
import pickle

# Load model
model, sc = pickle.load(open('models/logistic_model.pkl', 'rb'))

# Load new dataset
dataset1 = pd.read_csv('data/final1.csv')
d2 = dataset1.copy()

X_new = dataset1.iloc[:, [3, 4]].values
X_new = sc.transform(X_new)

# Predict
predictions = model.predict(X_new)
d2['y_pred1'] = predictions

# Save output
d2.to_csv('output/final2.csv', index=False)
