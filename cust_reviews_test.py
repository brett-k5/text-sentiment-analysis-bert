import joblib
import numpy as np
from src.cust_reviews_pre_processing import custom_reviews

with np.load('embedded_custom_reviews.npz') as data:
    embedded_custom_reviews = data['cust_test']

model_log = joblib.load('models/model_log.pkl')


predictions = model_log.predict(embedded_custom_reviews)

for i in range(len(custom_reviews)):
    print(f"Review: {custom_reviews['review'].iloc[i]}, Prediction: {predictions[i]}\n")