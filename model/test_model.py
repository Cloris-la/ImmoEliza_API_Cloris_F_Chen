from catboost import CatBoostRegressor
import pandas as pd
import random

model = CatBoostRegressor()
model.load_model('model/robocop_model.cbm')

df = pd.read_csv("model/data_cleaned.csv")

indexes = random.sample(range(len(df)), 10)

for i in indexes:
    test = df.drop(columns='price').iloc[[i]]

    pred_price = model.predict(test)[0]
    actual_price = df['price'].iloc[i]
    error = (pred_price/actual_price - 1)  * 100

    print(f"Predicted price: {pred_price:.0f}; Actual price: {actual_price:.0f}; Error: {error:.1f}%")
