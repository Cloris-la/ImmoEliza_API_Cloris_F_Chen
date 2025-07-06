
from catboost import CatBoostRegressor
import pandas as pd


def predict(dataInput):
    
    model = CatBoostRegressor()
    model.load_model('model/robocop_model.cbm')
    dataOutput = model.predict(dataInput)
    return float(dataOutput[0])



#for testing purposes
df = pd.read_csv("model/data_cleaned.csv")
#print(*df.columns, sep="\n")
print(df.iloc[2].value_counts())

import random
indexes = random.sample(range(len(df)), 10)
for i in indexes:
    dataInput = df.drop(columns='price').iloc[[i]]

    dataOutput = predict(dataInput)
    print(f"Predicted price: {dataOutput:.0f}")

    # actual_price = df['price'].iloc[i]
    # error = (dataOutput/actual_price - 1)  * 100
    # print(f"Predicted price: {dataOutput:.0f}; Actual price: {actual_price:.0f}; Error: {error:.1f}%")

    