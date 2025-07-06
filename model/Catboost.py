# Data wrangling libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore") 

from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor

# Output settings
pd.set_option('display.max_columns', None) # display all columns
pd.set_option('display.expand_frame_repr', False) # print all columns and in the same line
pd.set_option('display.max_colwidth', None) # display the full content of each cell
pd.set_option('display.float_format', lambda x: '%.3f' %x) # floats to be displayed with 3 decimal places


def modelCAT(df, target):
    """
    
    """   
    # Split into train/test
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    # Initialize model
    model = CatBoostRegressor(verbose=0, random_state=123)

    # Fit on training set
    print(f"Training CatBoost model...")
    model.fit(X_train, y_train)
    print(f"Model training complete.")

    # Save the trained model
    model_path = 'model/robocop_model.cbm'
    try:
        model.save_model(model_path)
        print(f"Model successfully saved to '{model_path}'")
    except Exception as e:
        print(f"Error saving model: {e}")

    return model


if __name__ == "__main__":

    # Read the csv data
    df = pd.read_csv("model/data_cleaned.csv")

    # Set price as our target variable
    target = 'price'

    # Train and save the model
    trainedCat = modelCAT(df, target)

