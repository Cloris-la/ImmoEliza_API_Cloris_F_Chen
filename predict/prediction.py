import pandas as pd
import numpy as np
import os
from catboost import CatBoostRegressor

class HousePricePredictor:
    '''
    House price predictor. Using Catboost directly
    '''
    def __init__(self, model_path='model/robocop_model.cbm'):  
        '''
        Initialize predictor.

        Arg:
            model path
        '''
        self.model_path = model_path
        self.model = None
        self._load_model()

    def _load_model(self):
        '''
        Load model file with catboost model 
        '''
        try:
            if os.path.exists(self.model_path):
                self.model = CatBoostRegressor()
                self.model.load_model(self.model_path)  
                print(f'✅ Load Catboost model successfully: {self.model_path}')
            else:
                print(f'❌ Model is not exist: {self.model_path}')
                print('Make sure path of .cbm file is right.')
                self.model = None
        except Exception as e:
            print(f'❌ Failed on loading Catboost model: {e}')
            self.model = None

    def predict(self, preprocessed_data):
        '''
        Using preprocessed data to do prediction

        Arg:
            preprocessed data(DataFrame, shape(1,15))

        Return:
            float: Predicted price
            None: If prediction fail
        '''
        if self.model is None:
            print('❌ No model loaded. Cannot predict price')
            return None
        
        try:
            # make sure input's format                
            if not isinstance(preprocessed_data, pd.DataFrame):
                print('❌ Input must be DataFrame.')
                return None
 
            # make sure one property's API input each time
            if preprocessed_data.shape[0] != 1:
                print(f'❌ Row of input data is wrong. Expect 1 row only. Input {preprocessed_data.shape[0]} rows.')
                return None
                
            # make sure 15 features are exist for model prediction
            if preprocessed_data.shape[1] != 15:
                print(f'❌ Column counts is wrong. Expect 15 columns. Only {preprocessed_data.shape[1]} columns.')
                return None
                
            # Catboost prediction
            prediction = self.model.predict(preprocessed_data)

            # Extract predicted price
            if isinstance(prediction, np.ndarray):
                predicted_price = float(prediction[0])
            else:
                predicted_price = float(prediction)

            # make sure prediction are greater than 0
            if predicted_price < 0:  
                print(f'⚠️ Prediction is negative number: {predicted_price}. Return absolute number')
                predicted_price = abs(predicted_price)
            
            print(f'✅ CatBoost predict successfully: €{predicted_price:,.2f}')
            return predicted_price
            
        except Exception as e:
            print(f'❌ Mistake of prediction: {e}')
            return None
        
def predict(preprocessed_data, model_path='model/robocop_model.cbm'):  
    '''
    Main predict function for API

    Args:
        preprocessed_data(DataFrame)
        model_path: CatBoost model file

    Return:
        float: Predicted price
        None: If prediction failed
    '''
    # Create a predictor instance
    predictor = HousePricePredictor(model_path)

    # predict
    return predictor.predict(preprocessed_data)
    
def predict_with_error_handling(preprocessed_data, model_path='model/robocop_model.cbm'):
    '''
    Function return error handling for API 

    Args:
        preprocessed_data(DataFrame)
        model_path: CatBoost model file

    Return:
        tuple:(prediction, error_message)
        prediction: float or None
        error_message: str or None
    '''
    try:
        prediction = predict(preprocessed_data, model_path)
        if prediction is not None:
            return prediction, None
        else:
            return None, "Prediction failed - CatBoost model error"
    except Exception as e:
        return None, f'Prediction error: {str(e)}'