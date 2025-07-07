import numpy as np
import pandas as pd
import os
import warnings
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib


class DataValidator:
    '''
    Just validate. No transform
    '''

    @staticmethod
    def validation_input(data_dict) :
        '''
        validate mandatory fields and types
        '''

        required_fields = ["area", "property-type", "rooms-number", 'zip-code']
        
        for field in required_fields:
            if field not in data_dict:
                return False, f"Missing required field: {field}"
            if data_dict[field] is None:
                return False, f"Required field {field} cannot be None."
        
        # check value of required fields
        if data_dict["area"] <= 0:
            return False, "Area must be greater than 0."
        
        valid_types = ["APARTMENT", "HOUSE", "OTHERS"]
        if data_dict["property-type"] not in valid_types:
            return False, f"Invalid property type. Must be one of {valid_types}."
        
        if data_dict["rooms-number"] < 0:  
            return False, "Number of rooms cannot be negative."
        
        zip_code = data_dict["zip-code"]
        if zip_code < 1000 or zip_code > 9999:
            return False, "Invalid Belgian postal code (must be 4 digits between 1000 to 9999)."
        
        return True, None


class InputCleaner:
    '''
    Transform API input to DataFrame format
    '''
    
    @staticmethod
    def json_to_dataframe(data_dict):
        '''
        Transform API input(json) to DataFrame format for model training
        '''
        # field mapping: API format -> Training format
        field_mapping = {
            "area": "habitablesurface",
            "rooms-number": "bedroomcount", 
            "lift":"lift",
            "garden": "garden",
            "swimming-pool": "swimmingpool",
            "terrace": "terrace",
            "parking":"parking",
            "epc-score":"epcscore",  
            "building-state": "building_state",
            "property-type": "property_type",
            "zip-code": "zip_code",
        }

        # create data of dataframe format
        df_data = {}
        for api_field, df_field in field_mapping.items() :
            if api_field in data_dict :
                df_data[df_field] = data_dict[api_field]
            else:
                df_data[df_field] = None
        return pd.DataFrame([df_data])
    

class PropertyTypeEncoder(BaseEstimator,TransformerMixin) :
    '''
    Define a sklearn transformer for property type
    '''
    def __init__(self):
        self.property_type_map = {
            "APARTMENT": 0,
            "HOUSE": 1,
            "OTHERS": 0
        }

    def fit(self, X,y = None) :
        return self
    
    def transform(self, X) :
        X_copy = X.copy()
        X_copy['type_encoded'] = X_copy["property_type"].map(self.property_type_map).fillna(0)
        return X_copy
    
    
class BuildingStateEncoder(BaseEstimator,TransformerMixin) :
    '''
    Building state transformer
    '''
    def __init__(self):
        self.building_state_map = {
            "NEW": 0,
            "TO RENOVATE": 1,
            "GOOD": 2,
            "TO BE DONE UP": 3,
            "JUST RENOVATED": 4,
            "TO REBUILD": 5
        }

    def fit(self,X,y = None) :
        return self
        
    def transform(self,X) :
        X_copy = X.copy()
        X_copy['building_state'] = X_copy["building_state"].fillna('GOOD')
        X_copy["buildingcondition_encoded"] = X_copy['building_state'].map(self.building_state_map).fillna(2)
        return X_copy
        

class EPCScoreEncoder(BaseEstimator,TransformerMixin) :
    '''
    EPC Score transformer
    '''
    def __init__(self):
        self.epc_score_map = {
            "A++": 0, "A+": 1, "A": 2, "B": 3, "C": 4,
            "D": 5, "E": 6, "F": 7, "G": 8
        }

    def fit(self, X, y=None) :
        return self
    
    def transform(self,X) :
        X_copy = X.copy()
        # fill default value
        if 'epcscore' not in X_copy.columns:
            X_copy['epcscore'] = 'C'
        X_copy["epcscore"] = X_copy["epcscore"].fillna('C')
        X_copy['epcscore_encoded'] = X_copy['epcscore'].map(self.epc_score_map).fillna(4)
        return X_copy


class BooleanFeatureEncoder(BaseEstimator,TransformerMixin):
    '''
    Boolean Feature transform
    '''
    def __init__(self):
        self.boolean_features = {
           'lift', 'garden', 'swimmingpool', 'terrace', 'parking'
        }
        self.feature_mapping = {
            'lift' : 'haslift',
            'garden' :'hasgarden',
            'swimmingpool' : 'hasswimmingpool',
            'terrace' : 'hasterrace',
            'parking' : 'hasparking'
        }

    def fit(self,X,Y = None):
        return self

    def transform(self,X) :
        X_copy = X.copy()
        for feature in self.boolean_features :
            if feature in X_copy.columns:
                encoded_name = self.feature_mapping.get(feature, f'has{feature}')
                X_copy[encoded_name] = X_copy[feature].fillna(False).astype(bool).astype(int)
            else:
                encoded_name = self.feature_mapping.get(feature,f"has{feature}")
                X_copy[encoded_name] = 0
        return X_copy
    

class GeographicEncoder(BaseEstimator,TransformerMixin):
    '''
    Geographic Encoder
    '''
    def __init__(self, geo_file_path="Aperol project/data/georef-belgium-postal-codes@public.csv"):
        self.geo_file_path = geo_file_path
        self.region_mapping = {
            'Région de Bruxelles-Capitale': 'Brussels',
            'Région flamande': 'Flanders',
            'Région wallonne': 'Wallonia'
        }
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        for idx, row in X_copy.iterrows():
            postcode = row['zip_code']
            latitude, longitude, region = self._get_location_info(postcode)
            
            X_copy.loc[idx, 'latitude'] = latitude
            X_copy.loc[idx, 'longitude'] = longitude
            X_copy.loc[idx, 'region_Brussels'] = 1 if region == 'Brussels' else 0
            X_copy.loc[idx, 'region_Flanders'] = 1 if region == 'Flanders' else 0
            X_copy.loc[idx, 'region_Wallonia'] = 1 if region == 'Wallonia' else 0
        
        return X_copy
    
    def _get_location_info(self, postcode):
        """
        Get geo information
        """
        try:
            if os.path.exists(self.geo_file_path):
                df_geo = pd.read_csv(self.geo_file_path, sep=";", encoding="utf-8")
                geo_row = df_geo[df_geo["Post code"] == postcode]
                
                if not geo_row.empty:
                    geo_point = geo_row['Geo Point'].iloc[0]
                    latitude, longitude = map(float, geo_point.split(','))
                    region_french = geo_row['Région name (French)'].iloc[0]
                    region = self.region_mapping.get(region_french, 'Unknown')
                    return latitude, longitude, region
        except Exception as e:
            pass
        
        # plan B: Using approximate location
        if 1000 <= postcode <= 1299:
            return 50.8503, 4.3517, "Brussels"
        elif ((1300 <= postcode <= 1499) or (2000 <= postcode <= 2999) or 
              (3000 <= postcode <= 3999) or (8000 <= postcode <= 8999) or 
              (9000 <= postcode <= 9999)):
            return 51.0, 4.5, "Flanders"
        else:
            return 50.5, 5.0, "Wallonia"
        
              
class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    choosing features needed for final model
    """
    
    def __init__(self):
        self.required_features = [
            'bedroomcount', 'habitablesurface', 'haslift', 'hasgarden', 
            'hasswimmingpool', 'hasterrace', 'hasparking', 'epcscore_encoded',
            'buildingcondition_encoded', 'region_Brussels', 'region_Flanders',
            'region_Wallonia', 'type_encoded', 'latitude', 'longitude'
        ]
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X_copy = X.copy()
        
        # make sure all needed features exist
        for feature in self.required_features:
            if feature not in X_copy.columns:
                X_copy[feature] = 0
        
        return X_copy[self.required_features]

def create_preprocessing_pipeline():
    """
    Create a complete proprecessing pipeline
    """
    pipeline = Pipeline([
        ('property_encoder', PropertyTypeEncoder()),
        ('building_encoder', BuildingStateEncoder()),
        ('epc_encoder', EPCScoreEncoder()),
        ('boolean_encoder', BooleanFeatureEncoder()),
        ('geo_encoder', GeographicEncoder()),
        ('feature_selector', FeatureSelector())
    ])
    
    return pipeline

def preprocess(data_dict):
    """
    Main preprocessing function of API
    
    Args:
        data_dict: JSON data from API
    
    Returns:
        (preprocessed_data, None) If successful
        (None, error_message) If fail
    """
    try:
        # 1. validate input
        is_valid, error_msg = DataValidator.validation_input(data_dict)
        if not is_valid:
            return None, error_msg
        
        # 2. transform to DataFrame
        df = InputCleaner.json_to_dataframe(data_dict)
        
        # 3. Using preprocessing pipeline
        pipeline = load_preprocessing_pipeline()
        if pipeline is None:
            # Create a new pipeline if no saved pipeline
            pipeline = create_preprocessing_pipeline()
        
        processed_data = pipeline.transform(df)
        
        return processed_data, None
        
    except Exception as e:
        return None, f'Error during preprocessing: {str(e)}'

def save_preprocessing_pipeline(pipeline, filepath='preprocessing/preprocessing_pipeline.pkl'):
    """
    save preprocessing pipeline
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)

def load_preprocessing_pipeline(filepath='preprocessing/preprocessing_pipeline.pkl'):
    """
    load preprocessing pipeline
    """
    try:
        return joblib.load(filepath)
    except:
        return None

# function for training
def fit_and_save_pipeline(training_data):
    """
    fitting and saving pipeline when training
    """
    pipeline = create_preprocessing_pipeline()
    pipeline.fit(training_data)
    save_preprocessing_pipeline(pipeline)
    print("✅ Pipeline saved successfully!")
    return pipeline