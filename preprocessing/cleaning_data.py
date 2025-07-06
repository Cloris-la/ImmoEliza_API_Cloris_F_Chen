import numpy as np
import pandas as pd
import os

from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import Pipeline
import warnings


def preprocess(data_dict):
    '''
    Preprocessing data from API gets

    Args:
        data_dict: data from API,for example:
                {"area": 100, "property-type": "HOUSE", ...}

    Returns:
        (preprocessed_data,None) If successful
        (preprocessed_data, warning_message) If successful, with warnings
        (None, error_message) If fail
    '''
    try:
        # 1. check required fields
        required_fields = ["area","property_type","rooms_number",'zip_code']

        for field in required_fields:
            #first check if all fields exist
            if field not in data_dict:
                return None,f"Missing required field:{field}"
            
            # check if no value of field
            if data_dict[field] is None:
                return None,f"Required field {field} cannot be None."
            
        print("✅ All fields and values exist")

        # 2. check values of fields
        # area must greater than 0
        area = data_dict["area"]
        if area <= 0 :
            return None,"Area must be greater than 0."
        
        # property type must be house, apartment,others
        valid_types = ["House","Apartment","Others"]
        property_type = data_dict["property_type"]
        if property_type not in valid_types :
            return None,f"Invalid property type. Must be one of {valid_types}."
        
        # rooms_number cannot less than 0
        rooms = data_dict["rooms_number"]
        if rooms < 0 :
            return None,"Number of rooms cannot be negative."
        
        # postcode is four numbers
        zip_code = data_dict["zip_code"]
        if zip_code < 1000 or zip_code > 9999 :
            return None,"Invalid Belgian postal code(must be 4 digits between 1000 to 9999)."
        
        print("✅verifying successful!")

        # 3. transfer data_dict to format of model training
        # 3.1 basic mandatory fields transfer
        model_data = {
            "habitablesurface": int(area),
            "bedroomcount": int(rooms)
        }

        # 3.2 property type transfer
        property_type_map = {
            "APARTMENT":0,
            "HOUSE":1,
            "OTHERS":0
        }
        model_data["type_encoded"] = property_type_map.get(property_type.upper(),0)

        # 初始化warnings列表
        warnings = []
        
        # 3.3  building state (OPTIONAL)transfer
        building_state_map ={
            "NEW":0,
            "TO RENOVATE":1,
            "GOOD":2,
            "TO BE DONE UP":3,
            "JUST RENOVATED":4,
            "TO REBUILD":5
        }
        building_state = data_dict.get("building_state",None)
        if building_state is None:
            model_data["buildingcondition_encoded"] = 2 # default "GOOD"
            warnings.append("⚠️ Building state is not provided. Using 'GOOD' as defualt. This may affect price accuracy.")
        else:
            if building_state not in building_state_map:
                return None,f"Invalid building state:{building_state}. Must be one of {list(building_state_map.keys())}"
            model_data["buildingcondition_encoded"] = building_state_map[building_state]

        # 3.4 epc score(optional) transfer
        epc_score_map = {
            "A++":0,
            "A+":1,
            "A":2,
            "B":3,
            "C":4,
            "D":5,
            "E":6,
            "F":7,
            "G":8
        }
        epc_score = data_dict.get("epc_score",None)
        if epc_score is None:
            model_data["epcscore_encoded"] = 4 #defaul epc:C
            warnings.append("⚠️ EPC score is not provided. Using 'C' as default. This may affect price accuracy.")
        else:
            if epc_score not in epc_score_map:
                return None,f"Invalid EPC score:{epc_score}. Must be one of {list(epc_score_map.keys())}"
            model_data["epcscore_encoded"] = epc_score_map[epc_score]

        # 3.5 optional fields transfer
        model_data["hasgarden"] = 1 if data_dict.get("garden",False) else 0
        model_data["haslift"] = 1 if data_dict.get("lift",False) else 0
        model_data["hasswimmingpool"] = 1 if data_dict.get("swimmingpool",False) else 0
        model_data["hasterrace"] = 1 if data_dict.get("terrace",False) else 0
        model_data["hasparking"] = 1 if data_dict.get("parking",False) else 0

        # 3.6 zip code (mandatory) tranfer
        postcode = data_dict["zip_code"]
        geo_data_used = False

        try:
            # using georef data
            geo_file = "Aperol project/data/georef-belgium-postal-codes@public.csv"
            
            if os.path.exists(geo_file):
                df_geo = pd.read_csv(geo_file,sep=";",encoding="utf-8")

                #find postcode
                geo_row = df_geo[df_geo["Post code"]==postcode]
                
                if not geo_row.empty:
                    # extract geo point
                    geo_point = geo_row['Geo Point'].iloc[0]
                    latitude,longitude = map(float,geo_point.split(','))

                    # extract region. French to english (mapping)
                    region_french = geo_row['Région name (French)'].iloc[0]
                    region_mapping = {
                        'Région de Bruxelles-Capitale':'Brussels',
                        'Région flamande':'Flanders',  
                        'Région wallonne':'Wallonia'   
                    }

                    region = region_mapping.get(region_french,None)

                    if region:
                        model_data['latitude'] = latitude
                        model_data['longitude'] = longitude
                        model_data['region_Brussels'] = 1 if region == 'Brussels' else 0
                        model_data['region_Flanders'] = 1 if region == 'Flanders' else 0  
                        model_data['region_Wallonia'] = 1 if region == 'Wallonia' else 0  
                        print(f'✅ Using geo data:{postcode} -> {latitude:.4f},{longitude:.4f} ({region})')
                        geo_data_used = True

                    else:
                        warnings.append(f'Unknown region: {region_french}. Using approximate location.')
                        geo_data_used = False
                else:
                    warnings.append(f'Post code {postcode} not found in geo file. Using approximate location.')
                    geo_data_used = False
            else:
                warnings.append('Geo file not found. Using approximate location.')
                geo_data_used = False

        except Exception as e:
            warnings.append(f'Error reading geo file: {str(e)}. Using approximate location.')
            geo_data_used = False

        # Plan B: Using approximate geo data when geo point is not working 
        if not geo_data_used:
            if 1000 <= postcode <= 1299 :
                region, latitude, longitude = "Brussels", 50.8503, 4.3517  
            elif ((1300 <= postcode <= 1499) or (2000 <= postcode <= 2999) 
                or (3000 <= postcode <= 3999) or (8000 <= postcode <= 8999) 
                or (9000 <= postcode <= 9999)):
                region, latitude, longitude = "Flanders", 51.0, 4.5  
            else:
                region, latitude, longitude = "Wallonia", 50.5, 5.0  

            model_data['latitude'] = latitude
            model_data['longitude'] = longitude
            model_data['region_Brussels'] = 1 if region == 'Brussels' else 0
            model_data['region_Flanders'] = 1 if region == 'Flanders' else 0
            model_data['region_Wallonia'] = 1 if region == 'Wallonia' else 0
            
            print(f"📍 Using Plan B (approximate location):{postcode} -> {latitude},{longitude},{region}")

        # create the final Dataframe
        df = pd.DataFrame([model_data])

        model_required_features = [
            'bedroomcount', 'habitablesurface', 'haslift', 'hasgarden', 
            'hasswimmingpool', 'hasterrace', 'hasparking', 'epcscore_encoded',
            'buildingcondition_encoded', 'region_Brussels', 'region_Flanders',
            'region_Wallonia', 'type_encoded', 'latitude', 'longitude'
        ]

        def get_default_value(feature) :
            # return suitable default value through feature's type
            int_features = {
            'haslift', 'hasgarden', 'hasswimmingpool', 'hasterrace', 
            'hasparking', 'type_encoded', 'region_Brussels', 
            'region_Flanders', 'region_Wallonia' 
            }
            if feature in int_features:
                return 0
            else:
                return 0.0
            
        # make sure all features existing
        for feature in model_required_features :
            if feature not in df.columns :
                df[feature] = get_default_value(feature)
            
        df_final = df[model_required_features]

        print('✅ Preprocessing finished.')
        print(f'✅ Shape output: {df_final.shape}')

        # return result
        if warnings:
            warning_message = "|".join(warnings)
            return df_final, warning_message
        else:
            return df_final, None
        
    except Exception as e:
        return None,f'Error during preprocessing: {str(e)}'
    


def test_with_real_data():
    """
    Test preprocess function with real data
    """
    
    print("=== Test preprocess function ===\n")
    # test cases
    test_cases = [
        {
            "name": "Data of Brusels",
            "data": {
                "area": 120,
                "property_type": "House",
                "rooms_number": 3,
                "zip_code": 1000,
                "building_state": "GOOD",
                "epc_score": "B",
                "garden": True
            }
        },
        {
            "name": "Minimum data of Antwerp ",
            "data": {
                "area": 100,
                "property_type": "Apartment",
                "rooms_number": 2,
                "zip_code": 2000
            }
        },
        {
            "name": "Data of wallonia",
            "data": {
                "area": 150,
                "property_type": "House",
                "rooms_number": 4,
                "zip_code": 5000,
                "building_state": "JUST RENOVATED"
            }
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"Test {i}: {case['name']}")
        print(f"input: {case['data']}")
        
        result, warning = preprocess(case['data'])
        
        if result is not None:
            print("✅ Preprocessing successfully")
            lat = result['latitude'].iloc[0]
            lon = result['longitude'].iloc[0]
            
            # Determine region
            if result['region_Brussels'].iloc[0] == 1:
                region = "Brussels"
            elif result['region_Flanders'].iloc[0] == 1:
                region = "Flanders"
            elif result['region_Wallonia'].iloc[0] == 1:
                region = "Wallonia"
            else:
                region = "Unknown"
            
            print(f"📍 Geo information: ({lat:.4f}, {lon:.4f}) - {region}")
            print(f" Amount of features: {result.shape[1]}")
            
            if warning:
                print(f"⚠️ warning: {warning}")
        else:
            print(f"❌ Preprocessing failed: {warning}")
        
        print("-" * 60)


if __name__ == "__main__":
    test_with_real_data()