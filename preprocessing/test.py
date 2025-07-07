# test_preprocessing.py
import pandas as pd
import numpy as np
from cleaning_data import preprocess, fit_and_save_pipeline, create_preprocessing_pipeline

def test_api_format_compliance():
    """Test whether input and output formats comply with API requirements"""
    
    print("=== 🧪 Testing API Format Compliance ===\n")
    
    # Test Case 1: Complete data (valid JSON format required by API)
    test_case_1 = {
        "area": 120,
        "property-type": "HOUSE",      # API format: hyphen
        "rooms-number": 3,             # API format: hyphen
        "zip-code": 1000,              # API format: hyphen
        "building-state": "GOOD",      # API format: hyphen
        "garden": True,
        "swimming-pool": False,        # API format: hyphen
        "terrace": True,
        "parking": True
    }
    
    # Test Case 2: Minimum required data
    test_case_2 = {
        "area": 80,
        "property-type": "APARTMENT",
        "rooms-number": 2,
        "zip-code": 2000
    }
    
    # Test Case 3: Invalid data (for validation testing)
    test_case_3 = {
        "area": -50,  # Error: negative value
        "property-type": "CASTLE",  # Error: invalid type
        "rooms-number": 2,
        "zip-code": 12345  # Error: 5-digit code
    }
    
    test_cases = [
        ("Complete Data", test_case_1),
        ("Minimum Data", test_case_2),
        ("Invalid Data", test_case_3)
    ]
    
    for name, test_data in test_cases:
        print(f"🔍 Test: {name}")
        print(f"Input: {test_data}")
        
        result, error = preprocess(test_data)
        
        if result is not None:
            print("✅ Preprocessing succeeded")
            print(f"📊 Output shape: {result.shape}")
            print(f"📋 Output columns: {list(result.columns)}")
            print(f"🔢 Sample data:")
            print(result.iloc[0].to_dict())
            print()
        else:
            print(f"❌ Preprocessing failed: {error}")
            print()
        
        print("-" * 60)

def test_csv_format_compliance():
    """Test whether output format is consistent with training CSV format"""
    
    print("=== 🧪 Testing CSV Format Consistency ===\n")
    
    # Expected columns from training CSV (based on your data_cleaned.csv info)
    expected_columns = [
        'bedroomcount', 'habitablesurface', 'haslift', 'hasgarden', 
        'hasswimmingpool', 'hasterrace', 'hasparking', 'epcscore_encoded',
        'buildingcondition_encoded', 'region_Brussels', 'region_Flanders',
        'region_Wallonia', 'type_encoded', 'latitude', 'longitude'
    ]
    
    # Test data
    test_data = {
        "area": 100,
        "property-type": "HOUSE",
        "rooms-number": 3,
        "zip-code": 1000,
        "building-state": "GOOD",
        "garden": True,
        "swimming-pool": False,
        "terrace": True,
        "parking": False
    }
    
    result, error = preprocess(test_data)
    
    if result is not None:
        actual_columns = list(result.columns)
        
        print(f"📋 Expected columns ({len(expected_columns)}): {expected_columns}")
        print(f"📋 Actual columns ({len(actual_columns)}): {actual_columns}")
        
        # Check for exact column match
        missing_columns = set(expected_columns) - set(actual_columns)
        extra_columns = set(actual_columns) - set(expected_columns)
        
        if missing_columns:
            print(f"❌ Missing columns: {missing_columns}")
        
        if extra_columns:
            print(f"⚠️ Extra columns: {extra_columns}")
        
        if not missing_columns and not extra_columns:
            print("✅ Columns match exactly!")
        
        # Check data types
        print(f"\n📊 Data types:")
        for col in result.columns:
            print(f"  {col}: {result[col].dtype}")
        
        # Show actual values
        print(f"\n🔢 Actual values:")
        for col in result.columns:
            print(f"  {col}: {result[col].iloc[0]}")
    
    else:
        print(f"❌ Test failed: {error}")

def create_and_save_test_pipeline():
    """Create and save a test pipeline"""
    
    print("=== 🧪 Creating Test Pipeline ===\n")
    
    # Create some dummy training data to fit the pipeline
    dummy_training_data = pd.DataFrame({
        'habitablesurface': [100, 120, 80],
        'bedroomcount': [2, 3, 1],
        'property_type': ['HOUSE', 'APARTMENT', 'HOUSE'],
        'zip_code': [1000, 2000, 5000],
        'building_state': ['GOOD', 'NEW', 'TO RENOVATE'],
        'epcscore': ['C', 'B', 'D'],
        'garden': [True, False, True],
        'swimmingpool': [False, False, False],
        'terrace': [True, True, False],
        'parking': [True, False, True],
        'lift': [False, True, False]
    })
    
    try:
        pipeline = fit_and_save_pipeline(dummy_training_data)
        print("✅ Pipeline created and saved successfully!")
        return True
    except Exception as e:
        print(f"❌ Pipeline creation failed: {e}")
        return False

def test_api_output_format():
    """Test whether API output format meets requirements"""
    
    print("=== 🧪 Testing API Output Format ===\n")
    
    # Expected API output format
    expected_api_output = {
        "prediction": "float or None",
        "status_code": "int or None"
    }
    
    print(f"📋 Expected API output format: {expected_api_output}")
    
    # Simulated output from predict function (to be implemented next)
    test_data = {
        "area": 120,
        "property-type": "HOUSE",
        "rooms-number": 3,
        "zip-code": 1000
    }
    
    processed_data, error = preprocess(test_data)
    
    if processed_data is not None:
        print("✅ Preprocessing succeeded, data ready for model prediction")
        print(f"📊 Preprocessed data shape: {processed_data.shape}")
        
        # Simulated successful API response
        mock_success_response = {
            "prediction": 350000.0,  # simulated predicted price
            "status_code": 200
        }
        
        # Simulated error API response
        mock_error_response = {
            "prediction": None,
            "status_code": 400
        }
        
        print(f"✅ Success response format: {mock_success_response}")
        print(f"❌ Error response format: {mock_error_response}")
        
    else:
        print(f"❌ Preprocessing failed: {error}")

if __name__ == "__main__":
    # Run all tests
    
    # 1. Create and save pipeline first (simulate training phase)
    print("🔄 Step 1: Creating Pipeline...")
    pipeline_created = create_and_save_test_pipeline()
    print()
    
    if pipeline_created:
        # 2. Test API format compliance
        print("🔄 Step 2: Testing API format...")
        test_api_format_compliance()
        print()
        
        # 3. Test CSV format consistency
        print("🔄 Step 3: Testing CSV consistency...")
        test_csv_format_compliance()
        print()
        
        # 4. Test API output format
        print("🔄 Step 4: Testing API output format...")
        test_api_output_format()
        print()
        
        print("🎉 All tests completed!")
    else:
        print("💥 Pipeline creation failed, cannot continue testing")
