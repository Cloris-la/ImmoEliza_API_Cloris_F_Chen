import os
import sys
import pandas as pd
import numpy as np
import joblib
import unittest
import traceback

# Temporarily add current directory to Python path
sys.path.insert(0, os.getcwd())

# Import your preprocessing module
try:
    import cleaning_data as cd
except ImportError:
    print("Please ensure cleaning_data.py is in the current directory")
    exit(1)

class ComprehensiveCleaningDataTest(unittest.TestCase):
    """Integrated test suite for cleaning_data module"""
    
    @classmethod
    def setUpClass(cls):
        """Create test environment and mock data"""
        # Create temporary directory for test files
        os.makedirs("test_temp", exist_ok=True)
        
        # Create mock geographical data file
        geo_data = """Post code;Geo Point;Région name (French)
        1000;50.8730,4.3752;Région de Bruxelles-Capitale
        2000;51.2484,4.3761;Région flamande
        5000;50.4578,4.8545;Région wallonne"""
        with open("test_temp/geo_data.csv", "w", encoding="utf-8") as f:
            f.write(geo_data)
        
        # Create mock training data - using correct field names
        cls.training_data = pd.DataFrame({
            "property_type": ["APARTMENT", "HOUSE", "OTHERS"] * 10,
            "building_state": ["NEW", "GOOD", "TO RENOVATE"] * 10,
            "epcscore": ["A", "B", "C"] * 10,
            "lift": [True, False, True] * 10,
            "garden": [False, True, False] * 10,
            "zip_code": [1000, 2000, 3000] * 10,
            "habitablesurface": [80, 120, 100] * 10,
            "bedroomcount": [2, 3, 2] * 10
        })
        
        # Define test cases
        cls.test_cases = [
            # Complete Brussels data
            {
                "name": "Complete Brussels data",
                "input": {
                    "area": 120, 
                    "property-type": "HOUSE", 
                    "bedroom-number": 3, 
                    "zip-code": 1000,
                    "building-state": "GOOD",
                    "epc_score": "B",
                    "garden": True,
                    "swimming-pool": True
                },
                "expected": {
                    "region_Brussels": 1,
                    "hasgarden": 1,
                    "hasswimmingpool": 1,
                    "type_encoded": 1
                }
            },
            # Minimal Antwerp data
            {
                "name": "Minimal Antwerp data",
                "input": {
                    "area": 100, 
                    "property-type": "APARTMENT", 
                    "bedroom-number": 2, 
                    "zip-code": 2000
                },
                "expected": {
                    "region_Flanders": 1,
                    "epcscore_encoded": 4,  # Default C rating
                    "buildingcondition_encoded": 2  # Default GOOD
                }
            },
            # Wallonia data
            {
                "name": "Wallonia data",
                "input": {
                    "area": 150, 
                    "property-type": "HOUSE", 
                    "bedroom-number": 4, 
                    "zip-code": 5000,
                    "building-state": "JUST RENOVATED"
                },
                "expected": {
                    "region_Wallonia": 1,
                    "buildingcondition_encoded": 4
                }
            },
            # Boundary value test
            {
                "name": "Boundary value test",
                "input": {
                    "area": 1,  # Minimum area
                    "property-type": "APARTMENT", 
                    "bedroom-number": 0,  # Zero rooms
                    "zip-code": 9999  # Maximum postal code
                },
                "expected": {
                    "habitablesurface": 1,
                    "bedroomcount": 0
                }
            },
            # Invalid data test
            {
                "name": "Invalid data test",
                "input": {
                    "area": -10,  # Invalid area
                    "property-type": "CASTLE",  # Invalid type
                    "bedroom-number": 2, 
                    "zip-code": 123
                },
                "should_fail": True,
                "expected_error": "Invalid property type|Area must be greater|Invalid Belgian postal code"
            }
        ]
    
    def test_01_data_validator(self):
        """Test data validator"""
        print("\n=== Testing Data Validator ===")
        
        # Valid data test
        valid_data = {
            "area": 100, 
            "property-type": "APARTMENT", 
            "bedroom-number": 2, 
            "zip-code": 1000
        }
        is_valid, error = cd.DataValidator.validation_input(valid_data)
        self.assertTrue(is_valid)
        self.assertIsNone(error)
        print("✅ Valid data validation passed")
        
        # Invalid data tests
        test_cases = [
            ({"property-type": "APARTMENT", "bedroom-number": 2, "zip-code": 1000}, "area"),
            ({"area": 100, "bedroom-number": 2, "zip-code": 1000}, "property-type"),
            ({"area": 100, "property-type": "APARTMENT", "zip-code": 1000}, "bedroom-number"),
            ({"area": 100, "property-type": "APARTMENT", "bedroom-number": 2}, "zip-code"),
            ({"area": -10, "property-type": "APARTMENT", "bedroom-number": 2, "zip-code": 1000}, "Area must be greater"),
            ({"area": 100, "property-type": "CASTLE", "bedroom-number": 2, "zip-code": 1000}, "Invalid property type"),
            ({"area": 100, "property-type": "APARTMENT", "bedroom-number": -1, "zip-code": 1000}, "Number of rooms cannot be negative"),
            ({"area": 100, "property-type": "APARTMENT", "bedroom-number": 2, "zip-code": 999}, "Invalid Belgian postal code")
        ]
        
        for data, expected_error in test_cases:
            is_valid, error = cd.DataValidator.validation_input(data)
            self.assertFalse(is_valid)
            self.assertIn(expected_error, error)
        
        print("✅ All invalid data tests passed")
    
    def test_02_input_cleaner(self):
        """Test input cleaner"""
        print("\n=== Testing Input Cleaner ===")
        
        input_data = {
            "area": 120, 
            "property-type": "HOUSE", 
            "bedroom-number": 3, 
            "zip-code": 1000,
            "garden": True,
            "swimming-pool": False
        }
        
        df = cd.InputCleaner.json_to_dataframe(input_data)
        
        # Verify transformation results
        self.assertEqual(df.shape, (1, 11))
        self.assertEqual(df["habitablesurface"].iloc[0], 120)
        self.assertEqual(df["property_type"].iloc[0], "HOUSE")
        self.assertEqual(df["garden"].iloc[0], True)
        self.assertEqual(df["swimmingpool"].iloc[0], False)
        
        # Verify epc_score to epcscore mapping
        input_data_with_epc = {**input_data, "epc_score": "B"}
        df_with_epc = cd.InputCleaner.json_to_dataframe(input_data_with_epc)
        self.assertEqual(df_with_epc["epcscore"].iloc[0], "B")
        
        print("✅ Input cleaner test passed")
    
    def test_03_individual_transformers(self):
        """Test individual transformers"""
        print("\n=== Testing Individual Transformers ===")
        
        # Create test data - using correct field names
        test_data = pd.DataFrame({
            "property_type": ["APARTMENT", "HOUSE", "OTHERS"],
            "building_state": ["NEW", None, "TO RENOVATE"],
            "epcscore": ["A", None, "G"],
            "lift": [True, False, None],
            "garden": [False, True, False]
        })
        
        # Test property type encoder
        property_encoder = cd.PropertyTypeEncoder()
        transformed = property_encoder.transform(test_data)
        self.assertEqual(transformed["type_encoded"].tolist(), [0, 1, 0])
        
        # Test building state encoder
        building_encoder = cd.BuildingStateEncoder()
        transformed = building_encoder.transform(test_data)
        self.assertEqual(transformed["buildingcondition_encoded"].tolist(), [0, 2, 1])
        
        # Test EPC score encoder
        epc_encoder = cd.EPCScoreEncoder()
        transformed = epc_encoder.transform(test_data)
        
        # Print actual values for debugging
        print("EPC encoder actual output:", transformed["epcscore_encoded"].tolist())
        self.assertEqual(transformed["epcscore_encoded"].tolist(), [2, 4, 8])
        
        # Test boolean feature encoder
        boolean_encoder = cd.BooleanFeatureEncoder()
        transformed = boolean_encoder.transform(test_data)
        self.assertEqual(transformed["haslift"].tolist(), [1, 0, 0])
        self.assertEqual(transformed["hasgarden"].tolist(), [0, 1, 0])
        
        print("✅ All individual transformer tests passed")
    
    def test_04_geographic_encoder(self):
        """Test geographic encoder"""
        print("\n=== Testing Geographic Encoder ===")
        
        # Use mock geography file
        encoder = cd.GeographicEncoder(geo_file_path="test_temp/geo_data.csv")
        
        # Test exact matches
        test_data = pd.DataFrame({
            "zip_code": [1000, 2000, 5000]
        })
        result = encoder.transform(test_data)
        
        # Verify results
        self.assertAlmostEqual(result["latitude"].iloc[0], 50.8730, places=4)
        self.assertAlmostEqual(result["longitude"].iloc[0], 4.3752, places=4)
        self.assertEqual(result["region_Brussels"].iloc[0], 1)
        self.assertEqual(result["region_Flanders"].iloc[1], 1)
        self.assertEqual(result["region_Wallonia"].iloc[2], 1)
        
        # Test fallback (postal codes not in file)
        test_data_fallback = pd.DataFrame({
            "zip_code": [3000, 4000, 6000]
        })
        result_fallback = encoder.transform(test_data_fallback)
        
        # Verify fallback results
        self.assertEqual(result_fallback["region_Flanders"].iloc[0], 1)  # 3000 in Flanders
        self.assertEqual(result_fallback["region_Wallonia"].iloc[1], 1)  # 4000 in Wallonia
        self.assertEqual(result_fallback["region_Wallonia"].iloc[2], 1)  # 6000 in Wallonia
        
        print("✅ Geographic encoder test passed")
    
    def test_05_full_pipeline(self):
        """Test complete preprocessing pipeline"""
        print("\n=== Testing Complete Pipeline ===")
        
        # Create and save pipeline
        pipeline = cd.create_preprocessing_pipeline()
        pipeline.fit(self.training_data)
        cd.save_preprocessing_pipeline(pipeline, "test_temp/pipeline.pkl")
        
        # Load pipeline
        loaded_pipeline = cd.load_preprocessing_pipeline("test_temp/pipeline.pkl")
        self.assertIsNotNone(loaded_pipeline)
        
        # Test all cases
        for case in self.test_cases:
            print(f"\nTest case: {case['name']}")
            
            if case.get("should_fail"):
                # Test expected failure
                result, error = cd.preprocess(case["input"])
                self.assertIsNone(result)
                self.assertIsNotNone(error)
                
                # Verify error message
                if "expected_error" in case:
                    self.assertTrue(
                        any(e in error for e in case["expected_error"].split("|")),
                        f"Error message mismatch: {error}"
                    )
                print(f"✅ Expected failure test passed: {error}")
            else:
                # Test expected success
                result, error = cd.preprocess(case["input"])
                
                # Print detailed error if failed
                if result is None:
                    print(f"❌ Preprocessing failed: {error}")
                    traceback.print_exc()
                
                self.assertIsNotNone(result, f"Preprocessing failed: {error}")
                self.assertIsNone(error, f"Preprocessing returned error: {error}")
                
                # Verify results
                self.assertEqual(result.shape, (1, 15), "Incorrect result shape")
                
                # Check expected feature values
                for feature, expected_value in case["expected"].items():
                    actual_value = result[feature].iloc[0]
                    self.assertEqual(
                        actual_value, expected_value,
                        f"Feature {feature} mismatch: expected {expected_value}, got {actual_value}"
                    )
                
                print("✅ Data processing results verified")
    
    def test_06_fit_and_save_pipeline(self):
        """Test pipeline fitting and saving"""
        print("\n=== Testing Pipeline Fitting and Saving ===")
        
        # Call training function
        pipeline = cd.fit_and_save_pipeline(self.training_data)
        
        # Verify pipeline was saved
        self.assertTrue(os.path.exists("model_deployment/preprocessing_pipeline.pkl"))
        
        # Load saved pipeline
        loaded_pipeline = cd.load_preprocessing_pipeline()
        self.assertIsNotNone(loaded_pipeline)
        
        print("✅ Pipeline fitting and saving test passed")
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test environment"""
        # Delete temporary files
        for file in ["test_temp/geo_data.csv", "test_temp/pipeline.pkl"]:
            if os.path.exists(file):
                os.remove(file)
        
        # Delete temporary directory
        if os.path.exists("test_temp"):
            os.rmdir("test_temp")
        
        # Clean model deployment directory
        if os.path.exists("model_deployment/preprocessing_pipeline.pkl"):
            os.remove("model_deployment/preprocessing_pipeline.pkl")
        
        print("\nTest environment cleaned")

if __name__ == "__main__":
    # Run all tests
    unittest.main(verbosity=2)