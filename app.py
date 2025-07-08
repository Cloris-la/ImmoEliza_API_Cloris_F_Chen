import sys
import os
import traceback

print("🚀 Starting ImmoEliza API...")

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field, field_validator
    from typing import Optional
    import uvicorn
    
    app = FastAPI(
        title="ImmoEliza House Price Prediction API",
        description="Simplified API - Contains only fields actually used by the model",
        version="2.0.0"
    )
    
    # Only fields actually needed by the model
    class HouseData(BaseModel):
        # Required fields
        area: int = Field(..., 
                        description="Living area in square meters (m²)",
                        examples= 100
                        )  
        
        property_type: str = Field(..., alias="property-type", 
                                 description="Property type. Must be: HOUSE, APARTMENT, or OTHERS (case insensitive)", 
                                 example="HOUSE")
        
        bedrooms_number: int = Field(..., alias="bedrooms-number", 
                                   description="Number of bedrooms", 
                                   example=3
                                   )  
        
        zip_code: int = Field(..., alias="zip-code", 
                            description="Belgian postal code. Format: 4-digit number (1000-9999)", 
                            example=1000,
                            ge=1000, le=9999)  # Strict range limit
        
        # Optional fields - Only those actually used by the model
        garden: Optional[bool] = Field(None, 
                                     description="Has garden. true=has garden, false=no garden", 
                                     example=True)
        
        swimming_pool: Optional[bool] = Field(None, alias="swimming-pool", 
                                            description="Has swimming pool. true=has pool, false=no pool", 
                                            example=False)
        
        terrace: Optional[bool] = Field(None, 
                                      description="Has terrace/balcony. true=has terrace, false=no terrace", 
                                      example=True)
        
        parking: Optional[bool] = Field(None, 
                                      description="Has parking space. true=has parking, false=no parking", 
                                      example=True)
        
        lift: Optional[bool] = Field(None, 
                                   description="Has elevator/lift. true=has elevator, false=no elevator", 
                                   example=False)
        
        building_state: Optional[str] = Field(None, alias="building-state",
                                            description="Building condition. Must be: NEW, GOOD, TO RENOVATE, JUST RENOVATED, TO BNE DONE UP, or TO REBUILD (case insensitive)",
                                            example="GOOD")
        
        epc_score: Optional[str] = Field(None, alias="epc-score",
                                       description="Energy Performance Certificate score. Must be: A++, A+, A, B, C, D, E, F, G (case insensitive)",
                                       example="C")
        
        # Field validation with clear error messages
        @field_validator('property_type')
        def validate_property_type(cls, v):
            # Convert to uppercase for comparison
            v_upper = v.upper() if isinstance(v, str) else v
            allowed = ['HOUSE', 'APARTMENT', 'OTHERS']
            
            if v_upper not in allowed:
                raise ValueError(f'Property type must be one of: HOUSE, APARTMENT, or OTHERS (case insensitive). You entered: "{v}"')
            return v_upper
        
        @field_validator('building_state')
        def validate_building_state(cls, v):
            if v is None:
                return v
            
            # Convert to uppercase for comparison
            v_upper = v.upper() if isinstance(v, str) else v
            allowed = ['NEW', 'GOOD', 'TO RENOVATE', 'JUST RENOVATED', 'TO BE DONE UP','TO REBUILD']
            
            if v_upper not in allowed:
                allowed_readable = ['NEW', 'GOOD', 'TO RENOVATE', 'JUST RENOVATED','TO BE DONE UP', 'TO REBUILD']
                raise ValueError(f'Building state must be one of: {", ".join(allowed_readable)} (case insensitive). You entered: "{v}"')
            return v_upper
        
        @field_validator('epc_score')
        def validate_epc_score(cls, v):
            if v is None:
                return v
            
            # Convert to uppercase for comparison
            v_upper = v.upper() if isinstance(v, str) else v
            allowed = ['A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G']
            
            if v_upper not in allowed:
                raise ValueError(f'EPC score must be one of: {", ".join(allowed)} (case insensitive). You entered: "{v}"')
            return v_upper
    
    class PredictionRequest(BaseModel):
        data: HouseData
    
    class PredictionResponse(BaseModel):
        prediction: Optional[float] = Field(None, description="Predicted price in EUR")
        status_code: Optional[int] = Field(None, description="Status code: 200=success, 400=input error, 500=server error")

    @app.get("/")
    def root():
        """
        Health check
        """
        return "alive"

    @app.get("/test")
    def test_imports():
        """
        Test if imports work
        """
        try:
            from preprocessing.cleaning_data import preprocess
            from predict.prediction import predict
            return {"status": "success", "message": "All imports working"}
        except Exception as e:
            return {"status": "error", "message": str(e), "traceback": traceback.format_exc()}
    
    @app.get("/predict")
    def predict_info():
        """
        API Usage Guide
        """
        return {
            "message": "Use POST method to predict house prices",
            "version": "2.0.0 - Simplified API",
            "description": "Contains only fields actually used by the model for a cleaner user experience",
            
            "required_fields": {
                "area": "int - Living area in square meters (m²)",
                "property-type": "string - Property type: HOUSE | APARTMENT | OTHERS",
                "bedrooms-number": "int - Number of bedrooms",
                "zip-code": "int - Belgian postal code (1000-9999)"
            },
            
            "optional_fields": {
                "garden": "bool - Has garden (affects prediction)",
                "swimming-pool": "bool - Has swimming pool (affects prediction)",
                "terrace": "bool - Has terrace (affects prediction)",
                "parking": "bool - Has parking space (affects prediction)",
                "lift": "bool - Has elevator (affects prediction)",
                "building-state": "string - Building condition: NEW | GOOD | TO RENOVATE | JUST RENOVATED | TO BE DONE UP | TO REBUILD (affects prediction)"
            }
        }
    
    @app.post("/predict", response_model=PredictionResponse)
    def predict_house_price(request: PredictionRequest):
        """
        Predict house price
        """
        try:
            # 延迟导入 - 只在实际使用时导入
            from preprocessing.cleaning_data import preprocess
            from predict.prediction import predict
            
            house_data = request.data
            house_dict = house_data.model_dump(by_alias=True)
            
            print(f"🔄 Received prediction request (simplified): {house_dict}")
            
            # Preprocessing
            processed_data, preprocess_error = preprocess(house_dict)
            
            if processed_data is None:
                print(f"❌ Preprocessing failed: {preprocess_error}")
                return PredictionResponse(prediction=None, status_code=400)
            
            print(f"✅ Preprocessing successful: {processed_data.shape}")
            
            # Prediction
            predicted_price = predict(processed_data)
            
            if predicted_price is None:
                print("❌ Prediction failed")
                return PredictionResponse(prediction=None, status_code=500)
            
            print(f"✅ Prediction successful: €{predicted_price:,.2f}")
            
            return PredictionResponse(
                prediction=round(predicted_price, 2),
                status_code=200
            )
            
        except Exception as e:
            print(f"❌ Error: {e}")
            traceback.print_exc()
            return PredictionResponse(prediction=None, status_code=500)
    
    print("✅ API configuration completed")
    
    if __name__ == "__main__":
        port = int(os.environ.get("PORT", 10000))
        print(f"🎯 Starting server on port {port}")
        print(f"📖 Visit http://localhost:{port}/docs for API documentation")
        print("🎉 Users now only need to input fields actually used by the model!")
        
        uvicorn.run(app, host="0.0.0.0", port=port, reload=False)

except Exception as e:
    print(f"💥 Startup failed: {e}")
    traceback.print_exc()