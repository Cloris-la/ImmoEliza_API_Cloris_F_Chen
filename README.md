# ImmoEliza House Price Prediction API

## 🏠 Project Overview

This is a **machine learning API** for predicting house prices in Belgium, developed for the real estate company "ImmoEliza". The API is built with FastAPI and deployed on Render using Docker.

### 🤝 Team Project Context
This project is part of a **three-person team collaboration**. Each team member developed their own implementation of the same API requirements. [main repository](https://github.com/Cloris-la/challenge-api-deployment). This current API is developed in the `Cloris_F_Chen_Deployment` branch as my individual contribution to the team project.

## 🚀 Live API

**Base URL:** `https://immoeliza-api-cloris-f-chen.onrender.com`

**Interactive Documentation:** `https://immoeliza-api-cloris-f-chen.onrender.com/docs`

## 📋 Available Endpoints

### 1. Health Check
- **Endpoint:** `GET /`
- **Description:** Check if the API server is alive
- **Response:** `"alive"`

### 2. API Information
- **Endpoint:** `GET /predict`
- **Description:** Get detailed information about how to use the prediction endpoint
- **Response:** JSON with API usage guide, required fields, and examples

### 3. House Price Prediction
- **Endpoint:** `POST /predict`
- **Description:** Predict the price of a house based on its characteristics
- **Content-Type:** `application/json`

## 📝 Request Format

### Required Fields
```json
{
  "data": {
    "area": 120,                    // int - Living area in m²
    "property-type": "HOUSE",       // string - "APARTMENT" | "HOUSE" | "OTHERS"
    "bedrooms-number": 3,           // int - Number of bedrooms
    "zip-code": 1000               // int - Belgian postal code (1000-9999)
  }
}
```

### Optional Fields (Improve Prediction Accuracy)
```json
{
  "data": {
    "area": 120,
    "property-type": "HOUSE",
    "bedrooms-number": 3,
    "zip-code": 1000,
    "garden": true,                 // bool - Has garden
    "swimming-pool": false,         // bool - Has swimming pool
    "terrace": true,                // bool - Has terrace
    "building-state": "GOOD",       // string - "NEW" | "GOOD" | "TO RENOVATE" | "JUST RENOVATED" | "TO BE DONE UP" | "TO REBUILD"
    "parking": true,                // bool - Has parking space
    "lift": false,                  // bool - Has elevator
    "epc-score": "C"               // string - "A++" | "A+" | "A" | "B" | "C" | "D" | "E" | "F" | "G"
  }
}
```

## 📤 Response Format

### Successful Prediction
```json
{
  "prediction": 450000.00,
  "status_code": 200
}
```

### Error Response
```json
{
  "prediction": null,
  "status_code": 400  // or 500 for server errors
}
```

## 🔧 Usage Examples

### cURL Example
```bash
curl -X POST "https://immoeliza-api-cloris-f-chen.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "data": {
         "area": 150,
         "property-type": "HOUSE",
         "bedrooms-number": 4,
         "zip-code": 1000,
         "garden": true,
         "building-state": "GOOD"
       }
     }'
```

### Python Example
```python
import requests

url = "https://immoeliza-api-cloris-f-chen.onrender.com/predict"
data = {
    "data": {
        "area": 120,
        "property-type": "APARTMENT",
        "bedrooms-number": 2,
        "zip-code": 2000,
        "terrace": True,
        "epc-score": "B"
    }
}

response = requests.post(url, json=data)
result = response.json()
print(f"Predicted price: €{result['prediction']:,.2f}")
```

### JavaScript Example
```javascript
const response = await fetch('https://immoeliza-api-cloris-f-chen.onrender.com/predict', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    data: {
      area: 100,
      'property-type': 'APARTMENT',
      'bedrooms-number': 2,
      'zip-code': 1050,
      garden: false,
      'swimming-pool': false
    }
  })
});

const result = await response.json();
console.log('Predicted price:', result.prediction);
```

## 🏗️ Technical Architecture

### Project Structure
```
├── app.py                      # FastAPI application
├── Aperol project/data         # Geo point data
│   └── georef-belgium-postal-codes@public.csv
├── preprocessing/
│   └── cleaning_data.py       # Data preprocessing pipeline
├── predict/
│   └── prediction.py          # ML prediction logic
├── model/
│   └── robocop_model.cbm     # Trained CatBoost model
├── requirements.txt           # Python dependencies
└── Dockerfile                # Docker configuration
```

### Key Features
- **FastAPI Framework:** High-performance, automatic API documentation
- **Machine Learning:** CatBoost regression model for price prediction
- **Data Validation:** Pydantic models for request/response validation
- **Docker Deployment:** Containerized application on Render
- **Error Handling:** Comprehensive error responses with status codes
- **Geographic Intelligence:** Region-based features (Brussels, Flanders, Wallonia)

## 🛠️ Technology Stack

- **Backend:** FastAPI (Python)
- **ML Model:** CatBoost Regressor
- **Data Processing:** Pandas, NumPy, Scikit-learn
- **Deployment:** Docker + Render
- **Documentation:** Automatic OpenAPI/Swagger generation

## 📊 Model Features

The prediction model uses the following features:
- **Property characteristics:** Area, bedrooms, property type
- **Location data:** ZIP code, geographic region, coordinates
- **Amenities:** Garden, swimming pool, terrace, parking, elevator
- **Building condition:** Energy performance certificate, building state
- **Encoded categorical variables:** Property type, building condition, EPC score

## 🔍 API Documentation

For complete interactive documentation with the ability to test endpoints directly in your browser, visit:

**📖 [Interactive API Documentation](https://immoeliza-api-cloris-f-chen.onrender.com/docs)**

The documentation includes:
- Complete endpoint descriptions
- Request/response schemas
- Interactive testing interface
- Example requests and responses
- Error code explanations

## 🚨 Important Notes

- **Case Sensitivity:** All string inputs are case-insensitive
- **Required Fields:** `area`, `property-type`, `bedrooms-number`, `zip-code` are mandatory
- **ZIP Code Range:** Must be between 1000-9999 (Belgian postal codes)
- **Response Time:** First request after inactivity may take 30-60 seconds (free tier limitation)
- **Rate Limiting:** Reasonable usage expected for free tier deployment

## 🤝 Team Development

This API represents my individual contribution to a collaborative team project. The implementation focuses on:
- Clean, maintainable code structure
- Comprehensive error handling
- User-friendly API design
- Professional documentation
- Robust data validation

The final production version will be selected based on performance, code quality, and feature completeness.

---

**Developer:** Cloris F. Chen  
**Branch:** `Cloris_F_Chen_Deployment`  
**Deployment:** Render  
**Last Updated:** July 2025