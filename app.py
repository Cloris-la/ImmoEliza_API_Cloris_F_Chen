from fastapi import FastAPI, HTTPException
from preprocessing.cleaning_data import preprocess
from predict.prediction import predict

app = FastAPI()

@app.get("/")
def health_check():
    return {"message": "alive"}

@app.get("/predict")
def explain_prediction_format():
    return {
        "message": "POST to /predict with JSON like: { 'data': { 'area': 120, ... } }"
    }

@app.post("/predict")
def predict_route(request: dict):
    try:
        house_data = request.get("data")
        if not house_data:
            raise HTTPException(status_code=400, detail="Missing 'data' field")

        processed = preprocess(house_data)
        prediction = predict(processed)

        return {"prediction": prediction, "status_code": 200}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception:
        raise HTTPException(status_code=500, detail="Internal server error")
