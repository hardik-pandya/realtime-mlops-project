#Fast API inference server for churn prediction model
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Churn Prediction API", 
              description="API for predicting customer churn using a trained model.", 
              version="1.0.0")

#Load model
with open('models/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

class CustomerData(BaseModel):
    age: int
    tenure_months: int
    monthly_charges: float
    total_charges: float
    num_support_calls: int

@app.get("/health")
def health():
    return {"status": "API is healthy"}

@app.post("/predict")
def predict(data: CustomerData):
    # Validate input features
    if data.age < 0 or data.tenure_months < 0 or data.monthly_charges < 0 or data.total_charges < 0 or data.num_support_calls < 0:
        raise HTTPException(status_code=400, detail="All input values must be non-negative.")

    features = np.array(
        [[
            data.age,
            data.tenure_months,
            data.monthly_charges,
            data.total_charges,
            data.num_support_calls
        ]]
    )

    try:
        prediction = model.predict(features)[0]
        probability = float(model.predict_proba(features)[0][1])  # Ensure the probability is a float
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {str(e)}")

    return {
        "prediction": int(prediction),  # 0 or 1 indicating no churn or churn
        "probability_of_churn": probability
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
