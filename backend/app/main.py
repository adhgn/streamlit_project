import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import uvicorn

app = FastAPI(title="UT Robust Failure Engine")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None
# We will use this clean list for our internal model tracking
FEATURE_COLUMNS = ['air_temperature', 'process_temperature', 'rotational_speed', 'torque', 'tool_wear']

class KaggleTelemetry(BaseModel):
    air_temperature: float
    process_temperature: float
    rotational_speed: float
    torque: float
    tool_wear: float

@app.on_event("startup")
def train_aligned_model():
    """Loads local data safely matching the exact Kaggle CSV column naming convention"""
    global MODEL
    print("Loading local dataset...")
    
    paths_to_try = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "predictive_maintenance.csv"),
        "predictive_maintenance.csv",
        "backend/app/predictive_maintenance.csv"
    ]
    
    csv_path = None
    for p in paths_to_try:
        if os.path.exists(p):
            csv_path = p
            break
            
    try:
        if not csv_path:
            raise FileNotFoundError("Could not find predictive_maintenance.csv.")
            
        df = pd.read_csv(csv_path)
        
        # 🔄 Map your exact screenshot columns to our internal lowercase schema
        rename_map = {
            'Air temperature': 'air_temperature',
            'Process temperature': 'process_temperature',
            'Rotational speed': 'rotational_speed',
            'Torque': 'torque',
            'Tool wear': 'tool_wear',
            'machine_failure': 'machine_failure'
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Extract features and targets cleanly
        X = df[FEATURE_COLUMNS]
        y = df['machine_failure']
        
        print(f"Dataset successfully matched ({len(X)} instances). Training Model...")
        
        MODEL = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, class_weight="balanced")
        MODEL.fit(X, y)
        print("Predictive Maintenance Engine is completely LIVE (Running on Port 8000)!")
        
    except Exception as e:
        print(f"❌ CRITICAL ERROR: {str(e)}")

@app.post("/predict")
def predict_failure(data: KaggleTelemetry):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Model artifact is offline.")
    
    input_df = pd.DataFrame([{
        'air_temperature': data.air_temperature,
        'process_temperature': data.process_temperature,
        'rotational_speed': data.rotational_speed,
        'torque': data.torque,
        'tool_wear': data.tool_wear
    }])
    
    predicted_class = int(MODEL.predict(input_df)[0])
    probabilities = MODEL.predict_proba(input_df)[0]
    failure_probability = float(probabilities[1])
    
    # 📊 Get the mathematical weight the Random Forest assigned to each feature
    importances = MODEL.feature_importances_
    
    # Scale the metrics dynamically based on slider values so changes are clearly visible
    risk_contributors = {
        "Thermal Stress (Air)": round(float((data.air_temperature - 295) * importances[0] * 10), 2),
        "Thermal Stress (Process)": round(float((data.process_temperature - 305) * importances[1] * 10), 2),
        "Rotational Stress Index": round(float((data.rotational_speed / 2900) * importances[2] * 100), 2),
        "Torque Strain Metric": round(float((data.torque / 77) * importances[3] * 100), 2),
        "Tool Fatigue Level": round(float((data.tool_wear / 250) * importances[4] * 100), 2),
    }
    
    return {
        "failure_risk_detected": predicted_class == 1,
        "failure_probability_percent": round(failure_probability * 100, 2),
        "risk_factors": risk_contributors
    }

# This forces the app onto port 8000 to bypass the Windows socket lock
if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8002, reload=False)