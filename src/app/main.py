from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Pitcher Velo Decline Predictor")

class PitcherInput(BaseModel):
    pitcher_id: int
    avg_velo: float
    max_velo: float
    spin_diff: float
    stress_pitches: int
    days_since_last: int

model = joblib.load("/app/models/velocity_model.pkl")

@app.post("/predict_decline")
async def predict_decline(input: PitcherInput):
    features = np.array([[input.avg_velo, input.max_velo, 
                         input.spin_diff, input.stress_pitches]])
    proba = model.predict_proba(features)[0][1]
    return {
        "pitcher_id": input.pitcher_id,
        "decline_prob": round(proba, 3),
        "risk_level": "High" if proba > 0.7 else "Medium" if proba > 0.4 else "Low"
    }
