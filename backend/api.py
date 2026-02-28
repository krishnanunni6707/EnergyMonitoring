import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import io
import os
import uvicorn
import json
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# --- 1. DEFINE MODEL ARCHITECTURE ---
class EnergyMonitorLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, num_classes=2):
        super(EnergyMonitorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 2. INITIALIZE API ---
app = FastAPI(title="Energy AI Monitor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. LOAD ASSETS ---
SEQ_LENGTH = 10
MODEL_PATH = "models/energy_lstm.pth"
SCALER_PATH = "models/scaler.pkl"

scaler = None
model = None

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    model = EnergyMonitorLSTM()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("✅ AI Model and Scaler loaded successfully.")
else:
    print("❌ ERROR: Ensure 'models/' folder contains your .pth and .pkl files!")

# --- 4. DATA SCHEMAS ---
# Modified Schema
class ApplianceData(BaseModel):
    appliance_id: str
    sequence: list  # The [voltage, current] pairs

class BatchPredictionRequest(BaseModel):
    appliances: list[ApplianceData]

# --- 5. ENDPOINTS ---
df_sensor = pd.read_csv("live_energy_data.csv")    
sensor_index = 0

@app.get("/sensor-data")
async def get_sensor_data():
    global sensor_index
    # Get the data for the current index for all appliances
    # We assume the CSV has data grouped by timestamp
    
    # Get unique appliances at this "time step"
    # To keep it simple, we return 3 rows (one for each appliance)
    current_batch = df_sensor.iloc[sensor_index : sensor_index + 3]
    
    # Move index forward for the next call (loop back if at end)
    sensor_index = (sensor_index + 3) % len(df_sensor)
    
    results = []
    for _, row in current_batch.iterrows():
        results.append({
            "appliance_id": row['appliance_id'],
            "voltage": float(row['voltage']),
            "current": float(row['current'])
        })
    
    return {"data": results}

@app.get("/")
async def read_index():
    return {"status": "Online", "model_loaded": model is not None}

@app.post("/predict")
async def predict_batch(data: BatchPredictionRequest):
    if model is None or scaler is None:
        return {"error": "Model not loaded"}
    
    results = []

    for item in data.appliances:
        try:
            # --- Existing Logic for one appliance ---
            raw_data = np.array(item.sequence)
            
            # Padding/Truncating to SEQ_LENGTH
            if len(raw_data) < SEQ_LENGTH:
                padding = np.zeros((SEQ_LENGTH - len(raw_data), 2))
                raw_data = np.vstack([padding, raw_data])
            else:
                raw_data = raw_data[-SEQ_LENGTH:]

            # Scale and Predict
            scaled_data = scaler.transform(raw_data)
            input_tensor = torch.tensor([scaled_data], dtype=torch.float32)

            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.softmax(output, dim=1)[0, 1].item()
                prediction = torch.argmax(output, dim=1).item()

            # Append result with the ID
            results.append({
                "appliance_id": item.appliance_id,
                "usage_score": round(prob * 10, 2),
                "status": "Abnormal" if prediction == 1 else "Normal",
                "probability": prob
            })
        except Exception as e:
            results.append({"appliance_id": item.appliance_id, "error": str(e)})

    return {"results": results}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)