import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import io
import os
import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import asyncio
from fastapi.responses import StreamingResponse
import json
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

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. ROUTES (Order matters: Define app first, then routes) ---

@app.get("/")
async def read_index():
    # This serves your index.html when you visit http://127.0.0.1:8000/
    if os.path.exists("index.html"):
        return FileResponse('index.html')
    return {"error": "index.html not found in project folder"}

# --- 4. LOAD SAVED ASSETS ---
SEQ_LENGTH = 10
MODEL_PATH = "models/energy_lstm.pth"
SCALER_PATH = "models/scaler.pkl"

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)
    model = EnergyMonitorLSTM()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
    model.eval()
    print("✅ AI Model and Scaler loaded successfully.")
else:
    print("❌ ERROR: Ensure 'models/' folder contains your .pth and .pkl files!")

# --- 5. PREDICTION ENDPOINT ---
import asyncio
from fastapi.responses import StreamingResponse
import json

@app.post("/predict_stream")
async def predict_energy_stream(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.BytesIO(contents))
    df.columns = df.columns.str.strip().str.lower()
    df = df.dropna(subset=['voltage', 'current'])

    async def event_generator():
        features = df[['voltage', 'current']].values
        scaled_features = scaler.transform(features)

        for i in range(len(scaled_features) - SEQ_LENGTH + 1):
            window = scaled_features[i : i + SEQ_LENGTH]
            input_tensor = torch.tensor([window], dtype=torch.float32)
            
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.softmax(output, dim=1)[0, 1].item()
            
            status = "Abnormal" if prob > 0.5 else "Normal"
            v_now = float(df['voltage'].iloc[i + SEQ_LENGTH - 1])
            c_now = float(df['current'].iloc[i + SEQ_LENGTH - 1])

            result = {
                "row": i + 1,
                "voltage": round(v_now, 2),
                "current": round(c_now, 3),
                "status": status
            }
            # Send the result as a JSON string followed by a separator
            yield json.dumps(result) + "\n"
            
            # Artificial micro-delay so the user can actually see the "stream" effect
            await asyncio.sleep(0.01) 

    return StreamingResponse(event_generator(), media_type="text/event-stream")
# --- 6. RUN SERVER ---
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)