import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
import json
import os
import uvicorn
import threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import paho.mqtt.client as mqtt

# -------------------------------
# 1. MODEL ARCHITECTURE
# -------------------------------
class EnergyMonitorLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=64, num_layers=2, num_classes=2):
        super(EnergyMonitorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


# -------------------------------
# 2. FASTAPI INITIALIZATION
# -------------------------------
app = FastAPI(title="Energy AI Monitor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 3. LOAD MODEL & SCALER
# -------------------------------
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
    print("✅ Model loaded")
else:
    print("❌ Model or scaler missing!")

# -------------------------------
# 4. Pydantic Schemas
# -------------------------------
class ApplianceData(BaseModel):
    appliance_id: str
    sequence: list

class BatchPredictionRequest(BaseModel):
    appliances: list[ApplianceData]


# -------------------------------
# 5. MQTT LIVE SENSOR STORAGE
# -------------------------------
latest_values = {}  # Example: {"plug1": {"voltage":230, "current":0.52}}

MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 8884
MQTT_TOPIC = "smart/plug/+/codedata"

def on_connect(client, userdata, flags, rc):
    print("🔥 MQTT Connected")
    client.subscribe(MQTT_TOPIC)

def on_message(client, userdata, msg):
    try:
        plug_id = msg.topic.split("/")[2]  # smart/plug/{plugId}/codedata
        data = json.loads(msg.payload.decode())

        latest_values[plug_id] = {
            "voltage": float(data["voltage"]),
            "current": float(data["current"])
        }

        print("📥 MQTT:", plug_id, latest_values[plug_id])

    except Exception as e:
        print("❌ MQTT Error:", e)

def start_mqtt():
    client = mqtt.Client()
    client.tls_set()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    client.loop_forever()

mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
mqtt_thread.start()


# -------------------------------
# 6. API ENDPOINTS
# -------------------------------
@app.get("/")
async def index():
    return {"status": "online", "model_loaded": model is not None}

@app.get("/sensor-data")
async def get_sensor_data():
    """
    Returns latest MQTT values:
    {
       "data": [
           { "appliance_id": "plug1", "voltage": 230, "current": 0.5 },
           ...
       ]
    }
    """
    results = []
    for plug_id, values in latest_values.items():
        results.append({
            "appliance_id": plug_id,
            "voltage": values["voltage"],
            "current": values["current"]
        })
    return {"data": results}


@app.post("/predict")
async def predict_batch(data: BatchPredictionRequest):
    if model is None or scaler is None:
        return {"error": "Model not loaded"}

    results = []

    for item in data.appliances:
        try:
            raw_data = np.array(item.sequence)

            # Fix sequence length
            if len(raw_data) < SEQ_LENGTH:
                padding = np.zeros((SEQ_LENGTH - len(raw_data), 2))
                raw_data = np.vstack([padding, raw_data])
            else:
                raw_data = raw_data[-SEQ_LENGTH:]

            scaled = scaler.transform(raw_data)
            tensor = torch.tensor([scaled], dtype=torch.float32)

            with torch.no_grad():
                output = model(tensor)
                prob = torch.softmax(output, dim=1)[0, 1].item()
                pred = torch.argmax(output, dim=1).item()

            results.append({
                "appliance_id": item.appliance_id,
                "usage_score": round(prob * 10, 2),
                "status": "Abnormal" if pred == 1 else "Normal",
                "probability": prob
            })

        except Exception as e:
            results.append({"appliance_id": item.appliance_id, "error": str(e)})

    return {"results": results}


# -------------------------------
# 7. RUN SERVER
# -------------------------------
if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)