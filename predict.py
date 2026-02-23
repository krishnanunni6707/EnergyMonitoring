import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os

# --- 1. SETTINGS (Must match train.py) ---
SEQ_LENGTH = 10
MODEL_PATH = "models/energy_lstm.pth"
SCALER_PATH = "models/scaler.pkl"
TEST_DATA_PATH = "data/test_data.csv" # Path to your new dataset
OUTPUT_PATH = "predicted_results.csv"

# --- 2. MODEL ARCHITECTURE (Must be identical to the one used in training) ---
class EnergyMonitorLSTM(nn.Module):
    def __init__(self):
        super(EnergyMonitorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# --- 3. PREDICTION LOGIC ---
def run_prediction():
    # Check if model exists
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print("Error: Model or Scaler not found! Please run train.py first.")
        return

    print(f"Loading data from {TEST_DATA_PATH}...")
    df = pd.read_csv(TEST_DATA_PATH).dropna()
    
    # Load Scaler and Model
    scaler = joblib.load(SCALER_PATH)
    model = EnergyMonitorLSTM()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Prepare features (Voltage and Current)
    features = df[['voltage', 'current']].values
    scaled_features = scaler.transform(features)

    # Create sequences
    X = []
    for i in range(len(scaled_features) - SEQ_LENGTH):
        X.append(scaled_features[i : i + SEQ_LENGTH])
    X = np.array(X)

    print("Analyzing patterns...")
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        outputs = model(inputs)
        _, predicted_classes = torch.max(outputs, 1)
        probabilities = torch.softmax(outputs, dim=1)[:, 1] # Probability of "Abnormal"

    # Align predictions with original data
    # (Since we lose the first 10 rows due to windowing)
    results_df = df.iloc[SEQ_LENGTH:].copy()
    results_df['anomaly_score'] = probabilities.numpy()
    results_df['status'] = ["Abnormal" if p > 0.5 else "Normal" for p in results_df['anomaly_score']]

    # --- 4. FINAL CONCLUSIONS (The "LLM" Conclusion Logic) ---
    def generate_conclusion(row):
        if row['status'] == "Abnormal":
            if row['voltage'] < 210: return "Conclusion: Abnormal - Low Voltage (Brownout Risk)"
            if row['voltage'] > 255: return "Conclusion: Abnormal - High Voltage (Overvoltage Risk)"
            if row['current'] > 1.0: return "Conclusion: Abnormal - High Current Draw (Overload)"
            return "Conclusion: Abnormal - Irregular Pattern Detected"
        return "Conclusion: Normal - System Healthy"

    results_df['final_conclusion'] = results_df.apply(generate_conclusion, axis=1)

    # Save to CSV
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"--- Analysis Complete ---")
    print(f"Results saved to: {OUTPUT_PATH}")
    
    # Print a summary to the console
    anomalies_count = len(results_df[results_df['status'] == "Abnormal"])
    print(f"Detected {anomalies_count} anomalies in the dataset.")

if __name__ == "__main__":
    run_prediction()