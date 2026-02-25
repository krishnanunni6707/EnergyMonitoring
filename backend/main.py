import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import joblib
import os

# --- 1. SETTINGS ---
SEQ_LENGTH = 10  # Look at 10 seconds of data to predict the next
BATCH_SIZE = 32
EPOCHS = 15
MODEL_PATH = "models/energy_lstm.pth"
SCALER_PATH = "models/scaler.pkl"

# --- 2. DATA PREPROCESSING ---
class EnergyDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def prepare_data(file_path, is_training=True):
    # Load and clean data
    df = pd.read_csv(file_path).dropna()
    
    # Use relevant features
    features = df[['voltage', 'current']].values
    
    if is_training:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(features)
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        
        # FIX: Changed fit_transform to fit and then predict
        iso = IsolationForest(contamination=0.05, random_state=42)
        iso.fit(scaled_data) # Train the forest
        raw_labels = iso.predict(scaled_data) # Get the labels
        
        # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
        labels = np.where(raw_labels == -1, 1, 0)
    else:
        scaler = joblib.load(SCALER_PATH)
        scaled_data = scaler.transform(features)
        labels = np.zeros(len(scaled_data)) 

    # Create Sequences (Windowing)
    X, Y = [], []
    for i in range(len(scaled_data) - SEQ_LENGTH):
        X.append(scaled_data[i : i + SEQ_LENGTH])
        if is_training:
            Y.append(labels[i + SEQ_LENGTH])
        else:
            Y.append(0) 

    return np.array(X), np.array(Y), df.iloc[SEQ_LENGTH:].reset_index(drop=True)

# --- 3. THE MODEL (LSTM) ---
class EnergyMonitorLSTM(nn.Module):
    def __init__(self):
        super(EnergyMonitorLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(64, 2) # Outputs 2 classes: Normal or Abnormal

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]) # Use the last time step

# --- 4. TRAINING FUNCTION ---
def train_model(train_path):
    print("Preparing training data and auto-labeling...")
    X, Y, _ = prepare_data(train_path, is_training=True)
    dataset = EnergyDataset(X, Y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = EnergyMonitorLSTM()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    print("Training started...")
    for epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}/{EPOCHS} complete.")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

# --- 5. PREDICTION FUNCTION ---
def predict_anomalies(test_path):
    print(f"Analyzing {test_path}...")
    X, _, original_df = prepare_data(test_path, is_training=False)
    
    model = EnergyMonitorLSTM()
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        probs = torch.softmax(outputs, dim=1)[:, 1] # Probability of being abnormal
 
    # Create Final Conclusion
    original_df['prediction_score'] = probs.numpy()
    original_df['status'] = ["Abnormal" if p > 0.5 else "Normal" for p in original_df['prediction_score']]
    
    # Logic for "Final Conclusion"
    def get_conclusion(row):
        if row['status'] == "Abnormal":
            if row['voltage'] < 200: return "Critical: Voltage Drop / Brownout"
            if row['current'] > 2.0: return "Critical: Current Surge / Overload"
            return "Warning: Pattern Anomaly Detected"
        return "System Healthy"

    original_df['final_conclusion'] = original_df.apply(get_conclusion, axis=1)
    
    output_file = "predicted_results.csv"
    original_df.to_csv(output_file, index=False)
    print(f"Done! Results saved to {output_file}")

# --- EXECUTION ---
if __name__ == "__main__":
    # 1. Train on your uploaded file
    train_model('data/train_data.csv')
    
    # 2. Predict on another dataset (make sure test_data.csv exists in /data)
    # predict_anomalies('data/test_data.csv')