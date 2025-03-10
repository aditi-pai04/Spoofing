import time
import requests
import random
from fastapi import FastAPI
import pandas as pd
import torch
import math
from fastapi.middleware.cors import CORSMiddleware
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from utils.preprocessing import pca_transform, data_preprocessing

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in kilometers
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c  # Distance in kilometers
    return distance
# Load trained model and data preprocessing
class CNN_BiLSTM_Model(nn.Module):
    def __init__(self, input_size, cnn_out_channels=64, lstm_hidden_size=128, num_lstm_layers=2, num_classes=1):
        super(CNN_BiLSTM_Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden_size, num_lstm_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        x = self.dropout(lstm_out[:, -1, :])
        x = self.fc(x)
        return self.sigmoid(x)

# FastAPI server
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, adjust this in production to a specific list
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)
# Load training data for fitting scaler & PCA
traces = ['data/Drive_Me_Not/trace'+ str(i) + '.csv' for i in range(1, 9)]
traces_df = [data_preprocessing(trace, trace_num=i+1, method="haversine") for i, trace in enumerate(traces)]
df_processed_with_trace = pd.concat(traces_df)

# Fit StandardScaler and PCA
scaler = StandardScaler()
df_processed = df_processed_with_trace.drop(columns=['trace'])
df_processed = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns)
pca_train = pca_transform(df_processed, n_components=3)[1]  # Get fitted PCA model

# Load model
input_size = 3
model = CNN_BiLSTM_Model(input_size)
model.load_state_dict(torch.load("cnn_bilstm_model2.pth"))
model.eval()

# Initialize last GPS data for velocity and acceleration calculation
last_gps_data = {"GPS_lat": 37.7749, "GPS_long": -122.4194, "vx": 0, "vy": 0, "ax": 0, "ay": 0}

@app.get("/detect")
async def detect_spoofing():
    global last_gps_data
    gps_data = generate_gps_data(last_gps_data)  # Simulate GPS data

    # Apply preprocessing
    gps_df = pd.DataFrame([gps_data])
    gps_data_scaled = pd.DataFrame(scaler.transform(gps_df), columns=gps_df.columns)
    gps_data_pca = pd.DataFrame(pca_train.transform(gps_data_scaled), columns=['pca-one', 'pca-two', 'pca-three'])
    input_tensor = torch.tensor(gps_data_pca.values, dtype=torch.float32).unsqueeze(1)
    
    with torch.no_grad():
        prediction = model(input_tensor).item()
        print(prediction)
        result = "Real" if prediction >= 0.5 else "Spoofed"

    return {"GPS_lat": gps_data["GPS_lat"], "GPS_long": gps_data["GPS_long"], "result": result, "confidence": prediction}

def generate_gps_data(last_gps_data):
    """Simulate normal GPS data with occasional extreme spoofed jumps."""
    
    spoof_chance = random.random()

    if spoof_chance > 0.5:  # 50% chance of extreme spoofing
        gps_data = {
            "GPS_lat": last_gps_data["GPS_lat"] + random.uniform(-100, 100),  # Large jumps
            "GPS_long": last_gps_data["GPS_long"] + random.uniform(-100, 100),
        }
        spoofed = True
    else:  # Normal movement (small variations)
        gps_data = {
            "GPS_lat": last_gps_data["GPS_lat"] + random.uniform(-0.001, 0.001),
            "GPS_long": last_gps_data["GPS_long"] + random.uniform(-0.001, 0.001),
        }
        spoofed = False

    # Calculate velocity and acceleration based on Haversine formula
    distance = haversine(last_gps_data["GPS_lat"], last_gps_data["GPS_long"], gps_data["GPS_lat"], gps_data["GPS_long"])
    # Assuming 1 second between consecutive points
    vx = distance  # Velocity (in km/s)
    vy = random.uniform(-1, 1)  # Simplified, you can calculate based on more data
    
    # Calculate acceleration
    ax = (vx - last_gps_data["vx"])  # Change in velocity over time
    ay = random.uniform(-0.1, 0.1)  # Simplified, can be refined

    gps_data["vx"] = vx
    gps_data["vy"] = vy
    gps_data["ax"] = ax
    gps_data["ay"] = ay

    last_gps_data.update(gps_data)  # Update last GPS data for next iteration

    return gps_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
