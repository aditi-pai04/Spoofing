import time
import random
import pandas as pd
import torch
import math
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from utils.preprocessing import pca_transform, data_preprocessing

# Haversine function
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

# CNN-BiLSTM model
class CNN_BiLSTM_Model(nn.Module):
    def __init__(self, input_size, cnn_out_channels=64, lstm_hidden_size=128, num_lstm_layers=2, num_classes=1, dropout_rate=0.2):
        super(CNN_BiLSTM_Model, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_out_channels)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden_size, num_lstm_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.dropout_lstm = nn.Dropout(dropout_rate)

        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)

        x = self.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        x = x.permute(0, 2, 1)  # Reshape for LSTM
        lstm_out, _ = self.lstm(x)
        x = self.dropout_lstm(lstm_out[:, -1, :])
        x = self.fc(x)
        return self.sigmoid(x)

# FastAPI app setup
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],
)

# Load and preprocess training data
traces = ['data/Drive_Me_Not/trace'+ str(i) + '.csv' for i in range(1, 9)]
traces_df = [data_preprocessing(trace, trace_num=i+1, method="haversine") for i, trace in enumerate(traces)]
df_processed_with_trace = pd.concat(traces_df)

# Apply StandardScaler and PCA
scaler = StandardScaler()
df_processed = df_processed_with_trace.drop(columns=['trace'])
df_processed = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns)
pca_train = pca_transform(df_processed, n_components=3)[1]

# Load the trained model
input_size = 3
model = CNN_BiLSTM_Model(input_size)
model.load_state_dict(torch.load("cnn_bilstm_model_best.pth"))
model.eval()

# Initialize GPS data for the first time
last_gps_data = {"GPS_lat": 37.7749, "GPS_long": -122.4194, "vx": 0, "vy": 0, "ax": 0, "ay": 0}

@app.get("/detect")
async def detect_spoofing():
    global last_gps_data
    gps_data = generate_gps_data(last_gps_data)  # Simulate GPS data

    # Preprocess the data
    gps_df = pd.DataFrame([gps_data])
    gps_data_scaled = pd.DataFrame(scaler.transform(gps_df), columns=gps_df.columns)
    gps_data_pca = pd.DataFrame(pca_train.transform(gps_data_scaled), columns=['pca-one', 'pca-two', 'pca-three'])
    input_tensor = torch.tensor(gps_data_pca.values, dtype=torch.float32).unsqueeze(1)
    
    # Prediction
    with torch.no_grad():
        prediction = model(input_tensor).squeeze()
        result = "Real" if prediction >= 0.5 else "Spoofed"
        confidence = prediction.item()

    return {"GPS_lat": gps_data["GPS_lat"], "GPS_long": gps_data["GPS_long"], "result": result, "confidence": confidence}

def generate_gps_data(last_gps_data):
    """Simulate normal GPS data with occasional extreme spoofed jumps."""
    spoof_chance = random.random()

    def clamp(value, min_value, max_value):
        """Clamp the value between min_value and max_value."""
        return max(min_value, min(max_value, value))

    if spoof_chance > 0.5:  # 50% chance of extreme spoofing
        gps_data = {
            "GPS_lat": clamp(last_gps_data["GPS_lat"] + random.uniform(-100, 100), -90, 90),
            "GPS_long": clamp(last_gps_data["GPS_long"] + random.uniform(-100, 100), -180, 180),
        }
        spoofed = True
    else:  # Normal movement (small variations)
        gps_data = {
            "GPS_lat": clamp(last_gps_data["GPS_lat"] + random.uniform(-0.001, 0.001), -90, 90),
            "GPS_long": clamp(last_gps_data["GPS_long"] + random.uniform(-0.001, 0.001), -180, 180),
        }
        spoofed = False

    # Calculate velocity and acceleration
    distance = haversine(last_gps_data["GPS_lat"], last_gps_data["GPS_long"], gps_data["GPS_lat"], gps_data["GPS_long"])
    vx = distance  # Simplified velocity (in km/s)
    vy = random.uniform(-1, 1)  # Simplified velocity calculation
    
    # Calculate acceleration
    ax = (vx - last_gps_data["vx"])  # Change in velocity over time
    ay = random.uniform(-0.1, 0.1)  # Simplified acceleration

    gps_data["vx"] = vx
    gps_data["vy"] = vy
    gps_data["ax"] = ax
    gps_data["ay"] = ay

    last_gps_data.update(gps_data)  # Update last GPS data for next iteration

    return gps_data

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
