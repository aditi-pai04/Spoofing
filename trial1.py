import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from utils.preprocessing import pca_transform, data_preprocessing
import sys
sys.path.append("..")

# Load and preprocess data
traces = ['data/Drive_Me_Not/trace'+ str(i) + '.csv' for i in range(1, 9)]
traces_df = [data_preprocessing(trace, trace_num=i+1, method="haversine") for i, trace in enumerate(traces)]
df_processed_with_trace = pd.concat(traces_df)
trace_num = df_processed_with_trace['trace'].astype("int")
df_processed = df_processed_with_trace.drop(columns=['trace'])

# Standard scaling
scaler = StandardScaler()
df_processed = pd.DataFrame(scaler.fit_transform(df_processed), columns=df_processed.columns)
trace_num.reset_index(drop=True, inplace=True)

# PCA transformation
N_COMPONENTS = 3
df_train, pca_train = pca_transform(df_processed, n_components=N_COMPONENTS)
df_train['trace'] = trace_num

# Convert PCA-transformed training data to tensors
X_train_tensor = torch.tensor(df_train[['pca-one', 'pca-two', 'pca-three']].values, dtype=torch.float32)
y_train_tensor = torch.tensor([1] * len(df_train), dtype=torch.float32)  # All training data is benign

dataset = TensorDataset(X_train_tensor.unsqueeze(1), y_train_tensor)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define LSTM model
class LSTMAnomalyDetector(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMAnomalyDetector, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return torch.sigmoid(out)

# Initialize model
input_size = 3
hidden_size = 64
num_layers = 2
model = LSTMAnomalyDetector(input_size, hidden_size, num_layers)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        X_batch, y_batch = batch
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")

# Evaluate on spoofed data
TRACE = 1
spoofed_data = data_preprocessing('data/Drive_Me_Not/spoofed/spoofed_trace'+str(TRACE)+'.csv',
                                    selected_attributes=['GPS_lat', 'GPS_long', 'Time', 'vx', 'vy', 'ax', 'ay', 'spoofed'],
                                    trace_num=TRACE)
spoofed_for_test = pd.DataFrame(scaler.transform(spoofed_data[['GPS_lat', 'GPS_long', 'vx', 'vy', 'ax', 'ay']]), columns=['GPS_lat', 'GPS_long', 'vx', 'vy', 'ax', 'ay'])
spoofed_for_test = pd.DataFrame(pca_train.transform(spoofed_for_test), columns=['pca-one', 'pca-two', 'pca-three'])
labels = torch.tensor(spoofed_data['spoofed'].apply(lambda x: 1 if x == 0 else 0).values, dtype=torch.float32)
spoofed_tensor = torch.tensor(spoofed_for_test.values, dtype=torch.float32).unsqueeze(1)

model.eval()
with torch.no_grad():
    predictions = model(spoofed_tensor).squeeze()
    predicted_labels = (predictions >= 0.5).float()
    accuracy = (predicted_labels == labels).sum().item() / len(labels)

print(f"LSTM Model Accuracy: {accuracy * 100:.2f}%")
