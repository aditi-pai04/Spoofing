import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from utils.preprocessing import pca_transform, data_preprocessing
import numpy as np
from sklearn.metrics import roc_curve, auc

# Device Selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
X_train_tensor = torch.tensor(df_train[['pca-one', 'pca-two', 'pca-three']].values, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor((trace_num % 2).astype(float).values, dtype=torch.float32).to(device)

# Dataset and DataLoader with WeightedRandomSampler
dataset = TensorDataset(X_train_tensor.unsqueeze(1), y_train_tensor)

class_counts = y_train_tensor.long().bincount()
weights = 1.0 / class_counts.float()
sample_weights = weights[y_train_tensor.long()]
sampler = torch.utils.data.WeightedRandomSampler(sample_weights, len(sample_weights))

train_loader = DataLoader(dataset, batch_size=64, sampler=sampler)

# Define Hybrid CNN-BiLSTM Model (Improved)
class CNN_BiLSTM_Model(nn.Module):
    def __init__(self, input_size, cnn_out_channels=64, lstm_hidden_size=128, num_lstm_layers=2, num_classes=1, dropout_rate=0.5): # Reduced LSTM hidden size, added dropout rate
        super(CNN_BiLSTM_Model, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1)  # Smaller kernel
        self.bn1 = nn.BatchNorm1d(cnn_out_channels)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden_size, num_lstm_layers, batch_first=True, bidirectional=True, dropout=dropout_rate) # Dropout in LSTM
        self.dropout_lstm = nn.Dropout(dropout_rate) # Dropout after LSTM

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



class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.0): 
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = nn.BCELoss()(inputs, targets)
        pt = torch.exp(-BCE_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss

# Initialize CNN-BiLSTM Model
input_size = 3
model = CNN_BiLSTM_Model(input_size, cnn_out_channels=64, lstm_hidden_size=128, dropout_rate=0.2).to(device)

# Define loss function and optimizer
criterion = FocalLoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5) # AdamW optimizer
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)  # CosineAnnealingLR with eta_min


# Training loop
num_epochs = 100
best_auc = 0.0  # Track best AUC
patience = 10
epochs_no_improve = 0
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in train_loader:
        X_batch, y_batch = batch
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    scheduler.step() # Step scheduler every epoch

    TRACE = 1
    # Validation on spoofed data (for early stopping based on AUC)
    spoofed_data = data_preprocessing('data/Drive_Me_Not/spoofed/spoofed_trace'+str(TRACE)+'.csv',
                                        selected_attributes=['GPS_lat', 'GPS_long', 'Time', 'vx', 'vy', 'ax', 'ay', 'spoofed'],
                                        trace_num=TRACE)
    spoofed_for_test = pd.DataFrame(scaler.transform(spoofed_data[['GPS_lat', 'GPS_long', 'vx', 'vy', 'ax', 'ay']]), columns=['GPS_lat', 'GPS_long', 'vx', 'vy', 'ax', 'ay'])

    spoofed_for_test = pd.DataFrame(pca_train.transform(spoofed_for_test), columns=['pca-one', 'pca-two', 'pca-three'])

    labels = torch.tensor(spoofed_data['spoofed'].apply(lambda x: 1 if x == 0 else 0).values, dtype=torch.float32).to(device) # Moved to device
    spoofed_tensor = torch.tensor(spoofed_for_test.values, dtype=torch.float32).unsqueeze(1).to(device) # Moved to device
    model.eval()
    with torch.no_grad():
        predictions = model(spoofed_tensor).squeeze()
        fpr, tpr, _ = roc_curve(labels.numpy(), predictions.numpy())
        current_auc = auc(fpr, tpr)

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}, AUC: {current_auc:.4f}")

        if current_auc > best_auc:
            best_auc = current_auc
            torch.save(model.state_dict(), 'cnn_bilstm_model_best2.pth') # Save the best model based on AUC
            print(f"Saving Best Model (AUC = {current_auc:.4f})")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                torch.save(model.state_dict(), 'cnn_bilstm_model_last.pth')
                print(f"Saving Last Model (AUC = {current_auc:.4f})")
                print(f"Early stopping at epoch {epoch+1} (No improvement in AUC for {patience} epochs)")
                
                break  # Stop training


            

# Load the best model
model.load_state_dict(torch.load('cnn_bilstm_model_best.pth'))


# Evaluate on spoofed data (using the best model)
model.eval()
with torch.no_grad():
    predictions = model(spoofed_tensor).squeeze()
    fpr, tpr, thresholds = roc_curve(labels.numpy(), predictions.numpy())
    optimal_threshold = thresholds[np.argmax(tpr - fpr)]  # Find optimal threshold based on Youden's J statistic
    predicted_labels = (predictions >= optimal_threshold).float()
    accuracy = (predicted_labels == labels).sum().item() / len(labels)
    print(f"CNN-BiLSTM Model Accuracy: {accuracy * 100:.2f}%")


# import torch
# import torch.nn as nn
# import torch.optim as optim
# import pandas as pd
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# import numpy as np
# from sklearn.metrics import roc_curve, auc
# from pytorch_ranger import Ranger  # Better optimizer
# import warnings
# from utils.preprocessing import pca_transform, data_preprocessing

# warnings.filterwarnings("ignore", category=UserWarning)

# # Device Selection
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

# # Data Loading & Preprocessing
# traces = ['data/drive-me-not/trace'+ str(i) + '.csv' for i in range(1, 9)]
# trace_data = [data_preprocessing(trace, trace_num=i+1, method="haversine") for i, trace in enumerate(traces)]
# df_processed = pd.concat(trace_data)

# # Feature Scaling
# scaler = StandardScaler()
# df_scaled = pd.DataFrame(scaler.fit_transform(df_processed.drop(columns=['trace'])), columns=df_processed.columns.drop(['trace']))
# trace_num = df_processed['trace'].astype("int").reset_index(drop=True)

# # PCA Transformation
# from utils.preprocessing import pca_transform
# N_COMPONENTS = 3
# df_train, pca_train = pca_transform(df_scaled, n_components=N_COMPONENTS)
# df_train['trace'] = trace_num

# # Convert to Torch Tensors
# X_train_tensor = torch.tensor(df_train[['pca-one', 'pca-two', 'pca-three']].values, dtype=torch.float32).to(device)
# y_train_tensor = torch.tensor((trace_num % 2).astype(float).values, dtype=torch.float32).to(device)

# # Fix Channel Issue: Reshape Tensor
# X_train_tensor = X_train_tensor.unsqueeze(1)  # Shape: (batch_size, channels=1, features)

# # Dataset & DataLoader
# dataset = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(dataset, batch_size=64, shuffle=True)

# # Define CNN-BiLSTM Model
# class CNN_BiLSTM(nn.Module):
#     def __init__(self, input_size=3, cnn_out_channels=32, lstm_hidden_size=64, num_lstm_layers=2, dropout_rate=0.3):
#         super(CNN_BiLSTM, self).__init__()
        
#         # Convolutional Layers
#         self.conv1 = nn.Conv1d(in_channels=1, out_channels=cnn_out_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv1d(in_channels=cnn_out_channels, out_channels=cnn_out_channels, kernel_size=3, padding=1)
        
#         self.ln1 = nn.LayerNorm([cnn_out_channels,3])  # More stable than BatchNorm
#         self.dropout1 = nn.Dropout(dropout_rate)

#         # BiLSTM
#         self.lstm = nn.LSTM(cnn_out_channels, lstm_hidden_size, num_lstm_layers, batch_first=True, bidirectional=True, dropout=dropout_rate)
#         self.dropout_lstm = nn.Dropout(dropout_rate)

#         # Fully Connected Layer
#         self.fc = nn.Linear(lstm_hidden_size * 2, 1)
        
#         # Activation Functions
#         self.relu = nn.ReLU()
#         self.sigmoid = nn.Sigmoid()
    
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.ln1(self.conv2(x))
#         x = self.dropout1(x)

#         x = x.permute(0, 2, 1)  # Reshape for LSTM
#         lstm_out, _ = self.lstm(x)
#         x = self.dropout_lstm(lstm_out[:, -1, :])
#         x = self.fc(x)
#         return self.sigmoid(x)

# # Improved Focal Loss
# class FocalLoss(nn.Module):
#     def __init__(self, alpha=0.5, gamma=2.0):  # Increased gamma for better handling of class imbalance
#         super(FocalLoss, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma

#     def forward(self, inputs, targets):
#         bce_loss = nn.BCELoss()(inputs, targets)
#         pt = torch.exp(-bce_loss)
#         focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
#         return focal_loss

# # Initialize Model
# model = CNN_BiLSTM().to(device)

# # Use Ranger Optimizer for Stability
# optimizer = Ranger(model.parameters(), lr=0.001, weight_decay=1e-5)

# # Learning Rate Scheduler
# scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, steps_per_epoch=len(train_loader), epochs=50)

# # Training Loop with Improved Stopping
# num_epochs = 100
# best_auc = 0.0
# patience = 10
# epochs_no_improve = 0
# criterion = FocalLoss()

# for epoch in range(num_epochs):
#     model.train()
#     total_loss = 0
    
#     for batch in train_loader:
#         X_batch, y_batch = batch
#         optimizer.zero_grad()
#         outputs = model(X_batch)
#         loss = criterion(outputs.squeeze(), y_batch)
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
    
#     scheduler.step()  # Update LR

#     # Early Stopping: Validate with Spoofed Data
#     spoofed_data = data_preprocessing('data/drive-me-not/spoofed/spoofed_trace1.csv', trace_num=1)
#     #spoofed_data = spoofed_data.drop(columns=['trace'])
#     spoofed_features = scaler.transform(spoofed_data[['GPS_lat', 'GPS_long', 'vx', 'vy', 'ax', 'ay']])
#     spoofed_pca = pca_train.transform(spoofed_features)
#     labels = torch.tensor(spoofed_data['spoofed'].apply(lambda x: 1 if x == 0 else 0).values, dtype=torch.float32).to(device)
#     spoofed_tensor = torch.tensor(spoofed_pca, dtype=torch.float32).unsqueeze(1).to(device)

#     # Evaluate Model
#     model.eval()
#     with torch.no_grad():
#         predictions = model(spoofed_tensor).squeeze()
#         fpr, tpr, _ = roc_curve(labels.cpu().numpy(), predictions.cpu().numpy())
#         current_auc = auc(fpr, tpr)

#         print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}, AUC: {current_auc:.4f}")

#         if current_auc > best_auc:
#             best_auc = current_auc
#             torch.save(model.state_dict(), 'cnn_bilstm_model_best.pth')
#             print(f"New Best Model (AUC = {current_auc:.4f})")
#             epochs_no_improve = 0
#         else:
#             epochs_no_improve += 1
#             if epochs_no_improve >= patience:
#                 torch.save(model.state_dict(), 'cnn_bilstm_model_last.pth')
#                 print(f"Early stopping at epoch {epoch+1} (No improvement in AUC for {patience} epochs)")
#                 break  

# # Load Best Model & Evaluate
# model.load_state_dict(torch.load('cnn_bilstm_model_best.pth'))
# model.eval()
# with torch.no_grad():
#     predictions = model(spoofed_tensor).squeeze()
#     fpr, tpr, thresholds = roc_curve(labels.cpu().numpy(), predictions.cpu().numpy())
#     optimal_threshold = thresholds[np.argmax(tpr - fpr)]
#     predicted_labels = (predictions >= optimal_threshold).float()
#     accuracy = (predicted_labels == labels).sum().item() / len(labels)
#     print(f"Final Model Accuracy: {accuracy * 100:.2f}%")
