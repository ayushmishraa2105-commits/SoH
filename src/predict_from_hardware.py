import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# Run this script from inside the src/ folder: cd src, then python predict_from_hardware.py

class SoHLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=32):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

feature_columns = [
    'discharge_time_s', 'avg_temp_C', 'max_temp_C',
    'voltage_drop_rate', 'avg_current_A', 'start_voltage_V', 'min_voltage_V'
]

train_df = pd.read_csv('../notebooks_or_results/B0005_features.csv')
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
feature_scaler.fit(train_df[feature_columns])
target_scaler.fit(train_df[['SoH_percent']])

new_data = pd.read_csv('../hardware_logs/hardware_features.csv')
new_scaled = feature_scaler.transform(new_data[feature_columns])

WINDOW_SIZE = 10
sequence = np.tile(new_scaled, (WINDOW_SIZE, 1))
sequence = sequence.reshape(1, WINDOW_SIZE, len(feature_columns))
sequence_t = torch.tensor(sequence, dtype=torch.float32)

model = SoHLSTM(input_size=len(feature_columns))
model.load_state_dict(torch.load('../models/lstm_soh_model.pth'))
model.eval()

with torch.no_grad():
    prediction_scaled = model(sequence_t).numpy()

prediction = target_scaler.inverse_transform(prediction_scaled)
print(f"Predicted SoH from hardware data: {prediction[0][0]:.2f}%")