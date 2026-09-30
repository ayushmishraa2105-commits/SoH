import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Run this script from inside the src/ folder: cd src, then python train_lstm.py
df = pd.read_csv('../notebooks_or_results/B0005_features.csv')

feature_columns = [
    'discharge_time_s', 'avg_temp_C', 'max_temp_C',
    'voltage_drop_rate', 'avg_current_A', 'start_voltage_V', 'min_voltage_V'
]
target_column = 'SoH_percent'

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

X_scaled = feature_scaler.fit_transform(df[feature_columns])
y_scaled = target_scaler.fit_transform(df[[target_column]])

WINDOW_SIZE = 10

def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, WINDOW_SIZE)
print(f"Created {len(X_seq)} sequences, each looking at {WINDOW_SIZE} past cycles")

split_point = int(len(X_seq) * 0.85)
X_train, X_test = X_seq[:split_point], X_seq[split_point:]
y_train, y_test = y_seq[:split_point], y_seq[split_point:]

print(f"Train sequences: {len(X_train)}, Test sequences: {len(X_test)}")

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32)

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

model = SoHLSTM(input_size=len(feature_columns))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

EPOCHS = 200
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    predictions = model(X_train_t)
    loss = criterion(predictions, y_train_t)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 40 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss.item():.5f}")

model.eval()
with torch.no_grad():
    test_predictions_scaled = model(X_test_t).numpy()

test_predictions = target_scaler.inverse_transform(test_predictions_scaled)
y_test_real = target_scaler.inverse_transform(y_test)

rmse = np.sqrt(mean_squared_error(y_test_real, test_predictions))
mae = mean_absolute_error(y_test_real, test_predictions)
r2 = r2_score(y_test_real, test_predictions)

print("\n--- LSTM Model Performance ---")
print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")
print(f"R2:   {r2:.3f}")

# ---- Save the trained model ----
torch.save(model.state_dict(), '../models/lstm_soh_model.pth')
print("\nSaved trained model to ../models/lstm_soh_model.pth")

test_cycle_numbers = df['cycle_number'].iloc[WINDOW_SIZE + split_point : WINDOW_SIZE + split_point + len(y_test)]

# ---- Save predictions in the format the dashboard/app expects ----
lstm_results = pd.DataFrame({
    'cycle_number': test_cycle_numbers.values,
    'actual_SoH': y_test_real.flatten(),
    'predicted_SoH': test_predictions.flatten()
})
lstm_results.to_csv('../notebooks_or_results/lstm_predictions.csv', index=False)
print("Saved ../notebooks_or_results/lstm_predictions.csv")

plt.figure(figsize=(10, 5))
plt.plot(df['cycle_number'], df['SoH_percent'], label='Full actual curve', color='gray', alpha=0.5)
plt.plot(test_cycle_numbers, y_test_real, label='Test (actual)', color='blue', marker='o', markersize=3)
plt.plot(test_cycle_numbers, test_predictions, label='Test (LSTM predicted)', color='green', marker='x', markersize=4)
plt.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% EOL threshold')
plt.xlabel('Cycle Number')
plt.ylabel('SoH (%)')
plt.title('LSTM: Actual vs Predicted SoH')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../notebooks_or_results/lstm_prediction_plot.png', dpi=150)
plt.show()