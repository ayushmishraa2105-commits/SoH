import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

st.set_page_config(page_title="Cell Health Monitor", page_icon="🔋", layout="wide")

# ---------------- Model definition (must match training) ----------------
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

FEATURE_COLUMNS = [
    'discharge_time_s', 'avg_temp_C', 'max_temp_C',
    'voltage_drop_rate', 'avg_current_A', 'start_voltage_V', 'min_voltage_V'
]
WINDOW_SIZE = 10

@st.cache_resource
def load_model_and_scalers():
    train_df = pd.read_csv('notebooks_or_results/B0005_features.csv')
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    feature_scaler.fit(train_df[FEATURE_COLUMNS])
    target_scaler.fit(train_df[['SoH_percent']])

    model = SoHLSTM(input_size=len(FEATURE_COLUMNS))
    model.load_state_dict(torch.load('models/lstm_soh_model.pth'))
    model.eval()
    return model, feature_scaler, target_scaler, train_df

model, feature_scaler, target_scaler, train_df = load_model_and_scalers()

# ---------------- Header ----------------
st.title("🔋 Cell Health Monitor")
st.caption("AI-based EV battery State of Health estimation — trained on NASA B0005 discharge cycles")

# ---------------- Sidebar: upload ----------------
st.sidebar.header("Upload hardware log")
uploaded = st.sidebar.file_uploader(
    "ESP32 CSV (time_s, voltage_V, current_mA, temperature_C)", type="csv"
)
st.sidebar.caption("No file? The app shows model benchmark results using existing project data below.")

# ---------------- Top metrics row ----------------
col1, col2, col3, col4 = st.columns(4)
col1.metric("Dataset", "NASA B0005")
col2.metric("Discharge cycles used", len(train_df))
col3.metric("Model", "LSTM (window=10)")
col4.metric("Best test RMSE", "0.87 %SoH")

st.divider()

# ---------------- If a file is uploaded: run the full pipeline live ----------------
if uploaded is not None:
    raw = pd.read_csv(uploaded)
    st.subheader("Uploaded hardware log")
    st.line_chart(raw.set_index('time_s')[['voltage_V']])

    # ---- Convert to features (same logic as hardware_to_features.py) ----
    discharge_time = raw['time_s'].iloc[-1] - raw['time_s'].iloc[0]
    avg_temp = raw['temperature_C'].mean()
    max_temp = raw['temperature_C'].max()
    voltage_drop = raw['voltage_V'].iloc[0] - raw['voltage_V'].iloc[-1]
    voltage_drop_rate = voltage_drop / discharge_time if discharge_time > 0 else 0
    avg_current_A = raw['current_mA'].abs().mean() / 1000
    start_voltage = raw['voltage_V'].iloc[0]
    min_voltage = raw['voltage_V'].min()

    features = pd.DataFrame([{
        'discharge_time_s': discharge_time, 'avg_temp_C': avg_temp, 'max_temp_C': max_temp,
        'voltage_drop_rate': voltage_drop_rate, 'avg_current_A': avg_current_A,
        'start_voltage_V': start_voltage, 'min_voltage_V': min_voltage
    }])

    # ---- Predict live, no copy-pasting needed ----
    scaled = feature_scaler.transform(features[FEATURE_COLUMNS])
    sequence = np.tile(scaled, (WINDOW_SIZE, 1)).reshape(1, WINDOW_SIZE, len(FEATURE_COLUMNS))
    with torch.no_grad():
        pred_scaled = model(torch.tensor(sequence, dtype=torch.float32)).numpy()
    predicted_soh = target_scaler.inverse_transform(pred_scaled)[0][0]

    st.subheader("Prediction")
    st.metric("Predicted SoH", f"{predicted_soh:.2f}%",
               delta="Healthy" if predicted_soh > 85 else "Near end-of-life" if predicted_soh > 70 else "Below 80% threshold")
    st.dataframe(features)

st.divider()

# ---------------- Model benchmark (existing real results) ----------------
st.subheader("Model benchmark — NASA B0005")
benchmark = pd.DataFrame({
    'Model': ['Random Forest (85/15 split)', 'LSTM (sliding window)'],
    'RMSE': [1.80, 0.87],
    'MAE': [1.63, 0.78],
    'R2': [-3.59, -0.12],
})
st.dataframe(benchmark, width = 'stretch')

st.subheader("SoH degradation across cycles")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(train_df['cycle_number'], train_df['SoH_percent'], color='#33D6A6')
ax.axhline(80, color='red', linestyle='--', alpha=0.6, label='80% EOL threshold')
ax.set_xlabel('Cycle number')
ax.set_ylabel('SoH (%)')
ax.legend()
st.pyplot(fig)