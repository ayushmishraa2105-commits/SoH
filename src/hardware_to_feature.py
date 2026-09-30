import pandas as pd
import numpy as np

# Run this script from inside the src/ folder: cd src, then python hardware_to_feature.py
raw = pd.read_csv('../hardware_logs/esp32_log.csv')

discharge_time = raw['time_s'].iloc[-1] - raw['time_s'].iloc[0]
avg_temp = raw['temperature_C'].mean()
max_temp = raw['temperature_C'].max()

voltage_drop = raw['voltage_V'].iloc[0] - raw['voltage_V'].iloc[-1]
voltage_drop_rate = voltage_drop / discharge_time if discharge_time > 0 else 0

avg_current_A = raw['current_mA'].abs().mean() / 1000
start_voltage = raw['voltage_V'].iloc[0]
min_voltage = raw['voltage_V'].min()

features = pd.DataFrame([{
    'discharge_time_s': discharge_time,
    'avg_temp_C': avg_temp,
    'max_temp_C': max_temp,
    'voltage_drop_rate': voltage_drop_rate,
    'avg_current_A': avg_current_A,
    'start_voltage_V': start_voltage,
    'min_voltage_V': min_voltage
}])

features.to_csv('../hardware_logs/hardware_features.csv', index=False)
print("Converted ESP32 log into model-ready features:")
print(features)
print("\nSaved to ../hardware_logs/hardware_features.csv")