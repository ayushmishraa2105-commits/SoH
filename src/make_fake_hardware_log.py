import pandas as pd
import numpy as np

# Run this script from inside the src/ folder: cd src, then python make_fake_hardware_log.py
t = np.arange(0, 600, 5)
df = pd.DataFrame({
    'time_s': t,
    'voltage_V': 4.2 - 1.1*(t/600)**1.5,
    'current_mA': 900 + np.random.randn(len(t))*10,
    'power_mW': 0,
    'temperature_C': 24 + 5*(t/600)
})
df.to_csv('../hardware_logs/esp32_log.csv', index=False)
print("Saved fake hardware log: ../hardware_logs/esp32_log.csv")
print(df.head())