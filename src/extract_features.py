import scipy.io
import numpy as np
import pandas as pd

# Run this script from inside the src/ folder: cd src, then python extract_features.py
file_path = "../data/B0005.mat"

mat = scipy.io.loadmat(file_path)
battery_name = list(mat.keys())[-1]
battery_data = mat[battery_name]
cycles = battery_data[0, 0]['cycle'][0]

RATED_CAPACITY = 2.0
records = []
discharge_index = 0

for c in cycles:
    if c['type'][0] == 'discharge':
        discharge_index += 1
        data = c['data'][0, 0]

        voltage = data['Voltage_measured'][0]
        current = data['Current_measured'][0]
        temperature = data['Temperature_measured'][0]
        time = data['Time'][0]
        capacity = data['Capacity'][0][0]

        soh = (capacity / RATED_CAPACITY) * 100

        discharge_time = time[-1] - time[0]
        avg_temp = np.mean(temperature)
        max_temp = np.max(temperature)

        voltage_drop = voltage[0] - voltage[-1]
        voltage_drop_rate = voltage_drop / discharge_time if discharge_time > 0 else 0

        avg_current = np.mean(np.abs(current))
        start_voltage = voltage[0]
        min_voltage = np.min(voltage)

        records.append({
            'cycle_number': discharge_index,
            'discharge_time_s': discharge_time,
            'avg_temp_C': avg_temp,
            'max_temp_C': max_temp,
            'voltage_drop_rate': voltage_drop_rate,
            'avg_current_A': avg_current,
            'start_voltage_V': start_voltage,
            'min_voltage_V': min_voltage,
            'capacity_Ah': capacity,
            'SoH_percent': soh
        })

df = pd.DataFrame(records)

df.to_csv('../notebooks_or_results/B0005_features.csv', index=False)

print(df.head())
print("\nSaved", len(df), "rows with", len(df.columns)-1, "features to ../notebooks_or_results/B0005_features.csv")
print("\nColumn summary:")
print(df.describe())