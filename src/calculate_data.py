import scipy.io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Run this script from inside the src/ folder: cd src, then python calculate_data.py
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
        capacity = data['Capacity'][0][0]
        soh = (capacity / RATED_CAPACITY) * 100

        records.append({
            'cycle_number': discharge_index,
            'capacity_Ah': capacity,
            'SoH_percent': soh
        })

df = pd.DataFrame(records)

df.to_csv('../notebooks_or_results/B0005_soh_table.csv', index=False)

print(df.head())
print("\nSaved", len(df), "rows to ../notebooks_or_results/B0005_soh_table.csv")

plt.figure(figsize=(10, 5))
plt.plot(df['cycle_number'], df['SoH_percent'], marker='o', markersize=3, linewidth=1)
plt.axhline(y=80, color='red', linestyle='--', label='80% EOL threshold')
plt.xlabel('Discharge Cycle Number')
plt.ylabel('State of Health (%)')
plt.title('Battery B0005 - SoH Degradation Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../notebooks_or_results/B0005_soh_curve.png', dpi=150)
plt.show()