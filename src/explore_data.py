import scipy.io
import numpy as np

# Run this script from inside the src/ folder: cd src, then python explore_data.py
file_path = "../data/B0005.mat"

mat = scipy.io.loadmat(file_path)
battery_name = list(mat.keys())[-1]
print("Top-level keys found:", mat.keys())
print("Using battery data under key:", battery_name)

battery_data = mat[battery_name]
cycles = battery_data[0, 0]['cycle'][0]

print("\nTotal number of cycles (charge+discharge+impedance combined):", len(cycles))

first_cycle = cycles[0]
print("\nFirst cycle type:", first_cycle['type'][0])
print("First cycle fields available:", first_cycle.dtype.names)

discharge_count = sum(1 for c in cycles if c['type'][0] == 'discharge')
print("\nNumber of DISCHARGE cycles found:", discharge_count)

for c in cycles:
    if c['type'][0] == 'discharge':
        data = c['data'][0, 0]
        print("\nFields inside a discharge cycle's data:", data.dtype.names)
        print("Capacity value in this cycle:", data['Capacity'][0][0])
        break