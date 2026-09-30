import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Run this script from inside the src/ folder: cd src, then python train_random_forest.py
df = pd.read_csv('../notebooks_or_results/B0005_features.csv')

split_point = int(len(df) * 0.85)

train_df = df.iloc[:split_point]
test_df = df.iloc[split_point:]

print(f"Training on cycles 1-{split_point} ({len(train_df)} rows)")
print(f"Testing on cycles {split_point+1}-{len(df)} ({len(test_df)} rows)")

feature_columns = [
    'discharge_time_s', 'avg_temp_C', 'max_temp_C',
    'voltage_drop_rate', 'avg_current_A', 'start_voltage_V', 'min_voltage_V'
]

X_train = train_df[feature_columns]
y_train = train_df['SoH_percent']

X_test = test_df[feature_columns]
y_test = test_df['SoH_percent']

model = RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, predictions))
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n--- Model Performance ---")
print(f"RMSE: {rmse:.3f}")
print(f"MAE:  {mae:.3f}")
print(f"R2:   {r2:.3f}")

importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n--- Feature Importance ---")
print(importance)

result_df = test_df[['cycle_number', 'SoH_percent']].copy()
result_df.rename(columns={'SoH_percent': 'actual_SoH'}, inplace=True)
result_df['predicted_SoH'] = predictions
result_df.to_csv('../notebooks_or_results/predictions.csv', index=False)
print("\nSaved ../notebooks_or_results/predictions.csv")

plt.figure(figsize=(10, 5))
plt.plot(train_df['cycle_number'], train_df['SoH_percent'], label='Train (actual)', color='gray')
plt.plot(test_df['cycle_number'], y_test, label='Test (actual)', color='blue', marker='o', markersize=3)
plt.plot(test_df['cycle_number'], predictions, label='Test (predicted)', color='orange', marker='x', markersize=4)
plt.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='80% EOL threshold')
plt.xlabel('Cycle Number')
plt.ylabel('SoH (%)')
plt.title('Random Forest: Actual vs Predicted SoH')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('../notebooks_or_results/rf_prediction_plot.png', dpi=150)
plt.show()