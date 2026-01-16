import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load data
file_path = 'data/household_power_consumption.txt'

# Data has ; separator, missing values are '?'
df = pd.read_csv(file_path, sep=';', low_memory=False, na_values='?')

# Combine Date and Time columns and convert to datetime
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], dayfirst=True)

# Sort by datetime
df = df.sort_values('Datetime')

# Use only relevant column: Global_active_power (energy consumption in kilowatts)
data = df[['Datetime', 'Global_active_power']].copy()

# Drop rows with missing values
data = data.dropna()

# Reset index
data = data.reset_index(drop=True)

# Resample to hourly average (to reduce noise and size)
data = data.set_index('Datetime').resample('H').mean().reset_index()

# Scale data between 0 and 1
scaler = MinMaxScaler()
data['Global_active_power_scaled'] = scaler.fit_transform(data[['Global_active_power']])

# Create sequences for LSTM (past 24 hours to predict next hour)
SEQ_LEN = 24

def create_sequences(data_series, seq_length=SEQ_LEN):
    X, y = [], []
    for i in range(len(data_series) - seq_length):
        X.append(data_series[i:i+seq_length])
        y.append(data_series[i+seq_length])
    return np.array(X), np.array(y)

X, y = create_sequences(data['Global_active_power_scaled'].values)

# Split into train and test (80/20 split)
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Reshape for LSTM: (samples, time_steps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Build LSTM model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LEN, 1)),
    Dropout(0.2),
    LSTM(25),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_split=0.1,
    verbose=1
)

# Predict on test data
y_pred = model.predict(X_test)

# Inverse transform to get real energy consumption values
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))
print(f'Test RMSE: {rmse:.3f} kW')

# Plot actual vs predicted energy consumption
plt.figure(figsize=(12,6))
plt.plot(y_test_inv[:200], label='Actual')
plt.plot(y_pred_inv[:200], label='Predicted')
plt.title('Energy Consumption Forecasting (UCI Dataset)')
plt.xlabel('Time Steps (hours)')
plt.ylabel('Global Active Power (kW)')
plt.legend()
plt.show()
