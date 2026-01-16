# Energy Consumption Forecasting using LSTM
# Author: Madeeha Rehan
# Description: Predicts future energy consumption using time-series data

# Step 1: Import libraries.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 2: Load and preprocess data
df = pd.read_csv('household_power_consumption.txt', sep=';',
                 parse_dates={'datetime':[0,1]}, infer_datetime_format=True,
                 low_memory=False, na_values=['?'])

# Fill missing data
df.fillna(method='ffill', inplace=True)

# Select the target variable and set datetime as index
data = df[['datetime', 'Global_active_power']].set_index('datetime')

# Plot the original data
plt.figure(figsize=(12,4))
plt.plot(data['Global_active_power'])
plt.title('Global Active Power Over Time')
plt.show()

# Step 3: Normalize data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Step 4: Create sequences
def create_sequences(data, seq_length=24):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

SEQ_LENGTH = 24
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Reshape X for LSTM input: (samples, timesteps, features)
X = X.reshape((X.shape[0], X.shape[1], 1))

print(f'Shape of X: {X.shape}, Shape of y: {y.shape}')

# Step 5: Train-test split (80-20)
split_idx = int(0.8 * len(X))
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

# Step 6: Build LSTM model
model = Sequential([
    LSTM(100, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.summary()

# Step 7: Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32,
                    validation_data=(X_test, y_test))

# Step 8: Predict and inverse scale
y_pred = model.predict(X_test)

y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1))
y_pred_inv = scaler.inverse_transform(y_pred)

# Step 9: Plot results
plt.figure(figsize=(12,4))
plt.plot(y_test_inv, label='Actual')
plt.plot(y_pred_inv, label='Predicted')
plt.title('Energy Consumption Forecast (LSTM)')
plt.xlabel('Time Steps')
plt.ylabel('Global Active Power')
plt.legend()
plt.show()
