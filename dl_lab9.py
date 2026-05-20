import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

import yfinance as yf

# Example: Apple stock
df = yf.download('AAPL', start='2015-01-01', end='2023-01-01')

df = df[['Close']]
df.head()

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df)

# Train-test split
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(len(data)-time_step):
        X.append(data[i:i+time_step])
        y.append(data[i+time_step])
    return np.array(X), np.array(y)

time_step = 60

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(60,1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=5, batch_size=32)

train_pred = model.predict(X_train)
test_pred = model.predict(X_test)

# inverse scaling
train_pred = scaler.inverse_transform(train_pred)
test_pred = scaler.inverse_transform(test_pred)

plt.figure(figsize=(10,5))

# original data
plt.plot(df.index, df['Close'], label='Actual')

# plotting predictions
train_plot = np.empty_like(scaled_data)
train_plot[:] = np.nan
train_plot[time_step:len(train_pred)+time_step] = train_pred

test_plot = np.empty_like(scaled_data)
test_plot[:] = np.nan
test_plot[len(train_pred)+(time_step*2):len(scaled_data)] = test_pred

plt.plot(df.index, train_plot, label='Train Prediction')
plt.plot(df.index, test_plot, label='Test Prediction')

plt.legend()
plt.show()

last_60 = scaled_data[-60:]
last_60 = last_60.reshape(1,60,1)

next_day = model.predict(last_60)
next_day_price = scaler.inverse_transform(next_day)

print("Next day predicted price:", next_day_price[0][0])

