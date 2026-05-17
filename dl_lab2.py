import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv('/content/advertising.csv')
print(data.head())

data = data.fillna(data.mean())

scaler = StandardScaler()

X = data.drop("Sales", axis=1)
y = data["Sales"]

X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Simple Linear Regression

X_simple = data[["TV"]]

X_simple_scaled = scaler.fit_transform(X_simple)

X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_simple_scaled, y, test_size=0.2, random_state=42)

model_simple = LinearRegression()
model_simple.fit(X_train_s, y_train_s)

y_pred_simple = model_simple.predict(X_test_s)

mse_s = mean_squared_error(y_test_s, y_pred_simple)
rmse_s = np.sqrt(mse_s)
r2_s = r2_score(y_test_s, y_pred_simple)

print("Simple Linear Regression:")
print("MSE:", mse_s)
print("RMSE:", rmse_s)
print("R2:", r2_s)

plt.scatter(X_test_s, y_test_s)
plt.plot(X_test_s, y_pred_simple, color='red')

plt.xlabel("TV Ad Spend (Scaled)")
plt.ylabel("Sales")
plt.title("Simple Linear Regression: TV Ad Spend vs. Sales")
plt.show()

# Multiple Linear Regression

model_multi = LinearRegression()
model_multi.fit(X_train, y_train)

y_pred_multi = model_multi.predict(X_test)

mse_m = mean_squared_error(y_test, y_pred_multi)
rmse_m = np.sqrt(mse_m)
r2_m = r2_score(y_test, y_pred_multi)

print("\nMultiple Linear Regression:")
print("MSE:", mse_m)
print("RMSE:", rmse_m)
print("R2:", r2_m)

plt.scatter(y_test, y_pred_multi)

plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Multiple Linear Regression: Actual vs. Predicted Sales")
plt.show()
