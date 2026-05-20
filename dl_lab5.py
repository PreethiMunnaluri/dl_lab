# REGRESSION

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor

data = pd.read_csv("/content/advertising.csv")
print(data.head())

X = data[['TV', 'Radio', 'Newspaper']]
y = data['Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt = DecisionTreeRegressor(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_pred)

print("\nDecision Tree MSE:", dt_mse)

path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

dt_pruned = DecisionTreeRegressor(
    ccp_alpha=ccp_alphas[5],
    random_state=42
)

dt_pruned.fit(X_train, y_train)
pruned_pred = dt_pruned.predict(X_test)

pruned_mse = mean_squared_error(y_test, pruned_pred)
print("Pruned Decision Tree MSE:", pruned_mse)

rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

rf_mse = mean_squared_error(y_test, rf_pred)
print("Random Forest MSE:", rf_mse)

ada = AdaBoostRegressor(
    estimator=DecisionTreeRegressor(max_depth=3),
    n_estimators=100,
    random_state=42
)

ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)

ada_mse = mean_squared_error(y_test, ada_pred)
print("AdaBoost MSE:", ada_mse)

print("\nRMSE Values:")
print("Decision Tree RMSE:", np.sqrt(dt_mse))
print("Pruned Tree RMSE:", np.sqrt(pruned_mse))
print("Random Forest RMSE:", np.sqrt(rf_mse))
print("AdaBoost RMSE:", np.sqrt(ada_mse))

# CLASSIFICATION

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

data = pd.read_csv("/content/Titanic-Dataset.csv")

# Show first few rows
print(data.head())

df = data[['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]

# Fill missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Encode categorical
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

train_pred = dt.predict(X_train)
test_pred = dt.predict(X_test)

print("Decision Tree Training Accuracy:", accuracy_score(y_train, train_pred))
print("Decision Tree Test Accuracy:", accuracy_score(y_test, test_pred))

path = dt.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

dt_pruned = DecisionTreeClassifier(ccp_alpha=ccp_alphas[10], random_state=42)
dt_pruned.fit(X_train, y_train)

pruned_pred = dt_pruned.predict(X_test)
print("Pruned Tree Accuracy:", accuracy_score(y_test, pruned_pred))

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
print("Random Forest Accuracy:", accuracy_score(y_test, rf_pred))

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    random_state=42
)

ada.fit(X_train, y_train)
ada_pred = ada.predict(X_test)

print("AdaBoost Accuracy:", accuracy_score(y_test, ada_pred))

