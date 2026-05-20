import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score

data = pd.read_csv("car evaluation.csv")

print(data.head())

encoder = LabelEncoder()

for col in data.columns:
    data[col] = encoder.fit_transform(data[col])

print(data.head())

data.columns = data.columns.str.strip()
X = data.drop("class", axis=1)
y = data["class"]

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

correct = 0
wrong = 0

for i in range(len(y_test)):
    if y_test.values[i] == y_pred[i]:
        correct += 1
    else:
        wrong += 1

print("Correct:", correct)
print("Wrong:", wrong)
print("Accuracy:", accuracy_score(y_test, y_pred))

knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

correct = 0
wrong = 0

for i in range(len(y_test)):
    if y_test.values[i] == y_pred[i]:
        correct += 1
    else:
        wrong += 1

print("Correct:", correct)
print("Wrong:", wrong)
print("Accuracy:", accuracy_score(y_test, y_pred) * 100, "%")

knn = KNeighborsClassifier(n_neighbors=7)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

correct = 0
wrong = 0

for i in range(len(y_test)):
    if y_test.values[i] == y_pred[i]:
        correct += 1
    else:
        wrong += 1

print("Correct:", correct)
print("Wrong:", wrong)
print("Accuracy:", accuracy_score(y_test, y_pred))
