import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris, fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error


# MULTIPLE LINEAR REGRESSION (HOUSING DATA)


housing = fetch_california_housing()
X_reg = housing.data
y_reg = housing.target

Xr_train, Xr_test, yr_train, yr_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42
)

mlr = LinearRegression()
mlr.fit(Xr_train, yr_train)
yr_pred = mlr.predict(Xr_test)

r2 = r2_score(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))


#  CLASSIFICATION MODELS (IRIS DATASET)


iris = load_iris()
X_clf = iris.data
y_clf = iris.target

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.3, random_state=42
)

# Logistic Regression
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(Xc_train, yc_train)
y_log_pred = log_reg.predict(Xc_test)
acc_log = accuracy_score(yc_test, y_log_pred)

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xc_train, yc_train)
y_knn_pred = knn.predict(Xc_test)
acc_knn = accuracy_score(yc_test, y_knn_pred)

# Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(Xc_train, yc_train)
y_dt_pred = dt.predict(Xc_test)
acc_dt = accuracy_score(yc_test, y_dt_pred)


# PERFORMANCE GRAPHS


# Regression Graph
plt.figure()
plt.bar(["R2 Score", "RMSE"], [r2, rmse])
plt.title("Multiple Linear Regression Performance")
plt.ylabel("Value")
plt.show()

# Classification Accuracy Graph
plt.figure()
plt.bar(
    ["Logistic Regression", "KNN", "Decision Tree"],
    [acc_log, acc_knn, acc_dt]
)
plt.title("IRIS Dataset - Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.show()





print("MULTIPLE LINEAR REGRESSION (California Housing)")
print("R2 Score:", r2)
print("RMSE:", rmse)

print("\nCLASSIFICATION (IRIS Dataset)")
print("Logistic Regression Accuracy:", acc_log)
print("KNN Accuracy:", acc_knn)
print("Decision Tree Accuracy:", acc_dt)
