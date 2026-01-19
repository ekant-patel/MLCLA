import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

# -----------------------------
# KNN Class Implementation
# -----------------------------
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):
        predictions = []
        for x in X:
            distances = []
            for x_train in self.X_train:
                distance = self.euclidean_distance(x, x_train)
                distances.append(distance)

            # Get indices of k nearest neighbors
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]

            # Majority vote
            most_common = Counter(k_nearest_labels).most_common(1)
            predictions.append(most_common[0][0])

        return np.array(predictions)


# -----------------------------
# Load Iris Dataset
# -----------------------------
iris = load_iris()
X = iris.data
y = iris.target

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Train KNN Model
# -----------------------------
k = 5
knn = KNN(k=k)
knn.fit(X_train, y_train)

# -----------------------------
# Make Predictions
# -----------------------------
y_pred = knn.predict(X_test)

# -----------------------------
# Evaluate Accuracy
# -----------------------------
accuracy = np.mean(y_pred == y_test)
print(f"KNN Accuracy (k={k}): {accuracy:.2f}")
