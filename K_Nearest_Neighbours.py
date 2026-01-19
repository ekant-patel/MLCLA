import math

def euclidean_distance(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(len(a))))

def knn_regression(X_train, y_train, x_test, k):
    distances = []

    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_test)
        distances.append((dist, y_train[i]))

    distances.sort(key=lambda x: x[0])

    total = 0
    for i in range(k):
        total += distances[i][1]

    return total / k
X_train = [[1,2], [2,3], [3,4], [6,7]]
y_train = [2, 3, 4, 7]

x_test = [3,3]
k = 2

print(knn_regression(X_train, y_train, x_test, k))
