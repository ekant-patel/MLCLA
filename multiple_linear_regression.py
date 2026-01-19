def transpose(M):
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]

def matmul(A, B):
    m, n, p = len(A), len(A[0]), len(B[0])
    result = [[0]*p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            for k in range(n):
                result[i][j] += A[i][k] * B[k][j]
    return result

def inverse(M):
    n = len(M)
    AM = [row[:] for row in M]
    IM = [[float(i == j) for i in range(n)] for j in range(n)]
    for fd in range(n):
        pivot = AM[fd][fd]
        for j in range(n):
            AM[fd][j] /= pivot
            IM[fd][j] /= pivot
        for i in range(n):
            if i != fd:
                factor = AM[i][fd]
                for j in range(n):
                    AM[i][j] -= factor * AM[fd][j]
                    IM[i][j] -= factor * IM[fd][j]
    return IM

def linear_regression_multi(X, y):
    X_b = [[1] + row for row in X]
    X_T = transpose(X_b)
    XT_X = matmul(X_T, X_b)
    XT_y = matmul(X_T, [[yi] for yi in y])
    XT_X_inv = inverse(XT_X)
    theta = matmul(XT_X_inv, XT_y)
    return theta

X = [
    [60, 22],
    [62, 25],
    [67, 24],
    [70, 20],
    [71, 15],
    [72, 14],
    [75, 14],
    [78, 11],
]

y = [140,155,159,179,192,200,212,215]

theta = linear_regression_multi(X, y)

print("Intercept:", theta[0][0])
for i in range(1, len(theta)):
    print("Coefficient", i, ":", theta[i][0])
