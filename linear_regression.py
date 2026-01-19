def linear_regression(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(x[i] * y[i] for i in range(n))
    sum_x2 = sum(x[i] * x[i] for i in range(n))

    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
    b = (sum_y - m * sum_x) / n

    return m, b

x = [-2, -1, 0, 1,2]
y = [65, 95, 80, 115, 105]  

m, b = linear_regression(x, y)

print("Slope (m):", m)
print("Intercept (b):", b)
x_new = 5
y_pred = m * x_new + b
#print("Prediction for x = 5:", y_pred)
