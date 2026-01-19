import numpy as np

# -----------------------------
# Batch Gradient Descent (BGD)
# -----------------------------
def batch_gradient_descent(x, y, alpha, b0, b1, epochs):
    n = len(x)

    for _ in range(epochs):
        y_pred = b0 + b1 * x
        error = y_pred - y
        
        # Gradients
        db0 = (1/n) * np.sum(error)
        db1 = (1/n) * np.sum(error * x)

        # Update parameters
        b0 -= alpha * db0
        b1 -= alpha * db1

    return b0, b1


# ---------------------------------------
# Stochastic Gradient Descent (SGD)
# ---------------------------------------
def stochastic_gradient_descent(x, y, alpha, b0, b1, epochs):
    n = len(x)

    for _ in range(epochs):
        for i in range(n):
            xi = x[i]
            yi = y[i]

            y_pred = b0 + b1 * xi
            error = y_pred - yi

            # Update parameters using one sample
            b0 -= alpha * error
            b1 -= alpha * error * xi

    return b0, b1


# -----------------------------
# Example Use
# -----------------------------
if __name__ == "__main__":
    # Sample data (replace with any dataset)
    x = np.array([1, 2, 3, 4, 5], dtype=float)
    y = np.array([5,6,10,13,11], dtype=float)

    # Initial values
    alpha = 0.1
    b0_init = 3.3
    b1_init = 1.9
    epochs = 10

    # Run Batch GD
    b0_bgd, b1_bgd = batch_gradient_descent(x, y, alpha, b0_init, b1_init, epochs)
    print("Batch GD:  b0 =", b0_bgd, ", b1 =", b1_bgd)

    # Run Stochastic GD
    b0_sgd, b1_sgd = stochastic_gradient_descent(x, y, alpha, b0_init, b1_init, epochs)
    print("Stochastic GD:  b0 =", b0_sgd, ", b1 =", b1_sgd)
