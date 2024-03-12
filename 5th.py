import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            # Calculate predicted values
            y_pred = np.dot(X, self.weights) + self.bias

            # Compute cost
            cost = (1 / (2 * n_samples)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            # Update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Get input data from user
num_samples = int(input("Enter the number of data points: "))
X = []
y = []
for i in range(num_samples):
    x_val = float(input(f"Enter x{i+1}: "))
    y_val = float(input(f"Enter y{i+1}: "))
    X.append([x_val])
    y.append(y_val)
X = np.array(X)
y = np.array(y)

# Fit linear regression model
model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

# Plotting the data points and the fit line
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, color='blue', label='Data points')
plt.plot(X[:, 0], model.predict(X), color='red', label='Best-fit line')
plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()