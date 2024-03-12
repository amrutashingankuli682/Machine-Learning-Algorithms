import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
heights = np.array([160, 165, 170, 175, 180, 185, 190]).reshape(-1, 1)
weights = np.array([55, 60, 63, 70, 75, 77, 80]).reshape(-1, 1)
model = LinearRegression()
model.fit(heights, weights)
new_height = np.array([[170]])
predicted_weight = model.predict(new_height)
print(f"Predicted weight for a height of {new_height[0][0]} cm: {predicted_weight[0][0]} kg")
plt.scatter(heights, weights, color='blue', label='Data Points')
plt.plot(heights, model.predict(heights), color='black', linewidth=3, label='Regression Line')
plt.scatter(new_height, predicted_weight, color='red', marker='*', s=200, label='Predicted Weight')
plt.xlabel('Height (cm)')
plt.ylabel('Weight (kg)')
plt.title('Linear Regression: Height vs. Weight')
plt.legend()
plt.show()