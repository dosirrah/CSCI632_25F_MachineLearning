
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Parameters
n_samples = 400  # Total number of points
noise = 0.15      # Amount of noise to add to the points

# Generate XOR pattern points
np.random.seed(42)  # For reproducibility
X = np.empty((0, 2))
y = np.empty(0, dtype=int)

# Define XOR logic
xor_points = [(0, 0), (1, 1), (0, 1), (1, 0)]
labels = [0, 0, 1, 1]  # Points (0,0) and (1,1) are in class 0, others in class 1

for (x1, x2), label in zip(xor_points, labels):
    X = np.vstack([X, np.random.normal((x1, x2), noise, (n_samples // 4, 2))])
    y = np.hstack([y, np.full(n_samples // 4, label)])

# shuffle the data.
shuffled_indices = np.random.permutation(len(y))
X = X[shuffled_indices]
y = y[shuffled_indices]


# Visualize the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color='blue', label="Class 0", alpha=0.6)
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color='red', label="Class 1", alpha=0.6)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")
plt.legend()
plt.title("XOR Dataset")
plt.show()

# Save to CSV
data_xor = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))), columns=['$x_1$', '$x_2$', 'target'])
data_xor.to_csv('ex4_xor.csv', index=False)
print("XOR dataset saved as 'ex4_xor.csv'.")
