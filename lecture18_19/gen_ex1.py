import numpy as np
import pandas as pd

np.random.seed(42)

mean_0 = [-1, 1]
cov_0 = [[1, -0.8],
         [-0.8, 1]]
mean_1 = [2, 2]
cov_1 = [[1, 0],
         [0, 1]]

# Shuffle the dataset
def shuffle_data(X, y):
    # Generate a shuffled index array
    indices = np.random.permutation(X.shape[0])
    # Shuffle X and y using the indices
    return X[indices], y[indices]


# Generate synthetic data for class 0 (centered near the negative quadrant)
class_0 = np.random.multivariate_normal(mean_0, cov_0, 100)
#class_0 = class_0[(class_0[:, 0] < 0) & (class_0[:, 1] < 0)]  # Clip to negative quadrant

# Generate synthetic data for class 1 (centered near the positive quadrant)
class_1 = np.random.multivariate_normal(mean_1, cov_1, 100)
#class_1 = class_1[(class_1[:, 0] > 0) & (class_1[:, 1] > 0)]  # Clip to positive quadrant

# Combine data and labels
X = np.vstack((class_0, class_1))
y = np.hstack((np.zeros(len(class_0)), np.ones(len(class_1))))

X, y = shuffle_data(X, y)

# Convert X and y to a DataFrame
data = pd.DataFrame(X, columns=[f'$x_{i+1}$' for i in range(X.shape[1])])
data['target'] = y  # Add y as the target column

# Save to CSV
data.to_csv('ex1.csv', index=False)
