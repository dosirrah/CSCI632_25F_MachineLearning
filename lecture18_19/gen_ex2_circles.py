import numpy as np
import pandas as pd
from sklearn.datasets import make_circles

# Generate concentric circles dataset using make_circles
X, y = make_circles(n_samples=200, factor=0.5, noise=0.1, random_state=0)
data_circles = pd.DataFrame(np.hstack((X, y.reshape(-1, 1))), columns=['$x_1$', '$x_2$', 'target'])

# Save to CSV
data_circles.to_csv('ex2_circles.csv', index=False)
print("Concentric circles dataset saved as 'ex2_circles.csv'.")
