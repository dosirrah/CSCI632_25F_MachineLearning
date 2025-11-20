import numpy as np
import pandas as pd
from sklearn.datasets import make_moons

# Generate a moons dataset
X_moons, y_moons = make_moons(n_samples=200, noise=0.2, random_state=42)
X_moons += 1.5
# Save Moons dataset to CSV
data_moons = pd.DataFrame(np.hstack((X_moons, y_moons.reshape(-1, 1))),
                          columns=['$x_1$', '$x_2$', 'target'])
data_moons.to_csv('ex3_moons.csv', index=False)
print("Moons dataset saved as 'ex3_moons.csv'.")
