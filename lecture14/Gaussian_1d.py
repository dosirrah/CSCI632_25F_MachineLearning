import numpy as np
import matplotlib.pyplot as plt

# Define the parameters for the Gaussian distribution
mu = 0       # mean
sigma = 1    # standard deviation

# Create a range of x values
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)

# Compute the Gaussian probability density function (PDF)
pdf = (1/(np.sqrt(2 * np.pi * sigma**2))) * np.exp(- (x - mu)**2 / (2 * sigma**2))

# Plot the Gaussian distribution
plt.plot(x, pdf, label=f'Gaussian: $\mu$={mu}, $\sigma$={sigma}')
plt.title('1D Gaussian Distribution')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.legend()
plt.show()
