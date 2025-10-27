import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

def plot_2d_gaussian(mu, sigma):
    """
    Plots a 2D Gaussian distribution given a mean vector and a covariance matrix.
    
    Parameters:
    - mu: Mean vector of size 2 (list or numpy array)
    - sigma: 2x2 Covariance matrix (numpy array)
    """
    # Create a grid of points over which to evaluate the Gaussian
    x, y = np.meshgrid(np.linspace(mu[0] - 3*np.sqrt(sigma[0, 0]), mu[0] + 3*np.sqrt(sigma[0, 0]), 100),
                       np.linspace(mu[1] - 3*np.sqrt(sigma[1, 1]), mu[1] + 3*np.sqrt(sigma[1, 1]), 100))
    pos = np.dstack((x, y))

    # Create a 2D Gaussian based on the mean and covariance
    rv = multivariate_normal(mu, sigma)

    # Plot the Gaussian distribution as a contour plot
    plt.contourf(x, y, rv.pdf(pos), levels=20, cmap='viridis')
    plt.colorbar(label='Probability Density')
    plt.title('2D Gaussian Distribution')
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Example usage
    mu = [0, 0]  # Mean vector
    sigma = np.array([[1, 0.5], [0.5, 1]])  # Covariance matrix

    plot_2d_gaussian(mu, sigma)
