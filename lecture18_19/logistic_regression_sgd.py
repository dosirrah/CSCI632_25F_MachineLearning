import matplotlib.pyplot as plt
import numpy as np

# SGD = Stochastic Gradient Descent
class LogisticRegressionSGD:
    def __init__(self, learning_rate=0.01, epochs=100):
        self._learning_rate = learning_rate
        self._epochs = epochs
        self._theta_history = []  # Parameter vector including weights and bias

    @property
    def theta(self):
        return self._theta_history[-1]
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, n_features):
        # Initialize theta (weights + bias as the first element)
        theta = np.random.randn(n_features + 1)
        self._theta_history.append(theta)
    
    def gradient(self, x, y):
        """Returns the scaled gradeint.  The gradient is scaled
           by the difference between the the predicted posterior
           probability y_predicted and the known truth y."""
        # Add a bias term (x_0 = 1) for each x sample
        x_with_bias = np.insert(x, 0, 1)
        
        # Compute prediction (hypothesis)
        z = np.dot(self.theta, x_with_bias)
        y_predicted = self.sigmoid(z)
        
        # Compute the gradient for each theta_j
        gradient = (y_predicted - y) * x_with_bias
        return gradient
        
    def update(self, x, y):

        theta = self._theta_history[-1].copy()
        
        # Update parameters (theta) using SGD
        theta -= self._learning_rate * self.gradient(x, y)
        self._theta_history.append(theta)
        
        # Return current theta (weights and bias)
        return theta

    def update_backward(self):
        if len(self._theta_history) > 1:
            self._theta_history.pop()
        return self._theta_history[-1]
        
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_parameters(n_features)

        for epoch in range(self._epochs):
            for i in range(n_samples):
                self.update(X[i], y[i])
    
    def predict(self, X):
        # Add bias term to each sample
        X_with_bias = np.insert(X, 0, 1, axis=1)
        z = np.dot(X_with_bias, self.theta)
        y_predicted = self.sigmoid(z)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)



    

import matplotlib.transforms as transforms
EXIT_KEYS = ["escape", "q"]
STEP_BACK_KEYS = ["left", "backspace"]
return_val = None
plot_initialized = False

def on_key(event):
    global EXIT_KEYS, STEP_BACK_KEYS
    global return_val
    print(f"Key pressed: {event.key}")
    if event.key in EXIT_KEYS:
        return_val = 0
    elif event.key in STEP_BACK_KEYS:
        return_val = -1
    else:
        return_val = 1
        
def on_click(event):
    global return_val
    print(f"Mouse button pressed: {event.button}")
    return_val = 1


def plot_decision_boundary(X, y, model, step, sample_index=None) -> int:
    """
    Plot the decision boundary with a thick line, highlight the sample used for updating,
    and visualize vectors representing feature and surface normal with a bias offset.

    Parameters:
    - X: Features matrix (numpy array)
    - y: Labels (numpy array)
    - model: logistic regression model
    - step: Current step number in the training process
    - sample_index: Index of the sample being used for the current update (optional)

    Returns:
     -1 step backwards
      0 done. quit display, exit!
      1 step forward
    """
    global EXIT_KEYS, STEP_BACK_KEYS
    global return_val, plot_initialized

    return_val = None
    

    theta = model.theta
    
    # node color
    color = ["blue", "red"]
    
    if not plot_initialized:
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots(figsize=(8, 6))

        # # The next two lines don't work in Mac OS.
        # manager = fig.canvas.manager
        # manager.window.wm_geometry("1024x768+100+100")

        # install key callbacks so we can go forward, backward, or quit
        # displaying the plot.
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('button_press_event', on_click)
    else:
        ax = plt.gca()
        fig = ax.get_figure()
    
    ax.cla()  # Clear previous plot contents to update in the same window

    # Ensure equal aspect ratio
    ax.set_aspect('equal', adjustable='box')

    # Define grid for contour plot
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    # I want the axes to have the same scale so let's take the larger of the two
    # and use it both directions, and that the contours and meshgrid span 
    # entire plot.
    dimension = max(x_max - x_min, y_max - y_min) + 2
    mid_x = (x_max + x_min) / 2
    x_min = mid_x - dimension / 2
    x_max = mid_x + dimension / 2
    mid_y = (y_max + y_min) / 2
    y_min = mid_y - dimension / 2 
    y_max = mid_y + dimension / 2
    ax.set_xlim(x_min, x_max)  # Set x-axis bounds
    ax.set_ylim(y_min, y_max)  # Set y-axis bounds

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    
    # Calculate decision boundary values using sigmoid function
    Z = theta[0] + theta[1] * xx + theta[2] * yy 
    Z = 1 / (1 + np.exp(-Z))  # Apply sigmoid function for decision boundary

    # Plot contour for decision boundary probability levels
    contour = ax.contourf(xx, yy, Z, levels=20, cmap="viridis", alpha=0.6)
    if not plot_initialized:
        fig.colorbar(contour)
        plot_initialized = True

    # Scatter plot for data points
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color=color[0], label="Class 0")
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color=color[1], label="Class 1")

    ###
    # Define the decision boundary line
    decision_boundary_x = np.linspace(x_min, x_max, 200)
    decision_boundary_y = -(theta[1] * decision_boundary_x + theta[0]) / theta[2]
    ax.plot(decision_boundary_x, decision_boundary_y, 'k-', linewidth=2.5, label="Decision Boundary")

    # Plot surface normal vector from the origin (weights give the direction)
    #ax.quiver(0, 0, theta[1], theta[2], angles='xy', scale_units='xy',
    #    scale=1, color="black", alpha=0.7, label="Surface Normal (origin)")

    print(f"theta: {theta}")
    
    # Calculate offset position along x_1 or x_2 to represent bias
    offset_position = np.array([0., 0.])
    if theta[0] != 0:
        offset_position[0] = -theta[0] / theta[1]
        offset_label = f"$-\Theta_0 / \Theta_1$ = {-theta[0] / theta[1]:.3f}"

    elif theta[1] != 0:
        offset_position[1] = -theta[0] / theta[2]
        offset_label = f"$-\Theta_0 / \Theta_2$ = { -theta[0] / theta[2]:.3f}"

    # dotted line showing the shift
    print(f"offset_position: {offset_position}")
    
    # Plot shifted surface normal vector at decision boundary level
    ax.quiver(offset_position[0], offset_position[1], theta[1], theta[2], angles='xy',
              scale_units='xy', scale=1, color="black", alpha=0.7,
              label="Surface Normal (biased)")

    plt.annotate(
        '', xy=(offset_position[0], offset_position[1]), xytext=(0, 0),
        arrowprops=dict(arrowstyle='<->', color='white', linewidth=3)
    )

    mid = offset_position / 2
           
    # Define fixed position for the label in axis-relative coordinates
    # (e.g., bottom-right corner)
    text_x, text_y = 0.8, 0.1  # Adjust as needed for placement within the axis
    
    # Add text at the fixed position
    text_label = ax.text(
        text_x, text_y, offset_label, ha='center', va='top', 
        color='white', transform=ax.transAxes
    )

    # Convert the fixed text position in axis-relative coordinates to data coordinates
    display_coord = ax.transAxes.transform((text_x, text_y))  # Convert to display coordinates
    data_coord = ax.transData.inverted().transform(display_coord)  # Convert display to data coordinates

    # Draw a line (arrow) from the midpoint in data coordinates to the calculated
    # text position in data coordinates
    ax.annotate(
        '', xy=(data_coord[0], data_coord[1]), xytext=(mid[0], mid[1]),
        arrowprops=dict(arrowstyle='-', color='white', linestyle='--', linewidth=2)
    )

    # Highlight the current sample used for update, if provided
    if sample_index is not None:
        x_sample = X[sample_index]
        y_sample = round(y[sample_index])
        print(f"y_sample={y_sample}")
        ax.scatter(x_sample[0], x_sample[1], color=color[y_sample], edgecolor="white", s=100,
                   linewidths=4)

        # Plot vector from origin to the sample point
        #ax.quiver(0, 0, x_sample[0], x_sample[1], angles='xy', scale_units='xy', 
        #          scale=1, color="white")
        
        # Plot shifted vector to sample point at decision boundary level
        ax.quiver(offset_position[0], offset_position[1], 
                  x_sample[0] - offset_position[0], x_sample[1] - offset_position[1], 
                  angles='xy', scale_units='xy', 
                  scale=1, color="white")

        # plot update vector.
        upvec = -2*model.gradient(x_sample, y_sample)
        ax.quiver(offset_position[0] + theta[1], offset_position[1] + theta[2], 
                  upvec[1], upvec[2], 
                  angles='xy', scale_units='xy', 
                  scale=1, color="red", linewidth=3)
        

    # Labels and legend
    ax.set_title(f"Decision Boundary Update - Step {step}")
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.legend()
    ax.grid(True)

    #plt.show()
    
    # Draw the updated plot
    plt.draw()
    plt.pause(0.1)  # Pause to update plot

    # Use plt.pause with an indefinite loop to keep the window responsive
    print("Press button to continue...")
    while not plt.waitforbuttonpress() and return_val is None:  
        plt.pause(0.1)  # Keep the plot window responsive
        print(f"return_val={return_val}")

    return return_val


def main():
    import argparse

    # Create the parser
    parser = argparse.ArgumentParser(description="Apply Logistic Regression SGD to a specified file with optional seed.")
    
    # Add the positional argument for the file
    parser.add_argument('file', type=str, help="The file to which logistic regression will be applied.")
    
    # Add the optional argument for the seed
    parser.add_argument('--seed', type=int, default=83, help="Random seed for reproducibility.")

    # This default for alpha is insanely high, but this code is just for illlustration and
    # using a larger learning rate allows me to show significant changes to parameters
    # in a single update step.
    parser.add_argument('--alpha', type=float, default=0.4, help="learning rate")
    
    # Parse the arguments
    args = parser.parse_args()
    
    # Set the seed if specified
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Load data from the specified file
    # Assuming the file is a CSV with X and y
    import pandas as pd
    data = pd.read_csv(args.file)
    X = data.iloc[:, :-1].values  # All columns except the last as features
    y = data.iloc[:, -1].values   # The last column as target

    # Initialize the logistic regression model
    np.random.seed(args.seed)
    learning_rate = args.alpha
    step = 0
    model = LogisticRegressionSGD(learning_rate=learning_rate)
    
    # X is the "design matrix" where each row is a data point and 
    # the columns are features.
    model.initialize_parameters(X.shape[1])  # X.shape[0] is num rows. X.shape[1] = num columns
    
    step = i = 0
    while True:
        i = (step + 1) % len(y)
        action = plot_decision_boundary(X, y, model, step = step, sample_index=i)
        if action == 0:
            exit(0)
        elif action == -1:
            if step > 0:
                model.update_backward()
                step -= 1
        else:
            model.update(X[i], y[i])
            step += 1

if __name__ == "__main__":
    main()
    

    
    
