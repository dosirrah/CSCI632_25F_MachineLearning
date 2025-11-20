import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

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
fig_global = None
ax_main = None
ax_sig = None
cbar_global = None

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


    # --- globals ---
zscale_ema = None  # smooth estimate of ||w|| * x_max

def sigmoid_bounds_ema(theta, X, alpha=0.2):
    """
    Returns a smoothly updated symmetric z-range centered at 0 for the sigmoid plot.
    z ∈ [-L, L] where L ≈ ||w|| * x_max (EWMA-smoothed).

    Parameters:
        theta : array-like (bias + weights)
        X     : data matrix (samples × features)
        alpha : smoothing factor in [0,1]; higher = faster response
    """
    global zscale_ema
    w = theta[1:]                     # exclude bias
    x_max = np.linalg.norm(X, axis=1).max()
    L_tgt = np.linalg.norm(w) * x_max

    # initialize or update exponentially weighted moving average
    if zscale_ema is None:
        zscale_ema = L_tgt
    else:
        zscale_ema = (1 - alpha) * zscale_ema + alpha * L_tgt

    # always return symmetric range centered at 0
    return -zscale_ema, zscale_ema


def plot_decision_boundary(X, y, model, step, sample_index=None) -> int:
    """
    Plot the decision boundary (top) and a logistic panel (bottom) showing
    z = w^T x on the horizontal axis and sigma(z) on the vertical axis.
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
    global fig_global, ax_main, ax_sig, cbar_global

    return_val = None
    theta = model.theta
    color = ["blue", "red"]

    # --- figure / axes setup -------------------------------------------------
    if not plot_initialized:
        plt.ion()
        fig_global = plt.figure(figsize=(8, 8))
        gs = fig_global.add_gridspec(nrows=2, ncols=1, height_ratios=[4, 1], hspace=0.18)
        ax_main = fig_global.add_subplot(gs[0, 0])
        ax_sig  = fig_global.add_subplot(gs[1, 0])

        # callbacks
        fig_global.canvas.mpl_connect('key_press_event', on_key)
        fig_global.canvas.mpl_connect('button_press_event', on_click)

        plot_initialized = True
    else:
        # reuse existing axes
        # (fig_global, ax_main, ax_sig) were stored in globals
        pass

    ax_main.cla()
    ax_sig.cla()

    # --- main decision surface ----------------------------------------------
    ax_main.set_aspect('equal', adjustable='box')

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()

    # keep square view with padding
    dimension = max(x_max - x_min, y_max - y_min) + 2
    mid_x = (x_max + x_min) / 2
    mid_y = (y_max + y_min) / 2
    x_min, x_max = mid_x - dimension / 2, mid_x + dimension / 2
    y_min, y_max = mid_y - dimension / 2, mid_y + dimension / 2
    ax_main.set_xlim(x_min, x_max)
    ax_main.set_ylim(y_min, y_max)

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))

    # raw z over the grid (for z-range in logistic panel)
    Z_raw = theta[0] + theta[1] * xx + theta[2] * yy
    Z = 1 / (1 + np.exp(-Z_raw))

    contour = ax_main.contourf(xx, yy, Z, levels=20, cmap="viridis", alpha=0.6)
    if cbar_global is None:
        cbar_global = fig_global.colorbar(contour, ax=ax_main, pad=0.02)
    else:
        # update the colorbar mappable so it follows changes
        cbar_global.update_normal(contour)

    # data points
    ax_main.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color=color[0], label="Class 0")
    ax_main.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color=color[1], label="Class 1")

    # decision boundary line
    decision_boundary_x = np.linspace(x_min, x_max, 200)
    decision_boundary_y = -(theta[1] * decision_boundary_x + theta[0]) / theta[2]
    ax_main.plot(decision_boundary_x, decision_boundary_y, 'k-', linewidth=2.5, label="Decision Boundary")

    # biased normal arrow
    offset_position = np.array([0., 0.])
    if theta[1] != 0:
        offset_position[0] = -theta[0] / theta[1]
        offset_label = f"$-\\Theta_0/\\Theta_1$ = {-theta[0] / theta[1]:.3f}"
    else:
        # fall back to y-axis offset if needed
        offset_position[1] = -theta[0] / (theta[2] if theta[2] != 0 else 1.0)
        offset_label = f"$-\\Theta_0/\\Theta_2$ = {-theta[0] / (theta[2] if theta[2] != 0 else np.nan):.3f}"

    ax_main.quiver(offset_position[0], offset_position[1], theta[1], theta[2],
                   angles='xy', scale_units='xy', scale=1, color="black", alpha=0.7,
                   label="Surface Normal (biased)")

    ax_main.annotate('', xy=(offset_position[0], offset_position[1]), xytext=(0, 0),
                     arrowprops=dict(arrowstyle='<->', color='white', linewidth=3))

    mid = offset_position / 2
    text_x, text_y = 0.8, 0.1
    text_label = ax_main.text(text_x, text_y, offset_label, ha='center', va='top',
                              color='white', transform=ax_main.transAxes)

    display_coord = ax_main.transAxes.transform((text_x, text_y))
    data_coord = ax_main.transData.inverted().transform(display_coord)
    ax_main.annotate('', xy=(data_coord[0], data_coord[1]), xytext=(mid[0], mid[1]),
                     arrowprops=dict(arrowstyle='-', color='white', linestyle='--', linewidth=2))

    # highlight current sample & update vector
    z_sample = None
    sig_sample = None
    if sample_index is not None:
        x_sample = X[sample_index]
        y_sample = int(round(y[sample_index]))
        ax_main.scatter(x_sample[0], x_sample[1], color=color[y_sample],
                        edgecolor="white", s=100, linewidths=4)

        # shifted vector to sample
        ax_main.quiver(offset_position[0], offset_position[1],
                       x_sample[0] - offset_position[0], x_sample[1] - offset_position[1],
                       angles='xy', scale_units='xy', scale=1, color="white")

        # update vector (visual)
        upvec = -2 * model.gradient(x_sample, y_sample)
        ax_main.quiver(offset_position[0] + theta[1], offset_position[1] + theta[2],
                       upvec[1], upvec[2], angles='xy', scale_units='xy',
                       scale=1, color="red", linewidth=3)

        # compute z and sigma(z) for logistic panel
        z_sample = theta[0] + theta[1] * x_sample[0] + theta[2] * x_sample[1]
        sig_sample = 1 / (1 + np.exp(-z_sample))

    ax_main.set_title(f"Decision Boundary Update - Step {step}")
    ax_main.set_xlabel('$x_1$')
    ax_main.set_ylabel('$x_2$')
    ax_main.legend()
    ax_main.grid(True)

    # --- bottom logistic panel ----------------------------------------------
    # Choose a z-range that covers what the current surface produces.
    # given theta = [theta0, w1, w2, ...], X shape (n_samples, n_features)
    
    #z_min = float(np.nanmin(Z_raw))
    #z_max = float(np.nanmax(Z_raw))
    ## pad & keep a reasonable minimum span
    #span = max(6.0, (z_max - z_min))
    #z_center = 0.5 * (z_max + z_min)
    #z_min = z_center - 0.5 * span
    #z_max = z_center + 0.5 * span
    z_min, z_max = sigmoid_bounds_ema(model.theta, X, alpha=0.2)

    z_vals = np.linspace(z_min, z_max, 400)
    sig_vals = 1 / (1 + np.exp(-z_vals))

    ax_sig.plot(z_vals, sig_vals, linewidth=2)
    ax_sig.set_ylim(-0.05, 1.05)
    ax_sig.set_xlim(z_min, z_max)
    ax_sig.set_xlabel(r'$z = \mathbf{\Theta}^\top \mathbf{x}$')
    ax_sig.set_ylabel(r'$\sigma(z)$')
    ax_sig.grid(True, alpha=0.5)

    # mark current sample's z (if any)
    if z_sample is not None:
        ax_sig.axvline(z_sample, linestyle='--')
        ax_sig.plot([z_sample], [sig_sample], marker='o', markersize=8)
        ax_sig.text(z_sample, sig_sample,
                    f"  z={z_sample:.2f}\n  σ(z)={sig_sample:.2f}",
                    va='center', ha='left')

    # --- draw/update & wait --------------------------------------------------
    plt.draw()
    plt.pause(0.1)

    print("Press button to continue...")
    while not plt.waitforbuttonpress() and return_val is None:
        plt.pause(0.1)

    return return_val    

# def plot_decision_boundary(X, y, model, step, sample_index=None) -> int:
#     """
#     Plot the decision boundary with a thick line, highlight the sample used for updating,
#     and visualize vectors representing feature and surface normal with a bias offset.
# 
#     Parameters:
#     - X: Features matrix (numpy array)
#     - y: Labels (numpy array)
#     - model: logistic regression model
#     - step: Current step number in the training process
#     - sample_index: Index of the sample being used for the current update (optional)
# 
#     Returns:
#      -1 step backwards
#       0 done. quit display, exit!
#       1 step forward
#     """
#     global EXIT_KEYS, STEP_BACK_KEYS
#     global return_val, plot_initialized
# 
#     return_val = None
#     
# 
#     theta = model.theta
#     
#     # node color
#     color = ["blue", "red"]
#     
#     if not plot_initialized:
#         plt.ion()  # Enable interactive mode
#         fig, ax = plt.subplots(figsize=(8, 6))
# 
#         # # The next two lines don't work in Mac OS.
#         # manager = fig.canvas.manager
#         # manager.window.wm_geometry("1024x768+100+100")
# 
#         # install key callbacks so we can go forward, backward, or quit
#         # displaying the plot.
#         fig.canvas.mpl_connect('key_press_event', on_key)
#         fig.canvas.mpl_connect('button_press_event', on_click)
#     else:
#         ax = plt.gca()
#         fig = ax.get_figure()
#     
#     ax.cla()  # Clear previous plot contents to update in the same window
# 
#     # Ensure equal aspect ratio
#     ax.set_aspect('equal', adjustable='box')
# 
#     # Define grid for contour plot
#     x_min, x_max = X[:, 0].min(), X[:, 0].max()
#     y_min, y_max = X[:, 1].min(), X[:, 1].max()
# 
#     # I want the axes to have the same scale so let's take the larger of the two
#     # and use it both directions, and that the contours and meshgrid span 
#     # entire plot.
#     dimension = max(x_max - x_min, y_max - y_min) + 2
#     mid_x = (x_max + x_min) / 2
#     x_min = mid_x - dimension / 2
#     x_max = mid_x + dimension / 2
#     mid_y = (y_max + y_min) / 2
#     y_min = mid_y - dimension / 2 
#     y_max = mid_y + dimension / 2
#     ax.set_xlim(x_min, x_max)  # Set x-axis bounds
#     ax.set_ylim(y_min, y_max)  # Set y-axis bounds
# 
#     xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
#     
#     # Calculate decision boundary values using sigmoid function
#     Z = theta[0] + theta[1] * xx + theta[2] * yy 
#     Z = 1 / (1 + np.exp(-Z))  # Apply sigmoid function for decision boundary
# 
#     # Plot contour for decision boundary probability levels
#     contour = ax.contourf(xx, yy, Z, levels=20, cmap="viridis", alpha=0.6)
#     if not plot_initialized:
#         fig.colorbar(contour)
#         plot_initialized = True
# 
#     # Scatter plot for data points
#     ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], color=color[0], label="Class 0")
#     ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], color=color[1], label="Class 1")
# 
#     ###
#     # Define the decision boundary line
#     decision_boundary_x = np.linspace(x_min, x_max, 200)
#     decision_boundary_y = -(theta[1] * decision_boundary_x + theta[0]) / theta[2]
#     ax.plot(decision_boundary_x, decision_boundary_y, 'k-', linewidth=2.5, label="Decision Boundary")
# 
#     # Plot surface normal vector from the origin (weights give the direction)
#     #ax.quiver(0, 0, theta[1], theta[2], angles='xy', scale_units='xy',
#     #    scale=1, color="black", alpha=0.7, label="Surface Normal (origin)")
# 
#     print(f"theta: {theta}")
#     
#     # Calculate offset position along x_1 or x_2 to represent bias
#     offset_position = np.array([0., 0.])
#     if theta[0] != 0:
#         offset_position[0] = -theta[0] / theta[1]
#         offset_label = f"$-\Theta_0 / \Theta_1$ = {-theta[0] / theta[1]:.3f}"
# 
#     elif theta[1] != 0:
#         offset_position[1] = -theta[0] / theta[2]
#         offset_label = f"$-\Theta_0 / \Theta_2$ = { -theta[0] / theta[2]:.3f}"
# 
#     # dotted line showing the shift
#     print(f"offset_position: {offset_position}")
#     
#     # Plot shifted surface normal vector at decision boundary level
#     ax.quiver(offset_position[0], offset_position[1], theta[1], theta[2], angles='xy',
#               scale_units='xy', scale=1, color="black", alpha=0.7,
#               label="Surface Normal (biased)")
# 
#     plt.annotate(
#         '', xy=(offset_position[0], offset_position[1]), xytext=(0, 0),
#         arrowprops=dict(arrowstyle='<->', color='white', linewidth=3)
#     )
# 
#     mid = offset_position / 2
#            
#     # Define fixed position for the label in axis-relative coordinates
#     # (e.g., bottom-right corner)
#     text_x, text_y = 0.8, 0.1  # Adjust as needed for placement within the axis
#     
#     # Add text at the fixed position
#     text_label = ax.text(
#         text_x, text_y, offset_label, ha='center', va='top', 
#         color='white', transform=ax.transAxes
#     )
# 
#     # Convert the fixed text position in axis-relative coordinates to data coordinates
#     display_coord = ax.transAxes.transform((text_x, text_y))  # Convert to display coordinates
#     data_coord = ax.transData.inverted().transform(display_coord)  # Convert display to data coordinates
# 
#     # Draw a line (arrow) from the midpoint in data coordinates to the calculated
#     # text position in data coordinates
#     ax.annotate(
#         '', xy=(data_coord[0], data_coord[1]), xytext=(mid[0], mid[1]),
#         arrowprops=dict(arrowstyle='-', color='white', linestyle='--', linewidth=2)
#     )
# 
#     # Highlight the current sample used for update, if provided
#     if sample_index is not None:
#         x_sample = X[sample_index]
#         y_sample = round(y[sample_index])
#         print(f"y_sample={y_sample}")
#         ax.scatter(x_sample[0], x_sample[1], color=color[y_sample], edgecolor="white", s=100,
#                    linewidths=4)
# 
#         # Plot vector from origin to the sample point
#         #ax.quiver(0, 0, x_sample[0], x_sample[1], angles='xy', scale_units='xy', 
#         #          scale=1, color="white")
#         
#         # Plot shifted vector to sample point at decision boundary level
#         ax.quiver(offset_position[0], offset_position[1], 
#                   x_sample[0] - offset_position[0], x_sample[1] - offset_position[1], 
#                   angles='xy', scale_units='xy', 
#                   scale=1, color="white")
# 
#         # plot update vector.
#         upvec = -2*model.gradient(x_sample, y_sample)
#         ax.quiver(offset_position[0] + theta[1], offset_position[1] + theta[2], 
#                   upvec[1], upvec[2], 
#                   angles='xy', scale_units='xy', 
#                   scale=1, color="red", linewidth=3)
#         
# 
#     # Labels and legend
#     ax.set_title(f"Decision Boundary Update - Step {step}")
#     ax.set_xlabel('$x_1$')
#     ax.set_ylabel('$x_2$')
#     ax.legend()
#     ax.grid(True)
# 
#     #plt.show()
#     
#     # Draw the updated plot
#     plt.draw()
#     plt.pause(0.1)  # Pause to update plot
# 
#     # Use plt.pause with an indefinite loop to keep the window responsive
#     print("Press button to continue...")
#     while not plt.waitforbuttonpress() and return_val is None:  
#         plt.pause(0.1)  # Keep the plot window responsive
#         print(f"return_val={return_val}")
# 
#     return return_val


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
    

    
    
