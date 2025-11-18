import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import argparse

# Global variables for user interaction
return_val = None
plot_initialized = False

def on_key(event):
    global return_val
    print(f"Key pressed: {event.key}")
    if event.key in ["escape", "q"]:
        return_val = 0  # Exit
    elif event.key in ["left", "backspace"]:
        return_val = -1  # Step backward
    else:
        return_val = 1  # Step forward

def on_click(event):
    global return_val
    print(f"Mouse button pressed: {event.button}")
    return_val = 1  # Step forward

class Neuron:
    def __init__(self, learning_rate=0.01):
        self._learning_rate = learning_rate
        self._theta_history = []  # Parameter vector including weights and bias
        self._cache = None        # Cache for storing intermediate values during forward pass

    @property
    def last_activation(self):
        """Returns the last activation value computed during the forward pass."""
        if self._cache is None:
            raise ValueError("No cache available. Ensure that forward() has been called before accessing last_activation.")
        return self._cache['a']

    @property
    def theta(self):
        return self._theta_history[-1]
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def initialize_parameters(self, n_features):
        # Initialize theta (weights + bias as the first element)
        theta = np.random.randn(n_features + 1)
        self._theta_history.append(theta)

    def set_weights(self, weights):
        """
        Sets the neuron's weights and bias.
        
        Parameters:
        - weights: A numpy array containing [bias, weight1, weight2, ..., weightN]
        """
        self._theta_history = [weights.copy()]
        self._cache = None  # Reset cache when weights are set

    def forward(self, x):
        """Forward pass for a single sample."""
        # Add a bias term (x_0 = 1) for each x sample
        x_with_bias = np.insert(x, 0, 1)
        z = np.dot(self.theta, x_with_bias)
        a = self.sigmoid(z)
        # Store the cache for backpropagation
        self._cache = {'x_with_bias': x_with_bias, 'z': z, 'a': a}
        return a

    def backward(self, da):
        """Backward pass computes the gradient of the loss with respect to theta."""
        cache = self._cache
        if cache is None:
            raise ValueError("No cache available. Ensure that forward() has been called before backward().")
        x_with_bias = cache['x_with_bias']
        a = cache['a']
        dz = da * a * (1 - a)  # Derivative of sigmoid activation
        gradient = dz * x_with_bias
        # Update parameters (theta)
        theta = self.theta.copy()
        theta -= self._learning_rate * gradient
        self._theta_history.append(theta)
        # Clear the cache after backward pass
        self._cache = None
        return gradient

    def update_backward(self):
        """Revert the last update (for stepping backward)."""
        if len(self._theta_history) > 1:
            self._theta_history.pop()
        # No need to manage cache here since it's cleared after backward()
        return self._theta_history[-1]
    
    def activations(self, X):
        """Compute activations (outputs) for all input samples X."""
        # Add bias term to each sample
        X_with_bias = np.insert(X, 0, 1, axis=1)
        z = np.dot(X_with_bias, self.theta)
        a = self.sigmoid(z)
        return a

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, learning_rate=0.01):
        """
        Initializes a multi-layer neural network.

        Parameters:
        - input_size: Number of features in the input data.
        - hidden_sizes: List containing the number of neurons in each hidden layer.
        - learning_rate: Learning rate for weight updates.
        """
        self.learning_rate = learning_rate
        self.layers = []  # List to hold each layer's neurons
        self.step = 0     # To keep track of training steps

        # Initialize the layers
        previous_size = input_size
        for layer_size in hidden_sizes:
            layer = [Neuron(learning_rate) for _ in range(layer_size)]
            for neuron in layer:
                neuron.initialize_parameters(previous_size)
            self.layers.append(layer)
            previous_size = layer_size

        # Initialize the output neuron
        self.output_neuron = Neuron(learning_rate)
        self.output_neuron.initialize_parameters(previous_size)


    def forward(self, x):
        """
        Performs a forward pass through the network for a single sample.

        Parameters:
        - x: Input feature vector (numpy array).

        Returns:
        - output: The activation from the output neuron.
        """
        activations = x
        for layer in self.layers:
            next_activations = []
            for neuron in layer:
                a = neuron.forward(activations)
                next_activations.append(a)
            activations = np.array(next_activations)
        output = self.output_neuron.forward(activations)
        return output


    def backward(self, y_true):
        """
        Performs a backward pass through the network, updating weights.

        Parameters:
        - y_true: True label for the input sample.
        """
        # Compute loss derivative at output neuron
        a_output = self.output_neuron.last_activation
        da_output = a_output - y_true  # Derivative of binary cross-entropy loss

        # Backward pass through output neuron
        self.output_neuron.backward(da_output)

        # Initialize da_current for the last hidden layer
        da_current = []
        w_output = self.output_neuron.theta[1:]  # Exclude bias term

        # Compute da for each neuron in the last hidden layer
        for i, neuron in enumerate(self.layers[-1]):
            da = w_output[i] * da_output
            da_current.append(da)

        # Backpropagate through hidden layers
        for l in reversed(range(len(self.layers) - 1)):
            layer = self.layers[l]
            next_layer = self.layers[l + 1]
            da_next = []
            # Prepare weights from the next layer
            w_next_layer = np.array([neuron.theta[1:] for neuron in next_layer])
            for i, neuron in enumerate(layer):
                # Sum over all neurons in the next layer
                da = np.dot(w_next_layer[:, i], da_current)
                neuron.backward(da)
                da_next.append(da)
            da_current = da_next
            

    def step_forward(self, x, y_true):
        """
        Advances the training by one SGD update (forward and backward pass).

        Parameters:
        - x: Input feature vector (numpy array).
        - y_true: True label for the input sample.
        """
        self.forward(x)
        self.backward(y_true)
        self.step += 1

    def update_backward(self):
        """
        Reverts the network to the previous state by undoing the last update.
        """
        # Revert output neuron
        self.output_neuron.update_backward()
        # Revert hidden layers
        for layer in self.layers:
            for neuron in layer:
                neuron.update_backward()
        if self.step > 0:
            self.step -= 1

    def activations(self, X):
        """
        Computes activations for all input samples at each layer.

        Parameters:
        - X: Input data matrix (numpy array).

        Returns:
        - activations_per_layer: List of activations for each layer.
        """
        activations = X
        activations_per_layer = [activations]
        for layer in self.layers:
            next_activations = []
            for neuron in layer:
                a = neuron.activations(activations)
                next_activations.append(a)
            activations = np.column_stack(next_activations)
            activations_per_layer.append(activations)
        # Output neuron activations
        output_activations = self.output_neuron.activations(activations)
        activations_per_layer.append(output_activations)
        return activations_per_layer

def plot_neural_network_state(X, y, nn, step, sample_index=None):
    """
    Visualizes the state of the neural network at a given training step.

    Parameters:
    - X: Input features (numpy array of shape (n_samples, n_features))
    - y: True labels (numpy array)
    - nn: NeuralNetwork instance
    - step: Current training step
    - sample_index: Index of the current training sample (optional)

    Returns:
    -1 for step backward
     0 to exit
     1 for step forward
    """
    global return_val, plot_initialized

    return_val = None

    # Compute activations at each layer
    activations_per_layer = nn.activations(X)

    # Extract activations for each layer
    input_space = X  # Original input space
    hidden_layer1_activations = activations_per_layer[1]  # First hidden layer activations
    hidden_layer2_activations = activations_per_layer[2]  # Second hidden layer activations
    output_activations = activations_per_layer[-1]  # Output layer activations

    # Predictions (thresholded at 0.5)
    predictions = (output_activations >= 0.5).astype(int).flatten()

    # Set up the plot
    if not plot_initialized:
        plt.ion()  # Interactive mode on
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.canvas.mpl_connect('key_press_event', on_key)
        fig.canvas.mpl_connect('button_press_event', on_click)
        plot_initialized = True
    else:
        fig = plt.gcf()
        axes = np.array(fig.axes).reshape(2, 3)

    # Clear previous plots
    for ax_row in axes:
        for ax in ax_row:
            ax.cla()

    color_map = ListedColormap(['blue', 'red'])
    markers = ['o', 's']

    # First Column - Hidden Layer 1 Neurons
    # Plot for Hidden Layer 1 Neuron 1
    ax = axes[0, 0]
    plot_decision_boundary_for_single_neuron(X, y, nn.layers[0][0], ax, title='Hidden Layer 1 Neuron 1')
    if sample_index is not None:
        ax.scatter(X[sample_index, 0], X[sample_index, 1],
                   facecolors='none', edgecolors='yellow', s=100, linewidths=2)

    # Plot for Hidden Layer 1 Neuron 2
    ax = axes[1, 0]
    plot_decision_boundary_for_single_neuron(X, y, nn.layers[0][1], ax, title='Hidden Layer 1 Neuron 2')
    if sample_index is not None:
        ax.scatter(X[sample_index, 0], X[sample_index, 1],
                   facecolors='none', edgecolors='yellow', s=100, linewidths=2)

    # Second Column - Hidden Layer 2 Neurons
    # Plot for Hidden Layer 2 Neuron 1
    ax = axes[0, 1]
    plot_decision_boundary_for_hidden_layer2_neuron(hidden_layer1_activations, y, nn.layers[1][0], ax,
                                                   title='Hidden Layer 2 Neuron 1')
    if sample_index is not None:
        ax.scatter(hidden_layer1_activations[sample_index, 0], hidden_layer1_activations[sample_index, 1],
                   facecolors='none', edgecolors='yellow', s=100, linewidths=2)

    # Plot for Hidden Layer 2 Neuron 2
    ax = axes[1, 1]
    plot_decision_boundary_for_hidden_layer2_neuron(hidden_layer1_activations, y, nn.layers[1][1], ax,
                                                   title='Hidden Layer 2 Neuron 2')
    if sample_index is not None:
        ax.scatter(hidden_layer1_activations[sample_index, 0], hidden_layer1_activations[sample_index, 1],
                   facecolors='none', edgecolors='yellow', s=100, linewidths=2)

    # Third Column - Output Neuron and Predictions
    # Output Neuron Decision Boundary
    ax = axes[0, 2]
    plot_decision_boundary_for_output_neuron(hidden_layer2_activations, y, nn.output_neuron, ax)
    if sample_index is not None:
        ax.scatter(hidden_layer2_activations[sample_index, 0], hidden_layer2_activations[sample_index, 1],
                   facecolors='none', edgecolors='yellow', s=100, linewidths=2)

    # Predictions in Input Space
    ax = axes[1, 2]
    print(f"predictions[:10]={predictions[:10]}")
    #color_map = ListedColormap(['blue', 'red'])
    #color_map = ListedColormap(['red', 'blue'])   # this shouldn't be necessary. HEREXXX HEREPOOP
    ax.scatter(X[:, 0], X[:, 1], c=predictions, cmap=color_map, edgecolor='k')
    ax.set_title('Input Space with Predictions')
    if sample_index is not None:
        ax.scatter(X[sample_index, 0], X[sample_index, 1],
                   facecolors='none', edgecolors='yellow', s=100, linewidths=2)
        
    # Highlight misclassified points
    misclassified = predictions != y
    if any(misclassified):
        ax.scatter(X[misclassified, 0], X[misclassified, 1], facecolors='none', edgecolors='red', s=100, linewidths=2)

    # Adjust layout and titles
    plt.suptitle(f'Neural Network State at Step {step}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Draw the updated plot
    plt.draw()
    plt.pause(0.1)

    # Wait for user interaction
    while not plt.waitforbuttonpress() and return_val is None:
        plt.pause(0.1)

    return return_val

def plot_decision_boundary_for_single_neuron(X, y, neuron, ax, title='Neuron Decision Boundary'):
    """
    Plots the decision boundary of a single neuron in the input space.

    Parameters:
    - X: Input features
    - y: True labels
    - neuron: Neuron instance
    - ax: Matplotlib axis to plot on
    - title: Title of the plot
    """
    color_map = ListedColormap(['blue', 'red'])
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=color_map, edgecolor='k')
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = neuron.activations(grid)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0.5], linestyles=['--'], colors=['k'])
    ax.set_title(title)
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')

def plot_decision_boundary_for_hidden_layer2_neuron(activations, y, neuron, ax, title='Neuron Decision Boundary'):
    """
    Plots the decision boundary of a neuron in the second hidden layer in the activation space of the first hidden layer.

    Parameters:
    - activations: Activations from the previous layer (inputs to the neuron)
    - y: True labels
    - neuron: Neuron instance
    - ax: Matplotlib axis to plot on
    - title: Title of the plot
    """
    color_map = ListedColormap(['blue', 'red'])
    ax.scatter(activations[:, 0], activations[:, 1], c=y, cmap=color_map, edgecolor='k')
    a_min, a_max = activations.min(axis=0) - 0.1, activations.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(a_min[0], a_max[0], 200),
                         np.linspace(a_min[1], a_max[1], 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = neuron.activations(grid)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0.5], linestyles=['--'], colors=['k'])
    ax.set_title(title)
    ax.set_xlabel('Activation of Hidden Layer 1 Neuron 1')
    ax.set_ylabel('Activation of Hidden Layer 1 Neuron 2')

def plot_decision_boundary_for_output_neuron(activations, y, neuron, ax):
    """
    Plots the decision boundary of the output neuron in the activation space.

    Parameters:
    - activations: Activations from the previous layer (inputs to the output neuron)
    - y: True labels
    - neuron: Output neuron instance
    - ax: Matplotlib axis to plot on
    """
    color_map = ListedColormap(['blue', 'red'])
    ax.scatter(activations[:, 0], activations[:, 1], c=y, cmap=color_map, edgecolor='k')
    a_min, a_max = activations.min(axis=0) - 0.1, activations.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(a_min[0], a_max[0], 200),
                         np.linspace(a_min[1], a_max[1], 200))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = neuron.activations(grid)
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=20, cmap='viridis', alpha=0.5)
    ax.contour(xx, yy, Z, levels=[0.5], linestyles=['--'], colors=['k'])
    ax.set_title('Output Neuron Decision Boundary')
    ax.set_xlabel('Activation of Hidden Layer 2 Neuron 1')
    ax.set_ylabel('Activation of Hidden Layer 2 Neuron 2')

def initialize_weights_for_xor(nn, nudge=0.0, init_scale=1.0, seed=None):
    """
    Initializes the neural network weights to approximate the XOR function with optional perturbation and scaling.
    
    Parameters:
    - nn: NeuralNetwork instance.
    - nudge: Float indicating the magnitude of perturbation to apply to each weight.
             A value of 0.0 means no perturbation.
    - init_scale: Float indicating the scaling factor for the initial weights.
                  A value <1.0 starts with smaller weights, promoting gradual convergence.
    - seed: Integer seed for the random number generator.
    """
    # Define the base weights for XOR
    base_weights = {
        'hidden_layer_1_neuron_1': np.array([-5.0, 10.0, 0.0]),    # [bias, weight_x1, weight_x2]
        'hidden_layer_1_neuron_2': np.array([-5.0, 0.0, 10.0]),
        'hidden_layer_2_neuron_1': np.array([-5.0, 10.0, -10.0]),
        'hidden_layer_2_neuron_2': np.array([-5.0, -10.0, 10.0]),
        'output_neuron':          np.array([-5.0, 10.0, 10.0])
    }

    # Apply nudge if specified
    if nudge != 0.0:
        rng = np.random.default_rng(seed)  # Use seed for reproducibility
        for key in base_weights:
            perturbation = rng.uniform(-nudge, nudge, size=base_weights[key].shape)
            base_weights[key] += perturbation

    # Apply scaling to initial weights
    if init_scale != 1.0:
        for key in base_weights:
            base_weights[key] *= init_scale

    # Set weights using the setter method for encapsulation
    nn.layers[0][0].set_weights(base_weights['hidden_layer_1_neuron_1'])
    nn.layers[0][1].set_weights(base_weights['hidden_layer_1_neuron_2'])
    nn.layers[1][0].set_weights(base_weights['hidden_layer_2_neuron_1'])
    nn.layers[1][1].set_weights(base_weights['hidden_layer_2_neuron_2'])
    nn.output_neuron.set_weights(base_weights['output_neuron'])
        
def main():
    # Create the parser
    parser = argparse.ArgumentParser(description="Neural Network for XOR problem with visualization.")
    parser.add_argument('file', type=str, help="Data file (CSV format with features and labels).")
    parser.add_argument('--seed', type=int, default=83, help="Random seed for reproducibility.")
    parser.add_argument('--alpha', type=float, default=0.1, help="Learning rate")
    parser.add_argument('--answer', action='store_true', help="Assume initial state equal to the XOR solution")
    parser.add_argument('--nudge', type=float, default=0.0,
                        help="Magnitude of perturbation to apply to initial weights.")
    parser.add_argument('--init-scale', type=float, default=1.0,
                        help="Scaling factor for initial parameters. Values <1.0 start with smaller weights.")

    args = parser.parse_args()

    # Validate nudge factor
    if args.nudge < 0.0:
        parser.error("--nudge must be non-negative.")
    elif args.nudge > 2.0:
        parser.error("--nudge should not exceed 2.0 for stability.")
    
    # Set the seed
    if args.seed is not None:
        np.random.seed(args.seed)

    # Load data
    data = pd.read_csv(args.file)
    X = data.iloc[:, :-1].values  # Features
    y = data.iloc[:, -1].values   # Labels

    # Initialize the neural network with two hidden layers of two neurons each
    learning_rate = args.alpha
    hidden_sizes = [2, 2]  # Two hidden layers with two neurons each
    nn = NeuralNetwork(input_size=X.shape[1], hidden_sizes=hidden_sizes, learning_rate=learning_rate)

    if args.answer:
        initialize_weights_for_xor(nn, nudge=args.nudge, init_scale=args.init_scale, seed=args.seed)
        print(f"Network weights initialized to solve XOR with nudge={args.nudge}.")

    step = 0
    while True:
        i = step % len(y)
        x_sample = X[i]
        y_sample = y[i]

        # visualize before each SGD step highlighting the point about to be used.
        action = plot_neural_network_state(X, y, nn, step=step, sample_index=i)

        # Step forward (train on one sample)
        nn.step_forward(x_sample, y_sample)

        # Handle user input for stepping forward or backward
        if action == 0:
            exit(0)
        elif action == -1:
            if step > 0:
                nn.update_backward()
                step -= 1
        else:
            step += 1

if __name__ == "__main__":
    main()
