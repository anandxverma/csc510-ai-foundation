"""
ann.py - A simple 2-layer Artificial Neural Network (ANN) using only NumPy.

Task
----
Given 3 consecutive numbers from an arithmetic series, predict the 4th.
For example, the input ``5 10 15`` should produce a prediction close to ``20``.

Model
-----
Architecture : 3 inputs -> 6 hidden neurons (sigmoid) -> 1 output (linear)
Loss         : Mean Squared Error (MSE)
Training     : full-batch gradient descent with backpropagation

How it works
------------
1. Generate random arithmetic sequences as training data
   (start value in [1, 100], step in [1, 10]).
2. Normalize every value with ``(x - SHIFT) / SCALE`` so the inputs sit
   roughly in [-1.5, 1.8] and the sigmoid units do not saturate.
3. Train the network for ``EPOCHS`` iterations, printing the loss
   every 500 epochs.
4. Start an interactive prompt: the user enters 3 numbers, the network
   predicts the 4th, and (if matplotlib is installed) shows the training
   loss curve and the input series with its prediction.

Because the training sequences contain values in [1, 130], inputs outside
that range are extrapolations and may be less accurate.

Dependencies
------------
- numpy (required)
- matplotlib (optional; used only for plots)

Usage
-----
    python ann.py

Type ``q``, ``quit`` or ``exit`` at the prompt to leave.
"""

import numpy as np

try:
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False

# ----------------------------------------------------------------------
# STEP 1: Settings
# ----------------------------------------------------------------------
SHIFT = 60.0         # centre the data around 0 ...
SCALE = 40.0         # ... and shrink it so sigmoid doesn't saturate
N_INPUT = 3          # numbers given to the network
N_HIDDEN = 6         # neurons in the hidden layer
N_OUTPUT = 1         # numbers predicted by the network
LEARNING_RATE = 0.3  # gradient descent step size
EPOCHS = 5000        # well above the required 1,000
np.random.seed(42)   # reproducible data and weights


# ----------------------------------------------------------------------
# STEP 2: Training data (auto-generated arithmetic sequences)
# ----------------------------------------------------------------------
def make_data(n_samples=500):
    """Generate normalized training data from random arithmetic sequences.

    Each sample is a 4-term sequence ``start, start+step, start+2*step,
    start+3*step`` where ``start`` is in [1, 100] and ``step`` is in [1, 10].
    The first 3 terms are the input and the 4th term is the target. All
    values are normalized with ``(value - SHIFT) / SCALE``.

    Args:
        n_samples (int): Number of sequences to generate. Defaults to 500.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            X: Normalized inputs, shape ``(n_samples, 3)``.
            y: Normalized targets, shape ``(n_samples, 1)``.
    """
    start = np.random.randint(1, 101, size=(n_samples, 1))
    step = np.random.randint(1, 11, size=(n_samples, 1))
    seq = start + step * np.arange(4)          # shape (n_samples, 4)
    X = (seq[:, :3] - SHIFT) / SCALE
    y = (seq[:, 3:4] - SHIFT) / SCALE
    return X, y


# ----------------------------------------------------------------------
# STEP 3: Activation function and its derivative
# ----------------------------------------------------------------------
def sigmoid(z):
    """Apply the logistic sigmoid function element-wise.

    ``sigmoid(z) = 1 / (1 + e^-z)`` maps any real number into (0, 1).
    Its derivative, ``sigmoid(z) * (1 - sigmoid(z))``, is used in
    :func:`backward` as ``a1 * (1 - a1)``.

    Args:
        z (np.ndarray): Pre-activation values of any shape.

    Returns:
        np.ndarray: Activations with the same shape as ``z``, in (0, 1).
    """
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------
# STEP 4: Initialize weights (random) and biases (zeros)
# ----------------------------------------------------------------------
def init_params():
    """Create the network's initial weights and biases.

    Weights are drawn from a normal distribution scaled by 0.5 so the
    network does not start with all neurons identical (symmetry breaking).
    Biases start at zero.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            W1: Input-to-hidden weights, shape ``(N_INPUT, N_HIDDEN)``.
            b1: Hidden-layer biases, shape ``(1, N_HIDDEN)``.
            W2: Hidden-to-output weights, shape ``(N_HIDDEN, N_OUTPUT)``.
            b2: Output-layer bias, shape ``(1, N_OUTPUT)``.
    """
    W1 = np.random.randn(N_INPUT, N_HIDDEN) * 0.5
    b1 = np.zeros((1, N_HIDDEN))
    W2 = np.random.randn(N_HIDDEN, N_OUTPUT) * 0.5
    b2 = np.zeros((1, N_OUTPUT))
    return W1, b1, W2, b2


# ----------------------------------------------------------------------
# STEP 5: Feedforward
#   input -> (matrix multiply by weights) -> activation -> output
# ----------------------------------------------------------------------
def forward(X, W1, b1, W2, b2):
    """Run a forward pass through the network.

    Computes ``a1 = sigmoid(X @ W1 + b1)`` for the hidden layer and
    ``y_hat = a1 @ W2 + b2`` for the linear output layer.

    Args:
        X (np.ndarray): Normalized inputs, shape ``(m, N_INPUT)``.
        W1 (np.ndarray): Input-to-hidden weights, shape ``(N_INPUT, N_HIDDEN)``.
        b1 (np.ndarray): Hidden-layer biases, shape ``(1, N_HIDDEN)``.
        W2 (np.ndarray): Hidden-to-output weights, shape ``(N_HIDDEN, N_OUTPUT)``.
        b2 (np.ndarray): Output-layer bias, shape ``(1, N_OUTPUT)``.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            a1: Hidden-layer activations, shape ``(m, N_HIDDEN)``.
                Backpropagation needs these.
            y_hat: Normalized predictions, shape ``(m, N_OUTPUT)``.
    """
    z1 = X @ W1 + b1             # input times weights
    a1 = sigmoid(z1)             # hidden layer activation
    y_hat = a1 @ W2 + b2         # linear output layer
    return a1, y_hat


# ----------------------------------------------------------------------
# STEP 6: Loss function (MSE)
# ----------------------------------------------------------------------
def mse(y_hat, y):
    """Compute the Mean Squared Error between predictions and targets.

    Args:
        y_hat (np.ndarray): Predicted values, shape ``(m, N_OUTPUT)``.
        y (np.ndarray): True target values, same shape as ``y_hat``.

    Returns:
        float: Mean of ``(y_hat - y) ** 2`` over all samples.
    """
    return np.mean((y_hat - y) ** 2)


# ----------------------------------------------------------------------
# STEP 7: Backpropagation (chain rule, working backward from the error)
# ----------------------------------------------------------------------
def backward(X, y, a1, y_hat, W2):
    """Compute the gradients of the loss with respect to every parameter.

    Applies the chain rule from the output error back to the input layer:

    - Output layer (linear): ``dW2 = a1^T @ error / m``, ``db2 = mean(error)``
    - Hidden layer (sigmoid): ``d_hidden = (error @ W2^T) * a1 * (1 - a1)``,
      then ``dW1 = X^T @ d_hidden / m``, ``db1 = mean(d_hidden)``

    ``error = y_hat - y`` is the MSE gradient without its constant factor
    of 2. That factor is effectively folded into ``LEARNING_RATE``.

    Args:
        X (np.ndarray): Normalized inputs, shape ``(m, N_INPUT)``.
        y (np.ndarray): Normalized targets, shape ``(m, N_OUTPUT)``.
        a1 (np.ndarray): Hidden activations from :func:`forward`,
            shape ``(m, N_HIDDEN)``.
        y_hat (np.ndarray): Predictions from :func:`forward`,
            shape ``(m, N_OUTPUT)``.
        W2 (np.ndarray): Current hidden-to-output weights,
            shape ``(N_HIDDEN, N_OUTPUT)``.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            Gradients ``(dW1, db1, dW2, db2)``, each matching the shape of
            its parameter.
    """
    m = len(X)
    error = y_hat - y                              # predicted - desired

    dW2 = a1.T @ error / m                         # output-layer weights
    db2 = error.mean(axis=0, keepdims=True)

    d_hidden = (error @ W2.T) * a1 * (1 - a1)      # through sigmoid derivative
    dW1 = X.T @ d_hidden / m                       # hidden-layer weights
    db1 = d_hidden.mean(axis=0, keepdims=True)
    return dW1, db1, dW2, db2


# ----------------------------------------------------------------------
# STEP 8: Training loop (gradient descent)
# ----------------------------------------------------------------------
def train(X, y):
    """Train the network with full-batch gradient descent.

    Each epoch runs a forward pass on the whole dataset, records the MSE
    loss, computes gradients with :func:`backward`, and updates every
    parameter in place with ``param -= LEARNING_RATE * grad``. The loss is
    printed on the first epoch and every 500 epochs after that.

    Args:
        X (np.ndarray): Normalized training inputs, shape ``(m, N_INPUT)``.
        y (np.ndarray): Normalized training targets, shape ``(m, N_OUTPUT)``.

    Returns:
        tuple[tuple, list[float]]:
            params: Trained ``(W1, b1, W2, b2)``.
            losses: MSE loss for each epoch, length ``EPOCHS``.
    """
    W1, b1, W2, b2 = init_params()
    losses = []
    for epoch in range(1, EPOCHS + 1):
        a1, y_hat = forward(X, W1, b1, W2, b2)
        loss = mse(y_hat, y)
        losses.append(loss)

        dW1, db1, dW2, db2 = backward(X, y, a1, y_hat, W2)

        W1 -= LEARNING_RATE * dW1                  # apply gradients
        b1 -= LEARNING_RATE * db1
        W2 -= LEARNING_RATE * dW2
        b2 -= LEARNING_RATE * db2

        if epoch == 1 or epoch % 500 == 0:
            print(f"  Epoch {epoch:5d} | loss = {loss:.8f}")
    return (W1, b1, W2, b2), losses


# ----------------------------------------------------------------------
# STEP 9: Prediction + visuals
# ----------------------------------------------------------------------
def predict(numbers, params):
    """Predict the next number in a series from raw (unnormalized) values.

    The inputs are normalized the same way as the training data, passed
    through the network, and the output is converted back to the
    original scale.

    Args:
        numbers (Sequence[float]): ``N_INPUT`` consecutive numbers from a
            series, e.g. ``[5, 10, 15]``.
        params (tuple): Trained ``(W1, b1, W2, b2)`` from :func:`train`.

    Returns:
        float: The predicted next number in the original scale.
    """
    X = (np.array(numbers, dtype=float).reshape(1, -1) - SHIFT) / SCALE  # scale down
    _, y_hat = forward(X, *params)
    return float(y_hat[0, 0] * SCALE + SHIFT)                            # scale back up


def show_plots(losses, numbers, prediction):
    """Show the training loss curve and the user's series with its prediction.

    Opens a window with two side-by-side charts:

    - Left: training MSE per epoch on a log scale.
    - Right: the input numbers by position, with the predicted next value
      highlighted in red.

    If matplotlib is not installed, prints a notice and returns without
    plotting. ``plt.show()`` blocks until the window is closed.

    Args:
        losses (list[float]): Loss per epoch, as returned by :func:`train`.
        numbers (Sequence[float]): The user's input numbers.
        prediction (float): The predicted next number.
    """
    if not HAS_PLOT:
        print("  (matplotlib not installed - skipping plots)")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(losses)
    ax1.set_yscale("log")
    ax1.set_title("Training loss (MSE)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")

    idx = list(range(1, len(numbers) + 1))
    ax2.plot(idx, numbers, "o-", label="Your input")
    ax2.plot(len(numbers) + 1, prediction, "ro", markersize=10, label="Predicted next")
    ax2.set_title("Input series and prediction")
    ax2.set_xlabel("Position in series")
    ax2.set_xticks(idx + [len(numbers) + 1])
    ax2.legend()

    plt.tight_layout()
    plt.show()


def main():
    """Train the network, then run an interactive prediction loop.

    After training, repeatedly asks the user for ``N_INPUT`` numbers
    (separated by spaces or commas), prints the predicted next number,
    warns if any input is outside the training range [0, 130], and shows
    the plots. Invalid input prints an error and asks again. Entering
    ``q``, ``quit`` or ``exit`` ends the program.
    """
    print("Training a 2-layer neural network on arithmetic sequences...")
    X, y = make_data()
    params, losses = train(X, y)
    print(f"Done. Final loss: {losses[-1]:.8f}\n")

    print("Enter 3 numbers from a series (e.g. 5 10 15) and I'll predict the 4th.")
    print("Type 'q' to quit.\n")
    while True:
        text = input("Numbers: ").strip()
        if text.lower() in ("q", "quit", "exit"):
            break
        try:
            numbers = [float(t) for t in text.replace(",", " ").split()]
        except ValueError:
            print("  Please enter numbers only, separated by spaces.\n")
            continue
        if len(numbers) != N_INPUT:
            print(f"  Please enter exactly {N_INPUT} numbers.\n")
            continue

        prediction = predict(numbers, params)
        print(f"  Predicted next number: {prediction:.2f}\n")
        if max(numbers) > 130 or min(numbers) < 0:
            print("  Note: these values are outside the training range, "
                  "so the prediction may be less accurate.\n")
        show_plots(losses, numbers, prediction)


if __name__ == "__main__":
    main()
