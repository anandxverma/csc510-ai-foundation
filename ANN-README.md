# 2-Layer Neural Network in NumPy: Next-Number Predictor

**Author:** [Your Name]  **Course:** [Course Name]  **Script:** `ann.py`

A basic artificial neural network, built from scratch with NumPy only, that learns to predict the next number in a series. You give it 3 numbers (for example `5 10 15`) and it predicts the 4th (`20`).

---

## Quick Start

**1. Install the requirements** (Python 3.8+):

```
pip install numpy matplotlib
```

`numpy` is required. `matplotlib` is optional and only needed for the charts. Without it, the program still runs and prints predictions.

**2. Run the program:**

```
python ann.py
```

**3. Use it:**

1. The network trains first (a couple of seconds) and prints its loss as it learns.
2. When you see `Numbers:`, type 3 numbers separated by spaces and press Enter.
3. The predicted 4th number is printed, and a chart window opens. Close the chart to enter another series.
4. Type `q` to quit.

### Example session

```
Training a 2-layer neural network on arithmetic sequences...
  Epoch     1 | loss = 1.81037668
  Epoch   500 | loss = 0.01053942
  ...
  Epoch  5000 | loss = 0.00016756
Done. Final loss: 0.00016756

Numbers: 5 10 15
  Predicted next number: 20.37

Numbers: 7 14 21
  Predicted next number: 27.97
```

### What the chart shows

- **Left:** the training loss (error) over all epochs, falling as the network learns.
- **Right:** your 3 input numbers as a line, with the predicted 4th number as a red dot.

### Input rules

- Enter exactly 3 numbers, separated by spaces or commas.
- Anything else (letters, too few or too many numbers) gets a clear error message and a prompt to try again.
- Series work best with values between about 0 and 130, which is the range the network was trained on.

---

## Assignment Requirements and How They Are Met

### Network features

| Requirement | Where it is met in `ann.py` |
|---|---|
| An input layer that takes data as a matrix and passes it on | `make_data()` builds the input matrix `X` (500 rows × 3 columns), which is passed into `forward()` |
| A hidden layer | 6 neurons with a sigmoid activation (`N_HIDDEN = 6`) |
| An output layer | 1 neuron with a linear output, returning the predicted number |
| Weights between the layers | `W1` (3×6) and `W2` (6×1), plus biases `b1` and `b2`, created in `init_params()` |

### Functions the network must demonstrate

| Requirement | Where it is met |
|---|---|
| Multiply the input by a set of weights (matrix multiplication) | `forward()`: `X @ W1 + b1` and `a1 @ W2 + b2` |
| Apply a deliberate activation function to the hidden layer | `sigmoid()`, applied in `forward()` |
| Return an output | `forward()` returns the predicted value `y_hat` |
| Calculate error as the difference between desired and predicted output | `backward()`: `error = y_hat - y` |
| Loss function | Mean squared error, `mse()` |
| Apply the loss gradient to the weights | `backward()` computes the gradients using the chain rule, and `train()` updates the weights with `W -= LEARNING_RATE * dW` |
| Repeat at least 1,000 times | `EPOCHS = 5000` |

### Submission requirements

| Requirement | How it is met |
|---|---|
| A `.py` script that is easy to activate | Run with `python ann.py` |
| Accepts simple inputs in a clear manner | Prompts for 3 numbers, with input validation |
| Predicts and visualizes the final variable | Prints the prediction and plots it with the input series |
| NumPy | The network uses only NumPy. Matplotlib is used only for the charts |

---

## How the Program Works

### The task

A series such as `5, 8, 11, 14` is split into an input (`5, 8, 11`) and the correct answer (`14`). The network is never told the rule "add the gap to the last number." It learns the pattern from 500 auto-generated examples of arithmetic sequences (start between 1 and 100, step between 1 and 10).

### Architecture

```
Input (3)  →  Hidden layer (6 neurons, sigmoid)  →  Output (1, linear)
```

### One epoch of training

An epoch is one full pass through all the training data. Each epoch repeats four steps:

1. **Feedforward.** Multiply the inputs by the weights, apply sigmoid in the hidden layer, and produce the predictions:
   `z1 = X·W1 + b1`, `a1 = sigmoid(z1)`, `ŷ = a1·W2 + b2`
2. **Loss.** Measure the error with mean squared error: `mean((ŷ − y)²)`
3. **Backpropagation.** Work backward from the error to find how much each weight contributed to it, using the chain rule. The sigmoid derivative is `a1 · (1 − a1)`.
4. **Gradient descent.** Nudge every weight and bias a small step opposite to its gradient: `W = W − learning_rate × dW`

This repeats 5,000 times, and the loss falls as the weights improve.

### Data scaling

Inputs and targets are centered and shrunk before training: `(value − 60) / 40`. This keeps values in the steep middle part of the sigmoid curve. Without it, learning is slow or unstable. Predictions are converted back to normal numbers before being shown.

---

## Settings You Can Change

These are at the top of `ann.py`:

| Setting | Default | Effect |
|---|---|---|
| `EPOCHS` | 5000 | Number of training passes. Fewer gives rougher predictions |
| `LEARNING_RATE` | 0.3 | Size of each weight update. Too small learns slowly, and too large (about 2 or more) makes training diverge |
| `N_HIDDEN` | 6 | Number of hidden neurons |

A random seed (`np.random.seed(42)`) is set, so results are the same each run.

---

## Results and Observations

| Input | Predicted | Correct |
|---|---|---|
| `5 10 15` | 20.37 | 20 |
| `7 14 21` | 27.97 | 28 |
| `200 300 400` | about 206 | 500 |

- **Inside the training range** (roughly 0 to 130), predictions land within a couple of units of the correct answer.
- **Outside the training range**, accuracy drops sharply. Sigmoid neurons flatten out for large inputs, so the network cannot extrapolate beyond what it has seen. The program prints a warning when inputs fall outside the range.
- **Scaling mattered.** An early version that only divided by a constant learned slowly and was off by about 3 units. Centering the data fixed this.

---

## Limitations

- The network learns **arithmetic** sequences (constant gap). It has not been trained on multiplicative or other patterns.
- Predictions are approximate, not exact.
- It only accepts exactly 3 numbers.

---

## Files

| File | Purpose |
|---|---|
| `ann.py` | The neural network program |
| `README.md` | This file |
