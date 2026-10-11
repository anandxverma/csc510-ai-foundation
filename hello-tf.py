import tensorflow as tf
import numpy as np
from tensorflow import keras

# This is a simple example of a neural network using TensorFlow and Keras.
# It will train the model to learn the relationship between x and y variable based on the provided data.
# The expected relationship is y = 3x + 1.

# Create the simplest possible neural network. It has one layer, that layer has one neuron, and the input shape to it is only one value.
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

# Compile the neural network by specifying two functions—a loss and an optimizer.
# We will use mean_squared_error for the loss and stochastic gradient descent (sgd) for the optimizer
# # No need to understand the math for these for the time being
model.compile(optimizer='sgd', loss='mean_squared_error')

# Provide the input (xs) and output (ys) data for training the neural network.
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

# Train the model using the provided data.
model.fit(xs, ys, epochs=500)

# After training completes, let's predict for specific values
print(model.predict(np.array([10.0], dtype=float)))