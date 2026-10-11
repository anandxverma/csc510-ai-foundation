# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# This program demonstrates a simple image recognition task using the Fashion MNIST dataset with TensorFlow and Keras.
# The goal is to be able to recognize different types of clothing items in images.
# The dataset contains 60,000 training images and 10,000 test images, each of size 28x28 pixels.
# Each pixel value is an integer between 0 and 255, representing the grayscale intensity of the pixel.

print(tf.__version__)

# Load the dataset
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Create a list of class names corresponding to the labels in the dataset.
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print("Training images shape:", train_images.shape)
print("Number of training labels:", len(train_labels))
print("Test images shape:", test_images.shape)
print("Number of test labels:", len(test_labels))

# Plot the first training image to visualize it
"""
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
"""

# Normalize the pixel values of the images to be between 0 and 1. This helps the neural network train more effectively.
train_images = train_images / 255.0
test_images = test_images / 255.0

# This is just for visualizing
# Display a grid of the first 25 training images after normalization to verify the preprocessing step.
"""
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
"""

# Create the neural network model with layers for flattening the input, a dense hidden layer, and an output layer for classification.
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# Compile the model with the Adam optimizer, sparse categorical crossentropy loss, and accuracy metric.
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model on the training data for 10 epochs.
model.fit(train_images, train_labels, epochs=10)

# Evaluate the model on the test data to determine its accuracy.
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)
print(predictions[0])