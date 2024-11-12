import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set random seed for reproducibility :)
RANDOM_SEED = 42

# Paths to dataset and model save locations
dataset = 'model/keypoint_classifier/keypoint.csv'
model_save_path = 'model/keypoint_classifier/keypoint_classifier.hdf5'
tflite_save_path = 'model/keypoint_classifier/keypoint_classifier.tflite'

# Number of classes in the dataset (e.g., number of gestures)
NUM_CLASSES = 5 

# Load dataset
# Features: Keypoints (21 keypoints, 2 values per keypoint - x and y)
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (21 * 2) + 1)))
# print(X_dataset)
# Labels: Corresponding gesture classes
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))
# print(y_dataset)

# Split dataset into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

# Model building :
model = tf.keras.models.Sequential([
    # Input layer, expecting 42 features (21 keypoints * 2 coordinates)
    tf.keras.layers.Input((21 * 2, )),
    # Dropout layer to prevent overfitting (20% dropout)
    tf.keras.layers.Dropout(0.2),
    # Hidden layer with 20 neurons and ReLU activation
    tf.keras.layers.Dense(20, activation='relu'),
    # Another Dropout layer (40% dropout)
    tf.keras.layers.Dropout(0.4),
    # Hidden layer with 10 neurons and ReLU activation
    tf.keras.layers.Dense(10, activation='relu'),
    # Output layer with `NUM_CLASSES` neurons and softmax activation
    tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
])

# Printing model summary (architecture details)
model.summary()

# Define callbacks:
# Save model checkpoints (save model if validation loss improves)
model_save_path = 'model/keypoint_classifier/keypoint_classifier.keras'
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False)
# Early stopping callback (stop training if no improvement for 20 epochs)
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)

# Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training the model on training data and validate on test data
model.fit(
    X_train, # Features for training
    y_train, # Labels for training
    epochs=1000, # Max epochs
    batch_size=128, # Batch size
    validation_data=(X_test, y_test), # Validation set
    callbacks=[cp_callback, es_callback] # Callbacks to save model and stop early
)

# Evaluateing the model on the test set (get loss and accuracy)
val_loss, val_acc = model.evaluate(X_test, y_test, batch_size=128)

# Load the best saved model after training
model = tf.keras.models.load_model(model_save_path)

# Test inference by predicting on the first test sample
predict_result = model.predict(np.array([X_test[0]]))

# Function to plot confusion matrix and print classification report  (No need of doing this. I did it optionally to check performance)
def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()
    
    # Optionally print a detailed classification report
    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))

# Predict on the entire test set
Y_pred = model.predict(X_test)
# Get the predicted class labels (argmax of predicted probabilities)
y_pred = np.argmax(Y_pred, axis=1)

# Save the model (without optimizer) to the specified path
model.save(model_save_path, include_optimizer=False)

# Convert to model for Tensorflow-Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Transform model (quantization)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# Convert the model
tflite_quantized_model = converter.convert()

# Save the quantized TensorFlow Lite model
open(tflite_save_path, 'wb').write(tflite_quantized_model)

# TensorFlow Lite inference:
# Loading the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
# Allocate memory for model tensors
interpreter.allocate_tensors()

# Get I / O tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Setting the input tensor to the first test sample
interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))

# Inference implementation
interpreter.invoke()
# inference results
tflite_results = interpreter.get_tensor(output_details[0]['index'])

