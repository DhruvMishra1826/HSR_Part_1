import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Set a random seed for reproducibility
RANDOM_SEED = 42

# Path to the dataset (CSV file)
dataset = 'model/point_history_classifier/point_history.csv'
model_save_path = 'model/point_history_classifier/point_history_classifier.hdf5'

# Constants for the classification task
NUM_CLASSES = 3 # Number of output classes
TIME_STEPS = 16 # Time steps for each sequence in the dataset
DIMENSION = 2 # Number of features for each time step (e.g., X, Y coordinates)

# Load the dataset: X is the feature matrix, y is the target labels
X_dataset = np.loadtxt(dataset, delimiter=',', dtype='float32', usecols=list(range(1, (TIME_STEPS * DIMENSION) + 1)))
y_dataset = np.loadtxt(dataset, delimiter=',', dtype='int32', usecols=(0))

# Split the dataset into training and testing sets (75% training, 25% testing)
X_train, X_test, y_train, y_test = train_test_split(X_dataset, y_dataset, train_size=0.75, random_state=RANDOM_SEED)

# Flag to choose between LSTM model and Dense model
use_lstm = False
model = None

# Define the model structure based on the `use_lstm` flag
if use_lstm:
    # LSTM-based model
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(TIME_STEPS * DIMENSION, )),
        tf.keras.layers.Reshape((TIME_STEPS, DIMENSION), input_shape=(TIME_STEPS * DIMENSION, )), 
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.LSTM(16, input_shape=[TIME_STEPS, DIMENSION]),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])
else:
    #  Dense model (if LSTM is not used)
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=(TIME_STEPS * DIMENSION, )),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(24, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

# Display the model architecture
model.summary()  # tf.keras.utils.plot_model(model, show_shapes=True)

# Define the path to save the trained model (keras format)
model_save_path = 'model/keypoint_classifier/point_history_classifier.keras'

# Set up the callbacks for model checkpointing and early stopping
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    model_save_path, verbose=1, save_weights_only=False) # Save the model after each epoch
es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1) # Stop training if no improvement for 20 epochs


# Compile the model with Adam optimizer and sparse categorical crossentropy loss function
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy', # Loss function for multi-class classification
    metrics=['accuracy'] # Metric to monitor during training
)

# Training the model
model.fit(
    X_train, # Training data
    y_train, # Training labels
    epochs=1000, # Number of epochs
    batch_size=128, # Batch size for training
    validation_data=(X_test, y_test), # Validation data to monitor performance during training
    callbacks=[cp_callback, es_callback] # Callbacks for checkpointing and early stopping
)

# Load the trained model
model = tf.keras.models.load_model(model_save_path)
# Perform inference on the first test sample
predict_result = model.predict(np.array([X_test[0]]))
# print(np.squeeze(predict_result))
# print(np.argmax(np.squeeze(predict_result)))

# printing confusion matrix
def print_confusion_matrix(y_true, y_pred, report=True):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
 
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(df_cmx, annot=True, fmt='g' ,square=False)
    ax.set_ylim(len(set(y_true)), 0)
    plt.show()
    
    if report:
        print('Classification Report')
        print(classification_report(y_test, y_pred))

# Predict the labels for the test set
Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)  # Convert probabilities to predicted class labels

model.save(model_save_path, include_optimizer=False)
model = tf.keras.models.load_model(model_save_path)

# Define path for the TensorFlow Lite model
tflite_save_path = 'model/point_history_classifier/point_history_classifier.tflite'

converter = tf.lite.TFLiteConverter.from_keras_model(model)  # converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()
# Save the TensorFlow Lite model to disk
open(tflite_save_path, 'wb').write(tflite_quantized_model)

interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print(input_details)

# Run inference using the TFLite model
interpreter.set_tensor(input_details[0]['index'], np.array([X_test[0]]))


interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]['index'])




