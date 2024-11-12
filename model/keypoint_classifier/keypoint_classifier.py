import numpy as np
import tensorflow as tf


class KeyPointClassifier(object):
    """
    A class for classifying keypoints using a TensorFlow Lite model.
    
    Attributes:
        interpreter (tf.lite.Interpreter): The TensorFlow Lite interpreter to run inference.
        input_details (list): List of input details (e.g., tensor shape, type).
        output_details (list): List of output details (e.g., tensor shape, type).
    
    Methods:
        __call__(landmark_list): Perform inference using the input landmark list and return the predicted class index.
    """

    def __init__(
        self,
        model_path='model/keypoint_classifier/keypoint_classifier.tflite',
        num_threads=1,
    ):
        """
        Initializes the KeyPointClassifier with the given model path and number of threads.

        Args:
            model_path (str): The path to the TensorFlow Lite model file.
            num_threads (int): The number of threads to use for inference. Default is 1.
        """

        # Load the TensorFlow Lite model into the interpreter
        self.interpreter = tf.lite.Interpreter(model_path=model_path,
                                               num_threads=num_threads)
        # Allocate memory for the input and output tensors
        self.interpreter.allocate_tensors()
        # Get details of the input and output tensors (e.g., shape, data type)
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def __call__(
        self,
        landmark_list,
    ):
        """
        Performs inference using the provided landmark list and returns the predicted class index.

        Args:
            landmark_list (list or np.array): The input keypoint data (landmarks) to be classified.
                It should be a list or numpy array of keypoints in a specific format (e.g., flattened array of coordinates).

        Returns:
            int: The predicted class index (the class with the highest probability).
        """
        # Get the index of the input tensor from the model details
        input_details_tensor_index = self.input_details[0]['index']

        # Set the input tensor for the model (landmark_list is converted to a numpy array of float32)
        self.interpreter.set_tensor(
            input_details_tensor_index,
            np.array([landmark_list], dtype=np.float32))
        
        # Run the model inference
        self.interpreter.invoke()

        # Get the index of the output tensor from the model details
        output_details_tensor_index = self.output_details[0]['index']

        # Get the output result (probabilities for each class)
        result = self.interpreter.get_tensor(output_details_tensor_index)

        # Squeeze the result (remove any extra dimensions) and get the class with the highest probability
        result_index = np.argmax(np.squeeze(result))

        # Return the predicted class index
        return result_index