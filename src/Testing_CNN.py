import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model('farmland_classifier_model.keras')
print("Model loaded successfully.")

# Path to the single image you want to test
img_path = '/Users/nishanttiwari/Desktop/output_256_1024_328-1343_quad.tif'  # Update this with the path to your image

# Load the image and preprocess it
img = image.load_img(img_path, target_size=(256, 256))  # Resize image to match the input shape of the model
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Rescale the image

# Make a prediction
prediction = model.predict(img_array)
class_label = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

# Output the result
print(f"Prediction: {class_label}, Confidence: {prediction[0][0]:.4f}")
