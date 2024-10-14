import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename
        
        self.class_labels = {
            0: 'Normal',
            1: 'Cyst',
            2: 'Stone',
            3: 'Tumor'
        } 

    def predict(self):
        
        try:
            # Load the model
            model = load_model(os.path.join("model", "model.h5"))

            # Load and preprocess the image
            test_image = image.load_img(self.filename, target_size=(224, 224))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            test_image /= 255.0  # Normalize the image to [0, 1]

            # Make prediction
            predictions = model.predict(test_image)
            result = np.argmax(predictions, axis=1)

            # Get the predicted label
            predicted_label = self.class_labels.get(result[0], "Unknown Class")

            return [{"image": predicted_label}]

        except Exception as e:
            print(f"An error occurred: {e}")
            return [{"image": "Error in prediction"}]
