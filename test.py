from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt 
import numpy as np
import os

from tensorflow.keras.models import load_model

# Load the saved model
# model = load_model("./best_model.keras")
model = load_model("./efficient-net-b5-Brain Tumors-93.46.keras")

# Predict Single Image
image_path = "./brain-tumor/Training/notumor/Tr-noTr_0004.jpg"
img = load_img(image_path, target_size=(320, 320))

# Display the image
plt.imshow(img)
plt.axis('off')  # Turn off axis
plt.show()

def predictions(dir):
        
    total_images = 0
    wrong_pred = 0
    # files = os.path.join(data_dir,fold)
    imgs = os.listdir(dir)
    for img in imgs:
        image_path = f"./brain-tumor/Training/pituitary/{img}"
        img = load_img(image_path, target_size=(320, 320))
        # Preprocess Image
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array / 255.0, axis=0)  # Normalize and expand dimensions
        
        # Make Prediction
        predictions = model.predict(img_array)
        class_index = np.argmax(predictions)
        confidence = predictions[0][class_index]
        
        # Class labels and final result
        class_labels = {0:"Glioma",
                        1:"Meningioma",
                        2:"No Tumor",
                        3:"Pituitary Tumor"}
        
        predicted_label = class_labels[class_index]
        total_images += 1
        if predicted_label != class_labels[3]:
            wrong_pred += 1
            print(f"Predicted Tumor Class: {predicted_label}, Confidence: {confidence:.2f}")
    
    print(f" Total Predictions: {total_images}\n Wrong Predictions: {wrong_pred}\n Correct Predictions: {total_images-wrong_pred}\n Accuracy: {((total_images-wrong_pred)/total_images)*100}")

# Preprocess Image
img_array = img_to_array(img)
img_array = np.expand_dims(img_array / 255.0, axis=0)  # Normalize and expand dimensions

# Make Prediction
predictions = model.predict(img_array)
class_index = np.argmax(predictions)
confidence = predictions[0][class_index]

# Class labels and final result
class_labels = {0:"Glioma",
                1:"Meningioma",
                2:"No Tumor",
                3:"Pituitary Tumor"}

predicted_label = class_labels[class_index]
print(f"Predicted Tumor Class: {predicted_label}, Confidence: {confidence:.2f}")

# dir = "./brain-tumor/Training/pituitary"
# predictions(dir)
