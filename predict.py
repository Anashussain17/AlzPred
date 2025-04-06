import numpy as np
import tensorflow as tf

# Load and preprocess your image (modify as per your preprocessing pipeline)
image_path = "C:/160721733066/AlzPred/oasis_dataset/train/ModerateDemented/OAS1_0308_MR1_mpr-1_160.jpg"
img = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize if your training data was normalized
model=load_model(alz_model.keras)
# Perform prediction
preds = model.predict(img_array)

# Print raw probabilities
print("Raw predicted probabilities:", preds[0])  # Since batch size is 1

# Get the predicted class index
predicted_class = np.argmax(preds)
print("Predicted class index:", predicted_class)

# If your classes are ordered as per train_generator.class_indices:
class_labels = ["Mild Demented", "Moderate Demented", "Non Demented", "Very Mild Demented"]  # Adjust based on actual order
predicted_label = class_labels[predicted_class]

print("Final Prediction:", predicted_label)
