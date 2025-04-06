import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

model = load_model("alz_model_saved.keras", compile=False)  


model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),  
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_datagen.flow_from_directory(
    'C:/160721733066/AlzPred/oasis_dataset/test',  
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  
)

true_labels = test_generator.classes

class_labels = list(test_generator.class_indices.keys())
print("Class Labels:", class_labels)

predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions, axis=1)  


conf_matrix = confusion_matrix(true_labels, predicted_classes)
print("\nConfusion Matrix:\n", conf_matrix)

report = classification_report(true_labels, predicted_classes, target_names=class_labels)
print("\nClassification Report:\n", report)


