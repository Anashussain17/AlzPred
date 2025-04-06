import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

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

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_generator)

print(f"Test Accuracy: {test_accuracy:.2%}")
print(f"Test Loss: {test_loss:.4f}")
