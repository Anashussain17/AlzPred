import os
print("Current working directory:", os.getcwd())  
from tensorflow.keras.applications.resnet50 import preprocess_input 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import ReduceLROnPlateau
import numpy as np
from model import create_custom_resnet

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=30,  
    width_shift_range=0.25,
    height_shift_range=0.25,
    shear_range=0.25,
    zoom_range=0.25,
    horizontal_flip=True,
    vertical_flip=True,  
    brightness_range=[0.8, 1.2], 
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'C:/160721733066/AlzPred/oasis_dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)
print("Class indices:", train_generator.class_indices)

val_generator = train_datagen.flow_from_directory(
    'C:/160721733066/AlzPred/oasis_dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)
print("Class indices:", val_generator.class_indices)
# Compute class weights
train_classes = train_generator.classes
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_classes),
    y=train_classes
)
class_weights = dict(enumerate(class_weights))


model = create_custom_resnet()


callbacks = [
    EarlyStopping(patience=5, restore_best_weights=True),
    ModelCheckpoint('alz_model.keras', save_best_only=True),
    ReduceLROnPlateau(  
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7
    )
]


history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=50,
    validation_data=val_generator,
    validation_steps=val_generator.samples // 32,
    callbacks=callbacks,
    class_weight=class_weights  
)

model.save("alz_model.h5")  

print("Saved model files:", os.listdir())
# Save the training history in a pickle file
import pickle
with open('training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)
final_epoch = len(history.history['accuracy'])
print("\n--- Training Report ---")
print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.2%}")
print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")
print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")

from tensorflow.keras.models import load_model
