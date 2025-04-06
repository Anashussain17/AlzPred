import pickle

# Load training history
with open('training_history.pkl', 'rb') as f:
    history = pickle.load(f)

# Extract metrics
train_acc = history['accuracy']
val_acc = history['val_accuracy']
train_loss = history['loss']
val_loss = history['val_loss']

# Print final values
print(f"Final Training Accuracy: {train_acc[-1]:.2%}")
print(f"Final Validation Accuracy: {val_acc[-1]:.2%}")
print(f"Final Training Loss: {train_loss[-1]:.4f}")
print(f"Final Validation Loss: {val_loss[-1]:.4f}")

