import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Model Configuration
img_height, img_width = 224, 224
batch_size = 32

# Data Augmentation and Data Loading
val_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    'data/preprocessed/validation',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',  # Multi-class classification
    shuffle=False
)

# Load Model
model = load_model('models/model.h5')

# Evaluate Model
val_steps = val_generator.samples // batch_size
y_true = val_generator.classes  # True labels

# Get predictions
y_pred = model.predict(val_generator, steps=val_steps)
y_pred = np.argmax(y_pred, axis=1)  # Convert predictions to class indices

# Confusion Matrix and Classification Report
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred, target_names=val_generator.class_indices.keys()))
