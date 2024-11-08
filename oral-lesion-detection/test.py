import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Define the path to your test data and model
test_data_dir = 'data/test'
model_path = 'models/model.h5'  # Your specified model path

# Load the trained model
model = load_model(model_path)

# Preprocess test images using ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1.0/255)

# Load images from the test folder
test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(224, 224),  # Adjust this to match your model's input size if needed
    batch_size=32,
    class_mode='categorical',  # Change to 'binary' if you have only 2 classes
    shuffle=False
)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(test_generator)
