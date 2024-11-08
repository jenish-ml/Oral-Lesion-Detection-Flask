import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.callbacks import EarlyStopping

# Model Configuration
img_height, img_width = 224, 224
batch_size = 32

# Data Augmentation and Data Loading
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Adjusted to 'categorical' class_mode for multi-class classification
train_generator = train_datagen.flow_from_directory(
    'data/preprocessed/train',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'  
)

val_generator = val_datagen.flow_from_directory(
    'data/preprocessed/validation',
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical', 
    shuffle=False
)

# Get the number of classes from the generator
num_classes = train_generator.num_classes

# Model Definition
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(num_classes, activation='softmax')  # Changed to softmax for multi-class classification
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # Changed to categorical_crossentropy for multi-class classification
              metrics=['accuracy'])

# Training
early_stopping = EarlyStopping(monitor='val_loss', patience=3)
model.fit(
    train_generator,
    epochs=20,
    validation_data=val_generator,
    callbacks=[early_stopping]
)

# Save Model
model.save('models/model.h5')
