import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import math

# Define paths to your dataset
train_dir = '/Users/nishanttiwari/Desktop/farmland_classifier/dataset/train'
validation_dir = '/Users/nishanttiwari/Desktop/farmland_classifier/dataset/validation'

# Create ImageDataGenerators for data loading and augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_datagen = ImageDataGenerator(rescale=1./255)

# Load images from directories
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=5,
    classes=['positive', 'negative'],
    class_mode='binary',
    color_mode='rgb',
    shuffle=True
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(256, 256),
    batch_size=5,
    classes=['positive', 'negative'],
    class_mode='binary',
    color_mode='rgb',
    shuffle=False
)

# Calculate steps per epoch and validation steps
train_steps = math.ceil(train_generator.samples / train_generator.batch_size)
validation_steps = validation_generator.samples // validation_generator.batch_size

# Define the CNN model architecture
model = Sequential([
    Input(shape=(256, 256, 3)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Number of epochs set manually in the code
num_epochs = 5

# Train the model
model.fit(
    train_generator,
    steps_per_epoch=train_steps,
    epochs=num_epochs,
    validation_data=validation_generator,
    validation_steps=validation_steps
)

# Save the model
model.save('farmland_classifier_model.keras')
print("Model trained and saved as 'farmland_classifier_model.keras'")
