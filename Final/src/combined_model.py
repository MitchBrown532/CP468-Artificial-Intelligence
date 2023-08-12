import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from keras.models import load_model
from keras.models import Model
from keras.layers import Input, concatenate, Dense, Dropout


# Load Keras model
keras_model = load_model('weights.h5')


# Load VGGface Model
vggface_model = load_model('vgg_weights.h5')


# Define input layers
keras_input = keras_model.input
vggface_input = vggface_model.input

# Get intermediate layers from both models
keras_output = keras_model.layers[-2].output  # Replace with appropriate layer index
vggface_output = vggface_model.layers[-1].output

# Add some dense layers for dimension reduction
keras_output = Dense(256, activation='relu')(keras_output)
vggface_output = Dense(256, activation='relu')(vggface_output)

# Concatenate or merge the outputs
combined_output = concatenate([keras_output, vggface_output], axis=-1)

# Add more layers as needed
combined_output = Dense(128, activation='relu')(combined_output)
combined_output = Dropout(0.5)(combined_output)
combined_output = Dense(7, activation='softmax')(combined_output) 

# Create the combined model
combined_model = Model(inputs=[keras_input, vggface_input], outputs=combined_output)

combined_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define directories for training and testing data
train_dir = os.path.join(script_dir, '../data/preprocessed/train')
test_dir = os.path.join(script_dir, '../data/preprocessed/test')

# Define an ImageDataGenerator for data augmentation and normalization during training
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
# Define an ImageDataGenerator for normalization during testing
test_datagen = ImageDataGenerator(rescale=1./255)
# Create a generator for loading and augmenting training data
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    color_mode='grayscale',
                                                    target_size=(48,48),
                                                    batch_size=64,
                                                    class_mode='categorical',
                                                    shuffle=True)
# Create a generator for loading testing data
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  color_mode='grayscale',
                                                  target_size=(48,48),
                                                  batch_size=64,
                                                  class_mode='categorical',
                                                  shuffle=False)

# Fine-tune the combined model with your dataset
history = combined_model.fit(train_generator,
                  steps_per_epoch=int(train_generator.n // train_generator.batch_size),
                  epochs=30,
                  validation_data=test_generator,
                  validation_steps=test_generator.n // test_generator.batch_size)

combined_model.save('combined_weights.h5')

# Plotting - These results should be saved and placed into figures
# Plot training & validation loss values
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training & validation accuracy values
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()