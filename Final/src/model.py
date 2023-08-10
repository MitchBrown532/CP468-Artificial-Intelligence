import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# Load a pre-trained neural network model
model = load_model('weights.h5')

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

# Continue training the model using the training data and validation on the testing data
model.fit(train_generator,
          steps_per_epoch=int(train_generator.n//train_generator.batch_size),
          epochs=30,
          validation_data=test_generator,
          validation_steps=test_generator.n//test_generator.batch_size)

# Save the model again after training
model.save('weights.karas')
