import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

model = load_model('weights.h5')

script_dir = os.path.dirname(os.path.abspath(__file__))

train_dir = os.path.join(script_dir, '../data/preprocessed/train')
test_dir = os.path.join(script_dir, '../data/preprocessed/test')

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=30,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    color_mode='grayscale',
                                                    target_size=(48,48),
                                                    batch_size=64,
                                                    class_mode='categorical',
                                                    shuffle=True)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  color_mode='grayscale',
                                                  target_size=(48,48),
                                                  batch_size=64,
                                                  class_mode='categorical',
                                                  shuffle=False)

# Continue training the model
model.fit(train_generator,
          steps_per_epoch=int(train_generator.n//train_generator.batch_size),
          epochs=30,
          validation_data=test_generator,
          validation_steps=test_generator.n//test_generator.batch_size)

# Save the model again after training
model.save('weights.h5')


