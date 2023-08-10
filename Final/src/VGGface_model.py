import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras_vggface.vggface import VGGFace

# Load VGGFace model and freeze its layers
vggface_model = VGGFace(model='resnet50', include_top=False, input_shape=(48, 48, 1))

for layer in vggface_model.layers:
    layer.trainable = False

input_layer = Input(shape=(48, 48, 1))  
vggface_features = vggface_model(input_layer)

flatten_layer = Flatten()(vggface_features)
dense_layer = Dense(128, activation='relu')(flatten_layer)
dropout_layer = Dropout(0.5)(dense_layer) 
output_layer = Dense(7, activation='softmax')(dropout_layer)

emotion_model = Model(inputs=input_layer, outputs=output_layer)

emotion_model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])


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

emotion_model.fit(train_generator,
                  steps_per_epoch=int(train_generator.n // train_generator.batch_size),
                  epochs=30,
                  validation_data=test_generator,
                  validation_steps=test_generator.n // test_generator.batch_size)

loss, accuracy = emotion_model.evaluate(test_generator, steps=test_generator.n // test_generator.batch_size)
print("Test accuracy:", accuracy)

emotion_model.save('emotion_recognition_model.h5')
