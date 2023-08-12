import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Flatten, Dropout,BatchNormalization ,Activation
from tensorflow.keras.models import Sequential




# Load VGGFace model and freeze its layers
base_model = tf.keras.applications.VGG16(input_shape=(48,48,3),include_top=False,weights="imagenet")

# Freeze some layers
for layer in base_model.layers[:-4]:
    layer.trainable = False

# Building Model

# Build a deeper model
model = Sequential()

# Add base model layers
model.add(base_model)

# Batch Normalization before Activation
model.add(BatchNormalization())
model.add(Activation('relu'))

# MaxPooling layer to reduce spatial dimensions
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add a new Convolutional layer
model.add(Conv2D(128, (3, 3), padding='same', kernel_initializer='he_uniform'))

# Batch Normalization before Activation
model.add(BatchNormalization())
model.add(Activation('relu'))

# MaxPooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Add another Convolutional layer
model.add(Conv2D(256, (3, 3), padding='same', kernel_initializer='he_uniform'))

# Batch Normalization before Activation
model.add(BatchNormalization())
model.add(Activation('relu'))

# MaxPooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output for fully connected layers
model.add(Flatten())

# Fully connected layer with increased neurons
model.add(Dense(128, kernel_initializer='he_uniform'))

# Batch Normalization before Activation
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Another fully connected layer with reduced neurons
model.add(Dense(64, kernel_initializer='he_uniform'))

# Batch Normalization before Activation
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output layer
model.add(Dense(7, activation='softmax'))

model.summary()

optimizer_1 = tf.keras.optimizers.RMSprop(learning_rate=0.001)
model.compile(optimizer=optimizer_1, loss='categorical_crossentropy', metrics=['accuracy'])

# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define directories for training and testing data
train_dir = os.path.join(script_dir, '../data/preprocessed/train')
test_dir = os.path.join(script_dir, '../data/preprocessed/test')

# Define an ImageDataGenerator for data augmentation and normalization during training
train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2,
                                   rotation_range=30,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2)

# Define an ImageDataGenerator for normalization during testing
test_datagen = ImageDataGenerator(rescale=1./255) 

# Create a generator for loading and augmenting training data
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=(48,48),
                                                    batch_size=64,
                                                    class_mode='categorical',
                                                    shuffle=True)

# Create a generator for loading testing data
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(48,48),
                                                  batch_size=64,
                                                  class_mode='categorical',
                                                  shuffle=False)

history = model.fit(train_generator,
                  steps_per_epoch=int(train_generator.n // train_generator.batch_size),
                  epochs=30,
                  validation_data=test_generator,
                  validation_steps=test_generator.n // test_generator.batch_size)

loss, accuracy = model.evaluate(test_generator, steps=test_generator.n // test_generator.batch_size)
print("Test accuracy:", accuracy)

model.save('vgg_weights.h5')


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