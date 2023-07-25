import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

model = load_model('weights.h5')

script_dir = os.path.dirname(os.path.abspath(__file__))

test_dir = os.path.join(script_dir, '../data/preprocessed/test')

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(test_dir,
                                                  color_mode='grayscale',
                                                  target_size=(48,48),
                                                  batch_size=64,
                                                  class_mode='categorical',
                                                  shuffle=False)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(test_generator)

print("Test Loss:", loss)
print("Test Accuracy:", accuracy)
