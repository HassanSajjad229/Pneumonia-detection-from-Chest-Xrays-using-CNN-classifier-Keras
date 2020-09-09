"""
Created on Thu Sep 10 03:21:52 2020

@author: Hassan
"""
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D,MaxPool2D
from keras.layers import Activation, Dropout, Flatten, Dense
import os
from keras.preprocessing.image import ImageDataGenerator

num_skipped = 0
for folder_name in ("COVID-19", "Normal"):
    folder_path = os.path.join("D://DeepLearningPractice/COVID-Xray/TrainImages", folder_name)
    for fname in os.listdir(folder_path):
        fpath = os.path.join(folder_path, fname)
        try:
            fobj = open(fpath, "rb")
            is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
        finally:
            fobj.close()

        if not is_jfif:
            num_skipped += 1
            # Delete corrupted image
            os.remove(fpath)

print("Deleted %d images" % num_skipped)

#Fitting the CNN to the images
train_datagen = ImageDataGenerator(
rescale=1./255,
shear_range=0.2,
zoom_range=0.2,
horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)


training_set = train_datagen.flow_from_directory(
'D://DeepLearningPractice/COVID-Xray/TrainImages',
target_size=(64, 64),
batch_size=32,
class_mode='binary')

test_set = test_datagen.flow_from_directory(
    'D://DeepLearningPractice/COVID-Xray/TestImages',
    target_size=(64,64),
    batch_size=32,
    class_mode='binary')

classes = training_set.class_indices
print(classes)       

model = tf.keras.Sequential()

# Must define the input shape in the first layer of the neural network
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=3, activation='relu', input_shape=(64,64,3))) 
model.add(tf.keras.layers.MaxPooling2D(pool_size=2))
model.add(tf.keras.layers.Conv2D(filters=16,kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(units=128,activation='relu'))
model.add(Dense(units=1,activation='sigmoid'))

model.summary()

model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
model.fit(
training_set,
epochs=25,
validation_data=test_set)

score = model.evaluate(test_set, verbose=0)


# Print test accuracy
print('\n', 'Test accuracy:', score[1])