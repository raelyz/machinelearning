import os
import zipfile
import random
from shutil import copyfile
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# from google.colab import files
from tensorflow.keras.preprocessing import image


# local_zip = "./cats-and-dogs.zip"
# zip_ref = zipfile.ZipFile(local_zip, "r")
# zip_ref.extractall("./")
# zip_ref.close()
# print(len(os.listdir("./PetImages/Cat/")))
# print(len(os.listdir("./PetImages/Dog/")))

# try:
#     os.mkdir("./cats-v-dogs")
#     os.mkdir("./cats-v-dogs/training")
#     os.mkdir("./cats-v-dogs/testing")
#     os.mkdir("./cats-v-dogs/training/cats")
#     os.mkdir("./cats-v-dogs/training/dogs")
#     os.mkdir("./cats-v-dogs/testing/cats")
#     os.mkdir("./cats-v-dogs/testing/dogs")
# except OSError:
#     pass


# def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
#     files = []
#     for filename in os.listdir(SOURCE):
#         file = SOURCE + filename
#         if os.path.getsize(file) > 0:
#             files.append(filename)
#         else:
#             print(filename + " is zero length, so ignoring.")

#     training_length = int(len(files) * SPLIT_SIZE)
#     testing_length = int(len(files) - training_length)
#     shuffled_set = random.sample(files, len(files))
#     training_set = shuffled_set[0:training_length]
#     testing_set = shuffled_set[:testing_length]

#     for filename in training_set:
#         this_file = SOURCE + filename
#         destination = TRAINING + filename
#         copyfile(this_file, destination)

#     for filename in testing_set:
#         this_file = SOURCE + filename
#         destination = TESTING + filename
#         copyfile(this_file, destination)


# CAT_SOURCE_DIR = "./PetImages/Cat/"
# TRAINING_CATS_DIR = "./cats-v-dogs/training/cats/"
# TESTING_CATS_DIR = "./cats-v-dogs/testing/cats/"
# DOG_SOURCE_DIR = "./PetImages/Dog/"
# TRAINING_DOGS_DIR = "./cats-v-dogs/training/dogs/"
# TESTING_DOGS_DIR = "./cats-v-dogs/testing/dogs/"

# split_size = 0.9
# split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
# split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)
# Expected output
# 666.jpg is zero length, so ignoring
# 11702.jpg is zero length, so ignoring

model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Conv2D(
            16, (3, 3), activation="relu", input_shape=(150, 150, 3)
        ),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(
    optimizer=RMSprop(lr=0.001), loss="binary_crossentropy", metrics=["accuracy"]
)

TRAINING_DIR = "./cats-v-dogs/training/"
train_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = train_datagen.flow_from_directory(
    TRAINING_DIR, batch_size=100, class_mode="binary", target_size=(150, 150)
)

VALIDATION_DIR = "./cats-v-dogs/testing/"
validation_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
validation_generator = validation_datagen.flow_from_directory(
    VALIDATION_DIR, batch_size=100, class_mode="binary", target_size=(150, 150)
)

history = model.fit(
    train_generator, epochs=15, verbose=1, validation_data=validation_generator
)


# -----------------------------------------------------------
# Retrieve a list of list results on training and test data
# sets for each training epoch
# -----------------------------------------------------------
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs = range(len(acc))  # Get number of epochs

# ------------------------------------------------
# Plot training and validation accuracy per epoch
# ------------------------------------------------
plt.plot(epochs, acc, "r", "Training Accuracy")
plt.plot(epochs, val_acc, "b", "Validation Accuracy")
plt.title("Training and validation accuracy")
plt.figure()

# ------------------------------------------------
# Plot training and validation loss per epoch
# ------------------------------------------------
plt.plot(epochs, loss, "r", "Training Loss")
plt.plot(epochs, val_loss, "b", "Validation Loss")
plt.figure()

print("test")

# uploaded = files.upload()

# for fn in uploaded.keys():

#     # predicting images
#     path = "/content/" + fn
#     img = image.load_img(path, target_size=(150, 150))
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)

#     images = np.vstack([x])
#     classes = model.predict(images, batch_size=10)
#     print(classes[0])
#     if classes[0] > 0.5:
#         print(fn + " is a dog")
#     else:
#         print(fn + " is a cat")
