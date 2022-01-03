from tensorflow import keras
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

import DuckDuckGoImages as ddg
from pathlib import Path
face_types = 'Padron pepper','Carolina reaper pepper', 'red Spanish pepper', 'habanero pepper', 'Aji Dulce pepper'
print(len(face_types))
path = Path('face_types')
for o in face_types:
    ims = ddg.download(f'{o}', folder=f'./face_types/{o}',max_urls=200)
# from jmd_imagescraper.imagecleaner import *
# display_image_cleaner(path)
# from jmd_imagescraper.core import * 
from fastai.vision.all import *
fns=get_image_files(path)
failed=verify_images(fns)#looks for files that arent images
failed.map(Path.unlink);#unlinks the failed files from the folder
# Code for deleting corrupt images 
import os
import tensorflow as tf

# num_skipped = 0
# for folder_name in ("People_Face", "Face_With_Mask"):
#     folder_path = os.path.join("face_types", folder_name)
#     for fname in os.listdir(folder_path):
#         fpath = os.path.join(folder_path, fname)
#         try:
#             fobj = open(fpath, "rb")
#             is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
#         finally:
#             fobj.close()
#         if not is_jfif:
#             num_skipped += 1
#             # Delete corrupted image
#             os.remove(fpath)
# print("Deleted %d images" % num_skipped)

base_model = keras.applications.VGG16(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=(240, 240, 3),
    include_top=False)
print(base_model.summary())
#Model vastzetten
inputs = keras.Input(shape=(240, 240, 3))
for layers in base_model.layers:
            layers.trainable=False
last_output = base_model.layers[-1].output
vgg_x = keras.layers.Flatten()(last_output)
vgg_x = keras.layers.Dense(128, activation = 'relu')(vgg_x)
vgg_x = keras.layers.Dense(len(face_types), activation = 'softmax')(vgg_x)
model = keras.Model(base_model.input, vgg_x)
print(model.summary())
model.compile(loss = 'categorical_crossentropy', optimizer= 'Nadam', metrics=['acc'])
#model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), metrics=[keras.metrics.BinaryAccuracy()])

#Train all images
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# create a data generator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        rotation_range=25,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.25, # Randomly zoom image 
        width_shift_range=0.25,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.25,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=True) # we don't expect Bo to be upside-down so we will not flip vertically
# load and iterate training dataset
train_it = datagen.flow_from_directory('face_types/', 
                                       target_size=(240, 240), 
                                       color_mode='rgb', 
                                       class_mode='categorical', 
                                       batch_size=32)
# load and iterate test dataset
test_it = datagen.flow_from_directory('face_types/', 
                                      target_size=(240, 240), 
                                      color_mode='rgb', 
                                      class_mode='categorical', 
                                      batch_size=32)
hist= model.fit(train_it, steps_per_epoch=12, validation_data=test_it, validation_steps=4, epochs=35)

import matplotlib.pyplot as plt
#plt.plot(hist.history["binary_accuracy"])
#plt.plot(hist.history['val_binary_accuracy'])
#plt.plot(hist.history['loss'])
#plt.plot(hist.history['val_loss'])
#plt.title("model accuracy")
#plt.ylabel("Accuracy")
#plt.xlabel("Epoch")
#plt.legend(["Binary Accuracy","Validation Accuracy","loss","Validation Loss"])
#plt.show()

# Unfreeze the base model
base_model.trainable = True

# It's important to recompile your model after you make any changes
# to the `trainable` attribute of any inner layer, so that your changes
# are taken into account
model.compile(loss = 'categorical_crossentropy', optimizer= 'Nadam', metrics=['acc'])
#model.compile(optimizer=keras.optimizers.RMSprop(learning_rate = .00001),  # Very low learning rate
#              loss=keras.losses.BinaryCrossentropy(from_logits=True),
#              metrics=[keras.metrics.BinaryAccuracy()])
#hist = model.fit(train_it, steps_per_epoch=12, validation_data=test_it, validation_steps=4, epochs=20)
import matplotlib.pyplot as plt
# plt.plot(hist.history["binary_accuracy"])
# plt.plot(hist.history['val_binary_accuracy'])
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title("model accuracy")
# plt.ylabel("Accuracy")
# plt.xlabel("Epoch")
# plt.legend(["Binary Accuracy","Validation Accuracy","loss","Validation Loss"])
# plt.show()

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input

def show_image(image_path):
    image = mpimg.imread(image_path)
    plt.imshow(image)

def make_predictions(image_path):
    show_image(image_path)
    image = image_utils.load_img(image_path, target_size=(240, 240))
    image = image_utils.img_to_array(image)
    image = image.reshape(1,240,240,3)
    image = preprocess_input(image)
    preds = model.predict(image)
    return preds

def printTestPic(filename):
    tmp = make_predictions(filename)
    print(face_types)
    print(tmp)

printTestPic('0a4a83e6930f4d5686d1104e7f6a8107.jpg')
printTestPic('c.jpg')
printTestPic('dulce.jpg')
printTestPic('marchant.jpg')
printTestPic('0a4a83e6930f4d5686d1104e7f6a8107.jpg')