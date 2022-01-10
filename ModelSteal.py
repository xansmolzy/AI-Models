
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import random
import os
import cv2
from IPython.display import Image
import matplotlib.pyplot as plt
import seaborn as sns

#from keras.utils import plot_model
from sklearn.metrics import classification_report
from collections import Counter
import tensorflow as tf

import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16

from keras import Model, layers
from keras.models import Sequential
from tensorflow.keras.optimizers import Adam, SGD
from keras.layers import GlobalMaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, Input, Conv2D, MaxPooling2D, Flatten,MaxPooling3D

import DuckDuckGoImages as ddg
from pathlib import Path

face_types = 'Padron pepper','Carolina reaper pepper', 'habanero pepper', 'Aji Dulce pepper', 'Madame jeanette pepper', 'jalapeno pepper', 'cayenne pepper', 'bell pepper', 'naga pepper', 'brain strain pepper'

NEPOCHS = 30
PATIENCE = 3
LOGITS = len(face_types)
SIZEXYZ = [240, 240, 3]
cwd = os.path.dirname(os.path.realpath(__file__))

print(len(face_types))
path = Path('peppers')
for o in face_types:
    ims = ddg.download(f'{o}', folder=f'./peppers/{o}',max_urls=250)
from fastai.vision.all import *
fns=get_image_files(path)
failed=verify_images(fns)#looks for files that arent images
failed.map(Path.unlink);#unlinks the failed files from the folder

#path = cwd + "peppers/"
datagen = ImageDataGenerator(featurewise_center=False, samplewise_center=True, rotation_range=25, zoom_range = 0.25, width_shift_range=0.25, height_shift_range=0.25, horizontal_flip=True, vertical_flip=True)
train_it = datagen.flow_from_directory(path, target_size=(SIZEXYZ[0], SIZEXYZ[1]), color_mode='rgb', class_mode='categorical', batch_size=8)
test_it = datagen.flow_from_directory(path, target_size=(SIZEXYZ[0], SIZEXYZ[1]), color_mode='rgb', class_mode='categorical', batch_size=8)


VGG16_model = VGG16(pooling='avg', weights='imagenet', include_top=False, input_shape=(SIZEXYZ[0], SIZEXYZ[1], SIZEXYZ[2]))
VGG16_model.trainable = False
last_output = VGG16_model.layers[-1].output
vgg_x = keras.layers.Flatten()(last_output)
vgg_x = keras.layers.Dense(128, activation = 'relu')(vgg_x)
vgg_x = keras.layers.Dense(LOGITS, activation = 'softmax')(vgg_x)
vgg16_final_model = Model(VGG16_model.input, vgg_x)
vgg16_final_model.compile(loss = 'categorical_crossentropy', optimizer= 'Nadam', metrics=['acc'])
vgg16_filepath = cwd +'vgg_16_'+'{epoch:02d}-acc-{val_acc:.2f}.hdf5'
vgg_checkpoint = tf.keras.callbacks.ModelCheckpoint(vgg16_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
vgg_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=PATIENCE)
vgg16_history = vgg16_final_model.fit(train_it, epochs = NEPOCHS ,validation_data = test_it, callbacks=[vgg_checkpoint,vgg_early_stopping],verbose=1)


RN50_mod = ResNet50(weights='imagenet', include_top=False, input_shape=(SIZEXYZ[0], SIZEXYZ[1], SIZEXYZ[2]), classes=LOGITS)
RN50_mod.trainable = True
RN50_x = keras.layers.Flatten()(RN50_mod.output)
RN50_x = keras.layers.Dense(256,activation='relu')(RN50_x)
RN50_x = keras.layers.Dense(LOGITS,activation='softmax')(RN50_x)
RN50_final_model = keras.Model(inputs=RN50_mod.input, outputs=RN50_x)
RN50_final_model.compile(loss = 'categorical_crossentropy', optimizer= 'Nadam', metrics=['acc'])
RN_FP = cwd +'resnet50'+'{epoch:02d}-val_acc-{val_acc:.2f}.hdf5'
resnet_checkpoint = tf.keras.callbacks.ModelCheckpoint(RN_FP, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
resnet_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=PATIENCE)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.05, patience=PATIENCE, min_lr=0.000002)
resnet50_history = RN50_final_model.fit(train_it, epochs = NEPOCHS ,validation_data = test_it, callbacks=[resnet_checkpoint,resnet_early_stopping,reduce_lr], verbose=1)


InV3_mod = InceptionV3(input_shape=(SIZEXYZ[0], SIZEXYZ[1], SIZEXYZ[2]),weights='imagenet', include_top=False)
for layer in InV3_mod.layers[:249]:
   layer.trainable = False
for layer in InV3_mod.layers[249:]:
   layer.trainable = True
InV3_lastOut = InV3_mod.output
InV3output = keras.layers.Flatten()(InV3_lastOut)
InV3_x = keras.layers.Dense(1024, activation='relu')(InV3output)
InV3_x = keras.layers.Dropout(0.5)(InV3_x)
InV3_x = keras.layers.Dense(LOGITS, activation='softmax')(InV3_x)
InV3_x_final_model = Model(inputs=InV3_mod.input,outputs=InV3_x)
InV3_x_final_model.compile(optimizer= 'Nadam', loss='categorical_crossentropy',metrics=['accuracy'])
inception_filepath = cwd +'inceptionv3_'+'{epoch:02d}-loss-{loss:.2f}.hdf5'
inception_checkpoint = tf.keras.callbacks.ModelCheckpoint(inception_filepath, monitor='acc', verbose=1, save_best_only=True, mode='max')
inception_early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=PATIENCE)
inceptionv3_history = InV3_x_final_model.fit(train_it, epochs = NEPOCHS, validation_data = test_it,callbacks=[inception_checkpoint,inception_early_stopping],verbose=1)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.imagenet_utils import preprocess_input

def mode(my_list):
    ct = Counter(my_list)
    max_value = max(ct.values())
    return ([key for key, value in ct.items() if value == max_value])

img = cv2.resize(cv2.imread('dulce.jpg'),(SIZEXYZ[0], SIZEXYZ[1]))
img_normalized = img/255
vgg16_image_prediction = np.argmax(vgg16_final_model.predict(np.array([img_normalized])))
print(vgg16_image_prediction)
resnet_50_image_prediction = np.argmax(RN50_final_model.predict(np.array([img_normalized])))
print(resnet_50_image_prediction)
InceptionV3_image_prediction = np.argmax(InV3_x_final_model.predict(np.array([img_normalized])))
print(InceptionV3_image_prediction)
image_prediction = mode([vgg16_image_prediction, resnet_50_image_prediction, InceptionV3_image_prediction])   
print(image_prediction)
