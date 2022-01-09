import os
os.system('gst-launch-1.0 nvarguscamerasrc num-buffers=1 ! nvvidconv ! \'video/x-raw(memory:NVMM), format=I420\' ! nvjpegenc ! filesink location=test.jpeg')
from keras.models import load_model
import numpy as np
import pandas as pd
import cv2

SIZEXYZ = [240, 240, 3]

VGG16_MOD  = load_model('model.hdf5')
RESNET_MOD = load_model('model.hdf5')
INCEP_MOD  = load_model('model.hdf5')

def mode(my_list):
    ct = Counter(my_list)
    max_value = max(ct.values())
    return ([key for key, value in ct.items() if value == max_value])

img = cv2.resize(cv2.imread('test.jpeg'),(SIZEXYZ[0], SIZEXYZ[1]))
img_normalized = img/255
vgg16_image_prediction = np.argmax(VGG16_MOD.predict(np.array([img_normalized])))
print(vgg16_image_prediction)
resnet_50_image_prediction = np.argmax(RESNET_MOD.predict(np.array([img_normalized])))
print(resnet_50_image_prediction)
InceptionV3_image_prediction = np.argmax(INCEP_MOD.predict(np.array([img_normalized])))
print(InceptionV3_image_prediction)
image_prediction = mode([vgg16_image_prediction, resnet_50_image_prediction, InceptionV3_image_prediction])