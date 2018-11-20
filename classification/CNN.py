# CNN 分类猫狗
import os

# 这里为了屏蔽tf的输出
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from keras import callbacks
from keras.models import Sequential, model_from_yaml, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPool2D
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
from keras.utils import np_utils, plot_model
from sklearn.model_selection import train_test_split
from keras.applications.resnet50 import preprocess_input, decode_predictions
import re
import matplotlib.image as mpimg
import scipy.io as sio

path = "F:/jars/images/"
files = os.listdir(path)
images = []
labels = []
names = []
classes = []

with open("../data/cats-dogs/labels.txt") as lf:
    for line in lf:
        line = re.sub("\n", "", line)
        if line != "":
            info = line.split(" ")
            # print(info[0])
            names.append(info[0])
            classes.append(info[1])
print(len(names))
print(len(classes))
for f in files:
    img_path = path + f
    # print(img_path)
    if img_path.endswith(".jpg"):
        img = image.load_img(img_path)
        img_array = image.img_to_array(img)
        # img_array = sio.loadmat(img_path)
        # img_array = mpimg.imread(img_path)
        print(img_array)
        # print(len(img_array))
        images.append(img_array)
        breed = f.split(".")[0]
        # print(breed)
        if names.__contains__(breed):
            labels.append(classes[names.index(breed)])
            # print(f+":"+classes[names.index(breed)])

print(len(images))
print(len(labels))
