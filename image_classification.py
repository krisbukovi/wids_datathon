import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import os
from keras import layers
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv2D, MaxPooling2D
from keras.callbacks import Callback, ModelCheckpoint, History
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import confusion_matrix, roc_auc_score

import glob
import imageio as io

arr1 = []
arr2 = []

training_image_paths = [path for path in os.listdir('train_images')]
training_labels = pd.read_csv('traininglabels.csv')



#print(training_labels)

images = glob.glob('train_images/*.jpg')

'''
for index, row in training_labels.iterrows():

    print(row['image_id'])
    print(row['has_oilpalm'])
'''

for f in images:

    label = f.split('/')[1]

    image = io.imread(f)

    print((training_labels.loc[training_labels['image_id'] == label]['has_oilpalm']).values)
    print((training_labels.loc[training_labels['image_id'] == label]['score']).values)
    arr1.append(image)
    arr2.append()

train_images = np.array(arr1)
train_labels = np.array(arr2)

#print(train_images)
#print(train_labels)

#print(training_labels.info(memory_usage = 'deep'))
