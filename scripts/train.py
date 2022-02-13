from tensorflow.keras.preprocessing.image import ImageDataGenerator
import time
import json
from keras import layers
from keras.utils import np_utils
from keras import optimizers
from matplotlib import pyplot as plt
from keras.preprocessing import image
from keras import models
from tqdm import tqdm
import cv2
import os
import numpy as np
from matplotlib import pyplot 
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras import utils
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split

import os
from tqdm import tqdm
import cv2
import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import image
#from skimage.morphology import skeletonize, thin

from keras import layers
from keras import models
from keras import optimizers

from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU

# x_genuine_path='./aug/train/sign'
x_forgery_path = './scripts/aug/forg/forgery'
# x_test_path = './aug/test'
save_path = './scripts/modelsave'

from keras.preprocessing.image import ImageDataGenerator
tf.random.set_seed(42)
dataset=ImageDataGenerator(
    rescale= 1./ 255,
    rotation_range= 90, 
#     width_shift_range=0.1, 
#     height_shift_range=0.1,
    shear_range=0.1, 
    zoom_range=0.07,
    fill_mode='nearest'
)


def augment_image(image_dir, save_dir): 
    real = dataset.flow_from_directory(image_dir, save_to_dir=save_dir, save_prefix='N', save_format='jpg', batch_size=10)
    for i in range(30):
        real.next()


def preProcessImage(train_path, test, final_img_size=(300, 300)):
    x_train = []
    train_batch = []
    train_data = []
    if test == False:
        train_batch = os.listdir(train_path)
        train_data = [x for x in train_batch if x.endswith('png') or x.endswith(
            'PNG') or x.endswith('jpg') or x.endswith('JPG') or x.endswith('tif') or x.endswith('jpeg')]
    else:
        train_data = [1]

    for sample in tqdm(train_data):
        img_path = ""
        if test == True:
            img_path = train_path
        else:
            img_path = os.path.join(train_path, sample)


        # importing images from drive
        img = cv2.imread(img_path)
        
        # resize image to 600xH
        width = 600
        height = int(600*float(img.shape[0])/img.shape[1])
        dim = (width, height)
        img = cv2.resize(img, dim)

        # changing RGB to grayscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #img = cv2.resize(img, (600, int(600*float(img.shape[0])/img.shape[1])))

        # denoising the grayscale image
        img = cv2.fastNlMeansDenoising(img, None, 7, 21)

        # Perfom Median blur on image
        mbvalue = int(np.max(img.shape)/200)
        mbvalue = mbvalue if mbvalue % 2 == 1 else mbvalue+1
        img = cv2.medianBlur(img, mbvalue)

        # Threshold binary image
#         _, img = cv2.threshold(img,140, 255, cv2.THRESH_BINARY)
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # Image segmentation
        seg = segmentImage(img)
        img = img[seg[2]:seg[3], seg[0]:seg[1]]
        
        #Padding of image
        lp, rp, tp, bp = (0,0,0,0)
        if(img.shape[0]>img.shape[1]):
            lp = int((img.shape[0]-img.shape[1])/2)
            rp = lp
        elif(img.shape[1]>img.shape[0]):
            tp = int((img.shape[1]-img.shape[0])/2)
            bp = tp
        image_padded = cv2.copyMakeBorder(img, tp, bp, lp, rp, cv2.BORDER_CONSTANT, value=255)
    
        # resizing the image
        img = cv2.resize(image_padded, final_img_size)

        # producing image negative
#         img = 255-img

        img = img.astype('float')/255

        # appending it in list
        x_train.append(img)

    # converting it into np-array
    x_train = np.array(x_train)
    return x_train


def generate_model():
    tf.random.set_seed(42)
    mod = models.Sequential()
    mod.add(layers.Conv2D(16, (9, 9), activation='relu', input_shape=(300, 300,1)))
    mod.add(layers.MaxPooling2D((2, 2)))
    mod.add(layers.Conv2D(32, (5, 5), activation='relu'))
    mod.add(layers.MaxPooling2D((2, 2)))
    mod.add(layers.Conv2D(32, (3, 3), activation='relu'))
    mod.add(layers.MaxPooling2D((3, 3)))
    mod.add(layers.Conv2D(16, (2, 2), activation='relu'))
    mod.add(layers.MaxPooling2D((2, 2)))

    mod.add(layers.Flatten())
    mod.add(layers.Dropout(0.4))
    mod.add(layers.Dense(256, activation='relu'))
    mod.add(layers.Dropout(0.5))
    mod.add(layers.Dense(128, activation='relu'))

    mod.add(layers.Dense(1, activation='sigmoid'))

    mod.compile(loss='binary_crossentropy',
                optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3)  , metrics=['acc'])
    return mod;

def segmentImage(image):  
  hHist=np.zeros(shape=image.shape[0], dtype=int)
  vHist=np.zeros(shape=image.shape[1], dtype=int)

  for i in range(image.shape[0]):
    for j in range(image.shape[1]):
      if(image[i][j]==0):
        hHist[i]+=1
        vHist[j]+=1
  
  locLeft=0
  locRight=image.shape[0]
  locTop=0
  locBottom=image.shape[1]
  
  count=0
  for i in range(hHist.shape[0]):
    if(count<=0):
        count=0
        if(hHist[i]!=0):
            locTop=i
            count+=1
    else:
        if(hHist[i]!=0):
            count+=1
        else:
            count-=hHist.shape[0]/100

        if(count>hHist.shape[0]/30):
            break
            
  count=0
  for i in reversed(range(hHist.shape[0])):
    if(count<=0):
        count=0
        if(hHist[i]!=0):
            locBottom=i
            count+=1
    else:
        if(hHist[i]!=0):
            count+=1
        else:
            count-=hHist.shape[0]/100

        if(count>hHist.shape[0]/30):
            break
            
  count=0
  for i in range(vHist.shape[0]):
    if(count<=0):
        count=0
        if(vHist[i]!=0):
            locLeft=i
            count+=1
    else:
        if(vHist[i]!=0):
            count+=1
        else:
            count-=vHist.shape[0]/100

        if(count>vHist.shape[0]/30):
            break
            
  count=0
  for i in reversed(range(vHist.shape[0])):
    if(count<=0):
        count=0
        if(vHist[i]!=0):
            locRight=i
            count+=1
    else:
        if(vHist[i]!=0):
            count+=1
        else:
            count-=vHist.shape[0]/100

        if(count>vHist.shape[0]/30):
            break
            
  return locLeft, locRight, locTop, locBottom

def train(x_genuine_path, user_name):
    print("Genuine Path is ", x_genuine_path)
    print("User Name is " ,user_name)
    if 'mysign_weights.h5' not in os.listdir(save_path):
        # print("AUGMENTING IMAGES......")
        # augment_image(x_genuine_path.split("/signatures")[0], x_genuine_path)
        print("PRE PROCESSING GENUINE IMAGE......")
        x_genuine = preProcessImage(x_genuine_path, False)
        #x_random = csvReader(os.path.join(x_random_csv_path, 'random2'))
        print("PRE PROCESSING FORGERY IMAGE......")
        x_forgery = preProcessImage(x_forgery_path, False)
        
        # X and Y
        X = np.concatenate((x_genuine, x_forgery))
        Y = (np.ones(x_genuine.shape[0]))
        Y = np.append(Y,np.zeros(x_forgery.shape[0]))
        
        # Splitting into train and test 
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=1, shuffle=True)
        # print(X_test.shape)

        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1, shuffle=True)
        # print(y_val.shape)
        
        X_train = tf.expand_dims(X_train, axis=-1)
        # X_train.shape
        
        X_test = tf.expand_dims(X_test, axis=-1)
        
        X_val=tf.expand_dims(X_val, axis=-1)
        
        print("TRAINING YOUR MODEL....... This may take a while.")
        # another training
        mod = generate_model()
        history = mod.fit(
            X_train,
            y_train,
            epochs=10,
            batch_size=4,
            validation_data=(X_val, y_val),
            verbose=1, 
            shuffle=True)

        mod.summary()
        
        evaluated =  mod.evaluate(X_test, y_test)

        model_save_path = user_name + str(round(time.time() * 1000)) + "_weights.h5";
        
        print('Test-Accuracy:', evaluated[1]*100, '%')
        print('Model-Save-Name:', model_save_path)

        mod.save_weights(os.path.join(save_path, model_save_path))

        history_dict = history.history
        acc_value = history_dict['acc'][-1]
        return_data = {
            "accuracy": acc_value,
            "test_accuracy": evaluated[1] * 100,
            "model_save_path": model_save_path
        }
        print("Model-Summary: ", json.dumps(return_data))

    else:
        mod.load_weights(os.path.join(save_path, 'mysign_weights.h5'))


def test(x_test_path, model_weight_name):
    x_test = preProcessImage(os.path.join(x_test_path), True)
    x_test = x_test.reshape(300, 300)
    mod = generate_model()
    mod.load_weights(os.path.join(save_path, model_weight_name))
    predicted_acc = mod.predict(np.array([x_test.reshape(300, 300, 1), ]))
    return_data = {
        "test_accuracy": predicted_acc[0][0] * 100
    }
    print("\n\nAccuracy of Sign in Test Folder: ", predicted_acc*100, "%")
    print("*** If percent > 90% accept the signature ***")
    print("Test-Summary: ", json.dumps(return_data))

import sys

script_type = sys.argv[3]

if(script_type == 'train'):
    train_path = sys.argv[1]
    user_name = sys.argv[2]
    train(train_path, user_name)
elif(script_type == 'test'):
    test_path = sys.argv[1]
    model_weights = sys.argv[2]
    test(test_path, model_weights)