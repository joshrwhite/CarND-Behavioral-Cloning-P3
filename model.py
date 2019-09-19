from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers import Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import math
import numpy as np
from PIL import Image         
import cv2                 
import matplotlib.pyplot as plt
from os import getcwd
import csv
# Fix error with TF and Keras
import tensorflow as tf
# tf.python.control_flow_ops = tf
import sklearn

def displayCV2(img):
    # Displaying a CV2 Image
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

samples = [] #simple array to append all the entries present in the .csv file

with open('./data/driving_log.csv') as csvfile: #currently after extracting the file is present in this path
    reader = csv.reader(csvfile)
    next(reader, None) #this is necessary to skip the first record as it contains the headings
    for line in reader:
        samples.append(line)

# Code for Data Augmentation (Image Generator)
def generator(samples, batch_size=32):
    num_samples = len(samples)
   
    while 1: 
        shuffle(samples) # Shuffling the total images
        for offset in range(0, num_samples, batch_size):
            
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                for i in range(0,3): # Taking 3 images, first one is center, second is left, and third is right
                        
                    name = './data/data/IMG/'+batch_sample[i].split('/')[-1]
                    center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB) # Since CV2 reads an image in BGR we need to convert it to RGB since in drive.py it is RGB
                    center_angle = float(batch_sample[3]) # Getting the steering angle measurement
                    images.append(center_image)
                        
                    # Introducing correction for left and right images
                    # if using the left image (i == 1), then increase the steering angle by 0.2
                    # if using the right image (i == 2), then decrease the steering angle by 0.2
                        
                    if(i == 0):
                        angles.append(center_angle)
                    elif(i == 1):
                        angles.append(center_angle + 0.2)
                    elif(i == 2):
                        angles.append(center_angle - 0.2)
                        
                    # Code for Augmentation of data (6 augmented images per 1 source image)
                    # We flip the image and mirror the associated steering angle measurement
                        
                    images.append(cv2.flip(center_image,1))
                    if(i==0):
                        angles.append(center_angle*-1)
                    elif(i==1):
                        angles.append((center_angle+0.2)*-1)
                    elif(i==2):
                        angles.append((center_angle-0.2)*-1)
                    # Here we can get 6 images from one image    
                        
        
            X_train = np.array(images)
            y_train = np.array(angles)
            
            yield sklearn.utils.shuffle(X_train, y_train) # Here we do not hold the values of X_train and y_train instead we yield the values meaning we hold until generator() is running

### Main Program ###

# Getting the data
lines = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    next(reader)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = './data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
X_train = np.array(images)
y_train = np.array(measurements)

# The Neural Network Architecture (NVIDIA Model)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0)))) 
model.add(Conv2D(24, activation='relu', padding='valid', strides=(2,2), kernel_size=(5, 5)))
model.add(ELU())
model.add(Conv2D(36, activation='relu', padding='valid', strides=(2,2), kernel_size=(5, 5)))
model.add(ELU())
model.add(Conv2D(48, activation='relu', padding='valid', strides=(2,2), kernel_size=(5, 5)))
model.add(ELU())

model.add(Dropout(0.5))

model.add(Conv2D(64, activation='relu', padding='valid', kernel_size=(3, 3)))
model.add(ELU())
model.add(Conv2D(64, activation='relu', padding='valid', kernel_size=(3, 3)))
model.add(ELU())

model.add(Flatten())

model.add(Dense(100))
model.add(ELU())
model.add(Dense(50))
model.add(ELU())
model.add(Dense(10))
model.add(ELU())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')

train_samples, validation_samples = train_test_split(samples,test_size=0.15) #simply splitting the dataset to train and validation set usking sklearn. .15 indicates 15% of the dataset is validation set

# Compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator,   nb_val_samples=len(validation_samples), nb_epoch=5, verbose=1)


print(model.summary())

model.save('model.h5')

