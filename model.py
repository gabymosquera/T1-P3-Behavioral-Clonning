import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


#Read in .csv file and store each row in 'samples' array.
samples = []
with open('/home/carnd/P3-Gaby/data/data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
images = []
measurements = []
for line in samples:
    for i in range(3):
        #read in the first three columns/items of the samples array to extract the names of the left, center, and right images
        source_path = line[i]
        filename = source_path.split('\\')[-1]
        current_path = '/home/carnd/P3-Gaby/data/data/IMG/' + filename
        #read each image as and RGB image using mpimg
        image = mpimg.imread(current_path)
           
        #this if statement is to make sure that image is being correctly read.
        if image is None:
            print("Image path incorrect: ", current_path)
            continue  # skip adding these rows in the for loop
        
        #all indivual images will be appended to the empty array "images"
        images.append(image)
        
    correction = 0.3 #correction value for the steering angles to account for the left and right camera locations in the car.
    measurement = float(line[3]) #reads in the steering angle from the 4th column of the csv file.
    measurements.append(measurement) #appends the steering angle for the center image to the empty measurements array.
    measurements.append(measurement+correction) #appends the steering angle for the left image which is the center steering angle with a correction angle of .3 to the measurements array.
    measurements.append(measurement-correction) #appends the steering angle for the right image which is the center steering angle with a correction angle of -.3 to the measurements array.
        
        
X_train = np.array(images) #assigns the values of the images to X_train
y_train = np.array(measurements) #assigns the values of the steering angles to y_train


#import keras libraries
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers import Convolution2D, MaxPooling2D, Dropout

#initiate model
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3))) #normalization using lambda
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3))) #crop the bottom 25 pixels and top 70 pixels of the image for a cleaner more valuable image for training.
model.add(Convolution2D(6, 5, 5, activation = "relu")) #First convolutional layer with a 5x5 filter and a relu activation function
model.add(MaxPooling2D()) #First Max Pooling layer for discretization
model.add(Convolution2D(6, 5, 5, activation = "relu")) #Second convolutional layer with a 5x5 filter and a relu activation function
model.add(MaxPooling2D()) #Second Max Pooling layer for discretization
model.add(Convolution2D(6, 3, 3, activation = "relu")) #Third convolutional layer with a 3x3 filter and a relu activation function
model.add(MaxPooling2D()) #Third Max Pooling layer for discretization
model.add(Convolution2D(6, 3, 3, activation = "relu")) #Fourth convolutional layer with a 3x3 filter and a relu activation function
model.add(MaxPooling2D()) #Fourth Max Pooling layer for discretization
model.add(Flatten()) #Flatten layer
model.add(Dense(120)) #First Dense layer to output an array of size 180
model.add(Dropout(0.8)) #First Dropout of 80% of the training samples to prevent overfitting
model.add(Dense(84)) #Second Dense layer to output an array of size 84
model.add(Dropout(0.8)) #Second Dropout of 80% of the training samples to prevent overfitting
model.add(Dense(1)) #Third Dense layer to output an array of size 1


model.compile(loss = 'mse', optimizer = 'adam') #mean square error method together with adam optimizer to calculate loss.
model.fit(X_train, y_train, validation_split = 0.2, shuffle = True, nb_epoch = 2) #fitting the model using the X_train data, y_train data, splitting the samples to validate the model in 20% of the samples, shuffling the samples and training for a total of 2 epochs.

model.save('model.h5') #save the model into a .h5 file.
exit()
