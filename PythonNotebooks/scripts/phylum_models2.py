from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Input, Dense, Dropout, LeakyReLU, AveragePooling1D, BatchNormalization
from keras.models import Model
import numpy as np
from sklearn import preprocessing 
l_enc = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()
#import tensorflow as tf
#from tensorflow import keras

def fit_model(trainX, trainy,  n_epochs, val_data, opt, val_split):
    # define model
    f = 16
    k = 3
    model = Sequential()
    model.add(Conv1D(filters = f , kernel_size = 1, activation=keras.layers.LeakyReLU(alpha=0.01), input_shape = (trainX.shape[1],1)))
    #get output dim
    conv1_out = [trainX.shape[1]-k,f]
    #model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    #pool1_out = [round(conv1_out[0]/2), 16]
    
    #model.add(Conv1D(filters = 32 , kernel_size = 3, activation=keras.layers.LeakyReLU(alpha=0.01)))
    #conv2_out = [pool1_out[0]- 4, 32]
    #model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    #pool2_out = [round(conv1_out[0]/1), 32]
    model.add(BatchNormalization())
    model.add(Flatten())
    flat_out = conv1_out[0]*conv1_out[1]
    model.add(Dropout(0.3))
    
    #model.add(Dense(round(flat_out/2), activation = keras.layers.LeakyReLU(alpha=0.01)))
 
    model.add(Dense(round(flat_out/12), activation = keras.layers.LeakyReLU(alpha=0.01)))

    #model.add(Dense(1, activation='sigmoid'))

    #model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    # fit model if epoch above zero
    if n_epochs > 0 :
        model.fit(trainX, trainy, 
                  epochs=n_epochs, 
                  batch_size = 8, 
                  verbose=0,  
                  validation_split=val_split,
                  validation_data=val_data, 
                  #validation_batch_size=8,
                  shuffle=True)
        return model
    else:
        return model
