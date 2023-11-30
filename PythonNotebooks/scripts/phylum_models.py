from tensorflow import keras
from tensorflow import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Input, Dense, Dropout, LeakyReLU, AveragePooling1D, Conv2D, BatchNormalization
from keras.regularizers import l2
from keras.models import Model
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
from sklearn import preprocessing 
l_enc = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()
#from tensorflow import keras

def fit_model(trainX, trainy,  n_epochs, val_data, opt, val_split, filters):
    # define model
    fil = 16
    k_z = 3
    model = Sequential()
    model.add(Conv1D(filters = filters , kernel_size = 1, activation=keras.layers.LeakyReLU(alpha=0.01), input_shape = (trainX.shape[1],1)))
    #get output dim
    conv1_out = [trainX.shape[1]-k_z,fil]
    model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    pool1_out = [round(conv1_out[0]/2), fil]
    
    #model.add(Conv1D(filters = 32 , kernel_size = 3, activation=keras.layers.LeakyReLU(alpha=0.01)))
    #conv2_out = [pool1_out[0]- 4, 32]
    #model.add(MaxPooling1D(pool_size=2, strides=2, padding='same'))
    #pool2_out = [round(conv1_out[0]/1), 32]
    
    model.add(Flatten())
    flat_out = pool1_out[0]*pool1_out[1]
    model.add(Dropout(0.5))
    model.add(BatchNormalization())
    #model.add(Dense(round(flat_out/4), activation = keras.layers.LeakyReLU(alpha=0.01)))
    #model.add(BatchNormalization())
 
    model.add(Dense(round(flat_out/2), activation = keras.layers.LeakyReLU(alpha=0.01)))

    #model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
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
    
def fit_modelimg(trainX, trainy,  n_epochs, val_data, opt, val_split, naming, fsize, ksize):
    
    encoder_inputs = tf.keras.layers.Input(shape=(trainX.shape[1],trainX.shape[2],trainX.shape[3]), name="encoder_input"+str(naming))

    #conv layer, norm and flatten
    encoder_conv_layer1 = Conv2D(filters = fsize , kernel_size = ksize, padding='same', activation=keras.layers.LeakyReLU(alpha=0.01), name='conv1'+str(naming))(encoder_inputs)
    encoder_pool1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),name='pool1'+str(naming),padding='same')(encoder_conv_layer1)
    encoder_norm_layer1 = tf.keras.layers.BatchNormalization(name="encoder_norm_1"+str(naming))(encoder_pool1)
    if fsize==32:
        f2 = 32
    else:
        f2=fsize*2
    #record conv output
    shape_before_flatten = tf.keras.backend.int_shape(encoder_norm_layer1)[1:]
    encoder_flatten = tf.keras.layers.Flatten()(encoder_norm_layer1)
    encoder_dout1 = tf.keras.layers.Dropout(.5, name='encoder_dout1'+str(naming) )(encoder_flatten)
    #final dense layer
    encoder_activ_layer1 = tf.keras.layers.Dense(units = round(encoder_flatten.shape[1]/2), activation = keras.layers.LeakyReLU(alpha=0.01), name='encoder_dense_1'+str(naming))(encoder_dout1)
    # fit model if epoch above zero
    model = keras.Model(encoder_inputs, encoder_activ_layer1, name="encoder"+str(naming))
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
    
