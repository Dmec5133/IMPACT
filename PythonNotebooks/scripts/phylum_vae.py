from tensorflow import keras
from tensorflow import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Input, Dense, Dropout, LeakyReLU, AveragePooling1D, Conv2D
from keras.regularizers import l2
from keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn import preprocessing 
l_enc = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()

import tensorflow
from tensorflow import keras
from keras.models import load_model
def load_all_modelsvae(n_models,k, in_dict):
    all_models = list()
    encoders = list()
    for k2,v2 in in_dict.items():
    # define filename for this ensemble
        filename1 = 'phylum_vae/' + str(k) +  '/encoder_' + str(k2) + '.tf'
        # load model from file
        encoder = load_model(filename1,custom_objects={'Sampling': Sampling})
        filename2 = 'phylum_vae/' + str(k) +  '/decoder_' + str(k2) + '.tf'
        # load model from file
        decoder = load_model(filename2,custom_objects={'Sampling': Sampling})
        # add to list of members
        vae = VAE(encoder, decoder)
        # vae.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001))
        all_models.append(vae)
        encoders.append(encoder)
        
        print('>loaded %s' % filename1 +'>loaded %s' % filename2)
    return all_models, encoders


def get_enc(in_shape, fsize, ksize, latent_dim, naming):
    encoder_inputs = tensorflow.keras.layers.Input(shape=(in_shape.shape[1],1), name="encoder_input"+str(naming))

    #conv layer, norm and flatten
    encoder_conv_layer1 = Conv1D(filters = fsize , kernel_size = ksize, activation=keras.layers.LeakyReLU(alpha=0.01), name='conv1'+str(naming))(encoder_inputs)
    encoder_norm_layer1 = tensorflow.keras.layers.BatchNormalization(name="encoder_norm_1"+str(naming))(encoder_conv_layer1)
    #record conv output
    shape_before_flatten = tensorflow.keras.backend.int_shape(encoder_norm_layer1)[1:]
    encoder_flatten = tensorflow.keras.layers.Flatten()(encoder_conv_layer1)

    #shapes
    conv1_out = [in_shape.shape[1]-ksize,fsize]
    flat_out = conv1_out[0]*conv1_out[1]

    #final dense layer
    encoder_activ_layer1 = tensorflow.keras.layers.Dense(units = round(flat_out/2), activation = keras.layers.LeakyReLU(alpha=0.01), name='encoder_dense_1'+str(naming))(encoder_flatten)
    encoder_activ_layer2 = tensorflow.keras.layers.Dense(units = round(encoder_activ_layer1.shape[1]/2), activation = keras.layers.LeakyReLU(alpha=0.01), name='encoder_dense_2'+str(naming))(encoder_activ_layer1)

    #shape_before_flatten = tensorflow.keras.backend.int_shape(encoder_activ_layer5)[1:]

    latent_dim = latent_dim

    z_mean = layers.Dense(latent_dim, name="z_mean"+str(naming))(encoder_activ_layer2)
    z_log_var = layers.Dense(latent_dim, name="z_log_var"+str(naming))(encoder_activ_layer2)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder"+str(naming))
    return encoder, shape_before_flatten

def get_dec(in_shape, fsize, ksize, latent_dim, shape_before_flatten, naming):
    latent_inputs = tensorflow.keras.layers.Input(shape=(latent_dim), name="decoder_input"+str(naming))

    conv1_out = [in_shape.shape[1]-ksize,fsize]
    flat_out = conv1_out[0]*conv1_out[1]

    decoder_dense_layer1 = tensorflow.keras.layers.Dense(units=round(flat_out/2), activation = keras.layers.LeakyReLU(alpha=0.01), name="decoder_dense_1"+str(naming))(latent_inputs)
    decoder_dense_layer12 = tensorflow.keras.layers.Dense(units=round(decoder_dense_layer1.shape[1]/2), activation = keras.layers.LeakyReLU(alpha=0.01), name="decoder_dense_12"+str(naming))(decoder_dense_layer1)

    decoder_dense_layer2 = tensorflow.keras.layers.Dense(units=np.prod(shape_before_flatten), activation = keras.layers.LeakyReLU(alpha=0.01), name="decoder_dense_2"+str(naming))(decoder_dense_layer12)
    
    

    decoder_reshape = tensorflow.keras.layers.Reshape(target_shape=shape_before_flatten)(decoder_dense_layer2)

    decoder_norm_layer1 = tensorflow.keras.layers.BatchNormalization(name="decoder_norm_1"+str(naming))(decoder_reshape)

    decoder_conv_tran_layer1 = tensorflow.keras.layers.Conv1DTranspose(filters=fsize, kernel_size=ksize, name="decoder_convt"+str(naming))(decoder_norm_layer1)

    decoder_outputs = tensorflow.keras.layers.Dense(units = 1,activation =  'sigmoid', name="decoder_output"+str(naming))(decoder_conv_tran_layer1)

    decoder = tensorflow.keras.models.Model(latent_inputs, decoder_outputs, name="decoder_model"+str(naming))
    return decoder
def get_enc2(in_shape, fsize, ksize, latent_dim, naming):
    encoder_inputs = tensorflow.keras.layers.Input(shape=(in_shape.shape[1],in_shape.shape[2],in_shape.shape[3]), name="encoder_input"+str(naming))

    #conv layer, norm and flatten
    encoder_conv_layer1 = Conv2D(filters = fsize , kernel_size = ksize, padding='same', activation=keras.layers.LeakyReLU(alpha=0.01), name='conv1'+str(naming))(encoder_inputs)
    encoder_pool1 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),name='pool1'+str(naming),padding='same')(encoder_conv_layer1)
    encoder_norm_layer1 = tensorflow.keras.layers.BatchNormalization(name="encoder_norm_1"+str(naming))(encoder_pool1)
    
    if fsize==32:
        f2 = 32
    else:
        f2=fsize*2
    
    # encoder_conv_layer2 = Conv2D(filters = f2 , kernel_size = ksize, padding="same",activation=keras.layers.LeakyReLU(alpha=0.01), name='conv2'+str(naming))(encoder_norm_layer1)
    # encoder_pool2 = tf.keras.layers.AveragePooling2D(pool_size=(2, 2),name='pool2'+str(naming),padding='same')(encoder_conv_layer2)
    # encoder_norm_layer2 = tensorflow.keras.layers.BatchNormalization(name="encoder_norm_2"+str(naming))(encoder_pool2)
    #print(encoder_norm_layer2.shape)
    #record conv output
    shape_before_flatten = tensorflow.keras.backend.int_shape(encoder_norm_layer1)[1:]
    encoder_flatten = tensorflow.keras.layers.Flatten()(encoder_norm_layer1)
    #print(encoder_flatten.shape)

    #shapes
    #conv1_out = [in_shape.shape[1]-ksize,fsize]
    #flat_out = conv1_out[0]*conv1_out[1]
    encoder_dout1 = tf.keras.layers.Dropout(.5, name='encoder_dout1'+str(naming) )(encoder_flatten)
    #final dense layer
    encoder_activ_layer1 = tensorflow.keras.layers.Dense(units = round(encoder_flatten.shape[1]/2), activation = keras.layers.LeakyReLU(alpha=0.01), name='encoder_dense_1'+str(naming))(encoder_dout1)
    #encoder_dout2 = tf.keras.layers.Dropout(.2, name='encoder_dout2'+str(naming) )(encoder_activ_layer1)
    #encoder_activ_layer2 = tensorflow.keras.layers.Dense(units = round(encoder_activ_layer1.shape[1]/2), activation = keras.layers.LeakyReLU(alpha=0.01), name='encoder_dense_2'+str(naming))(encoder_activ_layer1)

    #shape_before_flatten = tensorflow.keras.backend.int_shape(encoder_activ_layer5)[1:]
    

    latent_dim = latent_dim

    z_mean = layers.Dense(latent_dim, name="z_mean"+str(naming))(encoder_activ_layer1)
    z_log_var = layers.Dense(latent_dim, name="z_log_var"+str(naming))(encoder_activ_layer1)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder"+str(naming))
    return encoder, shape_before_flatten

def get_dec2(in_shape, fsize, ksize, latent_dim, shape_before_flatten, naming):
    latent_inputs = tensorflow.keras.layers.Input(shape=(latent_dim), name="decoder_input"+str(naming))

    conv1_out = [in_shape.shape[1]-ksize,fsize]
    flat_out = conv1_out[0]*conv1_out[1]
    if fsize==32:
        f2 = 32
    else:
        f2=fsize*2

    decoder_dense_layer1 = tensorflow.keras.layers.Dense(units=round(flat_out/2), activation = keras.layers.LeakyReLU(alpha=0.01), name="decoder_dense_1"+str(naming))(latent_inputs)
    #decoder_dense_layer12 = tensorflow.keras.layers.Dense(units=decoder_dense_layer1.shape[1]*2, activation = keras.layers.LeakyReLU(alpha=0.01), name="decoder_dense_12"+str(naming))(decoder_dense_layer1)

    decoder_dense_layer2 = tensorflow.keras.layers.Dense(units=np.prod(shape_before_flatten), activation = keras.layers.LeakyReLU(alpha=0.01), name="decoder_dense_2"+str(naming))(decoder_dense_layer1 )
    
    encoder_dec_dout1 = tf.keras.layers.Dropout(.5, name='encoder_dout1'+str(naming) )(decoder_dense_layer2)
    
    

    decoder_reshape1 = tensorflow.keras.layers.Reshape(target_shape=shape_before_flatten, name="dec_resh1")(encoder_dec_dout1)

    # decoder_norm_layer1 = tensorflow.keras.layers.BatchNormalization(name="decoder_norm_1"+str(naming))(decoder_reshape1)
    # decoder_conv_tran_layer1 = tensorflow.keras.layers.Conv2DTranspose(filters=f2, kernel_size=ksize, padding='same',name="decoder_convt1"+str(naming))(decoder_norm_layer1)
    # decoder_up1 = tensorflow.keras.layers.UpSampling2D((2, 2), name="decoder_upsample1")(decoder_conv_tran_layer1)
    
    decoder_norm_layer2 = tensorflow.keras.layers.BatchNormalization(name="decoder_norm_2"+str(naming))(decoder_reshape1)
    decoder_conv_tran_layer2 = tensorflow.keras.layers.Conv2DTranspose(filters=fsize, kernel_size=ksize, name="decoder_convt2"+str(naming),padding='same')(decoder_norm_layer2)
    decoder_up2 = tensorflow.keras.layers.UpSampling2D((2, 2), name="decoder_upsample2")(decoder_conv_tran_layer2)
    
    

    #decoder_outputs =  tensorflow.keras.layers.Reshape(target_shape=(in_shape.shape[1],in_shape.shape[2],in_shape.shape[3]), name="dec_resh2")(decoder_up2)
    #Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)
    decoder_outputs = tensorflow.keras.layers.Dense(units = 3,activation =  'sigmoid', name="decoder_output"+str(naming))(decoder_up2 )
   # decoder_outputs2 = tensorflow.keras.layers.Cropping2D(cropping=((1, 0), (1, 0)), data_format=None)(decoder_outputs)


    decoder = tensorflow.keras.models.Model(latent_inputs, decoder_outputs, name="decoder_model"+str(naming))
    return decoder
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=-1),keepdims=True)
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }