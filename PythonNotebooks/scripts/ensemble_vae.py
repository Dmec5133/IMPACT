from tensorflow import keras
from tensorflow import *
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Input, Dense, Dropout, LeakyReLU, AveragePooling1D
from tensorflow.keras import layers
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Input, Dense, Dropout, LeakyReLU, AveragePooling1D, BatchNormalization
from keras.regularizers import l2
from keras.models import Model
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn import preprocessing 
from sklearn.utils import class_weight
from sklearn.utils import compute_class_weight
from keras.models import load_model
from keras.callbacks import History
l_enc = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()


def define_stacked_vae(members, opt, ens_pth, in_dict):
    
    for i in range(len(members)):
        model = members[i]
        
        for layer in model.layers:
            #layer.trainable = False
            layer._name = 'ensemble_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [ model.output[2] for model in members]
    #print(model.output)
    #print(ensemble_outputs)
    merge = concatenate(ensemble_outputs)
    dout = Dropout(0.5, name = 'ens_dout1')(merge)
    bnorm = BatchNormalization(name="ens_norm1")(dout)
    hidden1 = Dense(round(merge.shape[1]/2), activation=keras.layers.LeakyReLU(alpha=0.01), name = 'vae1.fc')(bnorm)
   
    output = Dense(1, activation='sigmoid', name='vae.out')(hidden1)
    model = Model(inputs=ensemble_visible, outputs=output, name='stacked_vae')
    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file=ens_pth + 'model_graph.png')
    # compile
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.fit(stackedX, inputy, epoch = 100 )
    
    return model
def define_stacked_vaeimg(members, opt, ens_pth, in_dict):
    
    for i in range(len(members)):
        model = members[i]
        
        for layer in model.layers:
            #layer.trainable = False
            layer._name = 'ensemble_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [ model.output[2] for model in members]

    merge = concatenate(ensemble_outputs)
    dout = Dropout(0.5, name = 'ens_dout1')(merge)
    bnorm = BatchNormalization(name="ens_norm1")(dout)
    
    hidden1 = Dense(round(merge.shape[1]/2), activation=keras.layers.LeakyReLU(alpha=0.01), name = 'vae1.fc')(dout)
    
    output = Dense(1, activation='sigmoid', name='vae.out')(hidden1)
    model = Model(inputs=ensemble_visible, outputs=output, name='stacked_vae')
    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file=ens_pth + 'model_graph.png')
    # compile
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.fit(stackedX, inputy, epoch = 100 )
    
    return model


def fit_stacked_vae(model, input_dict, n_epochs,  es, val_dat):
    # prepare input data
    input_list = list(input_dict.values())
    cleanx_input = map(stackedx_input_prep, input_list)
    X = [x for x in cleanx_input]
    
    cleany_sig = ysig_input_prep(input_list[1])
    class_weights = compute_class_weight(class_weight="balanced",
                        classes = np.unique(cleany_sig.ravel()),
                                        y = cleany_sig.ravel() )
    class_weights = dict(zip(np.unique(cleany_sig.ravel()), class_weights))
    if val_dat != None:
        input_val = list(val_dat.values())
        cleanxv_input = map(stackedx_input_prep, input_val)

        Xv = [x for x in cleanxv_input]
        cleanvy_sig = ysig_input_prep(input_val[1])
        val = (Xv,cleanvy_sig)
    
    else:
        val= None
    
    #val_list = list(val_dict.values())
    #cleanx_val = map(stackedx_input_prep, val_list)
    
    #X_val = [xv for xv in cleanx_val]
    #valy_sig = ysig_input_prep(val_list[1])
    #cleany_smax = ysmax_input_prep(input_list[1])
    #inputy_enc = np.concatenate(N).ravel()
    # fit model
    model.fit(X, cleany_sig, 
              epochs=n_epochs, 
              verbose=0, 
              batch_size = 8, 
             # validation_split= val_split,
              validation_data= val,
              callbacks=[es],
              class_weight=class_weights,
              shuffle=True)
    return model
def fit_stacked_vaeimg(model, input_train, input_test, n_epochs,  es, y_train, y_test):
    # prepare input data
    input_list = list(input_train.values())
    cleanx_input = map(stackedx_img_prep, input_list)
    X = [x for x in cleanx_input]
    
    y_train_list = list(y_train.values())
    cleany_sig = ysig_input_prep(y_train_list[1])

    class_weights = compute_class_weight(class_weight="balanced",
                        classes = np.unique(cleany_sig.ravel()),
                                        y = cleany_sig.ravel() )
    class_weights = dict(zip(np.unique(cleany_sig.ravel()), class_weights))
    if input_test != None:
        input_val = list(input_test.values())
        cleanxv_input = map(stackedx_img_prep, input_val)

        Xv = [x for x in cleanxv_input]
        y_test_list= list(y_test.values())
        cleanvy_sig = ysig_input_prep(y_test_list[1])
        
        val = (Xv,cleanvy_sig)
    
    else:
        val= None
    model.fit(X, cleany_sig, 
              epochs=n_epochs, 
              verbose=0, 
              batch_size = 8, 
             # validation_split= val_split,
              validation_data= val,
              callbacks=[es,keras.callbacks.History()],
              class_weight=class_weights,
              shuffle=True)
    return model
    
    
 
    ###
def predict_stacked_model(model, input_dict):
    # prepare input data
    input_list = list(input_dict.values())
    cleanx_input = map(stackedx_input_prep, input_list)
    X = [x for x in cleanx_input]
    # make prediction
    return model.predict(X, verbose=1)
def stackedx_input_prep(element):
    rawX = element.drop(['y'], axis=1).astype(float)
    inputX = np.expand_dims(rawX, axis=-1)
    return(inputX)
def stackedx_img_prep(element):
    inputX = element
    return(inputX)
def ysig_input_prep(element):
    inputy = element['y']
    # encode output data
    inputy_enc = l_enc.fit_transform(inputy)
    return(inputy_enc)
def ysmax_input_prep(element):
    inputy = np.array(element['y']).reshape(-1,1)
    # encode output data
    inputy_enc = enc.transform(inputy).todense()
    return(inputy_enc)
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
