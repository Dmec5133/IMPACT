from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Input, Dense, Dropout, LeakyReLU, AveragePooling1D, BatchNormalization
from keras.models import Model
from keras.layers.merge import concatenate
from keras.utils.vis_utils import plot_model
from sklearn import preprocessing
from sklearn.utils import class_weight
from sklearn.utils import compute_class_weight
import numpy as np
from sklearn import preprocessing
l_enc = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()

def stackedx_input_prep(element):
    rawX = element.drop(['y'], axis=1).astype(float)
    inputX = np.expand_dims(rawX, axis=-1)
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


def define_stacked_model(members, opt, ens_pth):
    
    for i in range(len(members)):
        model = members[i]
        
        for layer in model.layers:
            #layer.trainable = False
            layer._name = 'ensemble_' + str(i+1) + '_' + layer.name
    # define multi-headed input
    ensemble_visible = [model.input for model in members]
    # concatenate merge output from each model
    ensemble_outputs = [model.output for model in members]
    merge = concatenate(ensemble_outputs)
    dout = Dropout(0.5)(merge)
    bnorm = BatchNormalization(name="decoder_norm_1")(dout)
    #dout = Dropout(0.2)(bnorm)
    #print(merge.shape)
    hidden = Dense(round(merge.shape[1]/2), activation=keras.layers.LeakyReLU(alpha=0.01))(bnorm)
    #bnorm2 = BatchNormalization(name="decoder_norm_2")(hidden)
    #dout2 = Dropout(0.2)(hidden)
    output = Dense(1, activation='sigmoid')(hidden)
    model = Model(inputs=ensemble_visible, outputs=output)
    # plot graph of ensemble
    plot_model(model, show_shapes=True, to_file=ens_pth + 'model_graph.png')
    # compile
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    #model.fit(stackedX, inputy, epoch = 100 )
    
    return model


def fit_stacked_model(model, input_dict, n_epochs, es, val_dat):
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
              verbose=2, 
              batch_size = 8, 
             # validation_split= val_split,
              validation_data= val,
              callbacks=[es],
              class_weight=class_weights,
              shuffle=True,)
    return model
    
 
    ###
def predict_stacked_model(model, input_dict):
    # prepare input data
    input_list = list(input_dict.values())
    cleanx_input = map(stackedx_input_prep, input_list)
    X = [x for x in cleanx_input]
    # make prediction
    return model.predict(X, verbose=1)