import tensorflow
from tensorflow import keras
from keras.models import load_model
def load_all_models(n_models,k, in_dict):
    all_models = list()
    for k2,v2 in in_dict.items():
    # define filename for this ensemble
        filename = 'phylum_models/' + str(k) +  '/model_' + str(k2) + '.tf'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models
def load_stacked_model(name1):
    filename = 'models/' +  str(name1) + '.tf'
    # load model from file
    model = load_model(filename)
    # add to list of members
    print('>loaded %s' % filename)
    return model
def load_stacked_vae(name1):
    filename = 'modelsvae/' +  str(name1) + '.tf'
    # load model from file
    model = load_model(filename)
    # add to list of members
    print('>loaded %s' % filename)
    return model
