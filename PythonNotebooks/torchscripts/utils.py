import pickle 
import torchvision.transforms as transforms
import re

def clean_index(index):
    return re.sub('[^a-zA-Z0-9]', '', index)

def save_picke(obj, img_size):
    fname = "image_transformer"+str(img_size)
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

