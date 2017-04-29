import os
import pickle 
from SETTINGS import DATA_DIR

FEATURES_DIR = os.path.join(DATA_DIR, 'features')

def get_train_features():
    features_fname = 'verifiabilityNumFeatures_len_5_liwc_train_20comp'
    return get_features(os.path.join(FEATURES_DIR, features_fname))

def get_test_features():
    features_fname = 'verifiabilityNumFeatures_len_5_liwc_test_20comp'
    return get_features(os.path.join(FEATURES_DIR, features_fname))

def get_features(path):
    return pickle.load(open(path, 'rb'))

