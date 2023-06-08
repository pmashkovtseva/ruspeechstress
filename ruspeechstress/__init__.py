import os

from .stress_detector import StressDetector, FEATURES
from .dataset import dataset
import numpy as np


def create_dataset(dataset_dir=None, dictionary_path=None):
    dataset(dataset_dir=dataset_dir, dictionary_path=dictionary_path, mode='collect_dataset')


def train_model(dataset_dir=os.getcwd(), features_path='data/complete_features.tsv', clf=None):
    sd = StressDetector(dataset_dir, FEATURES)
    sd.get_features(features_path)
    evaluation = sd.train(clf())
    print(f'F1 Score: {np.mean(evaluation["f1"])}')


def extend_dictionary(dataset_dir=None, dictionary_path=None, classifier='models/classifier_vot.pkl', scaler='models'
                                                                                                             '/scaler'
                                                                                                             '.pkl'):
    dataset(dataset_dir=dataset_dir, dictionary_path=dictionary_path, mode='extend_dictionary', model_path=classifier, scaler_path=scaler)