import os
from .stress_detector import StressDetector, FEATURES
from .dataset import dataset
import numpy as np


def create_dataset(working_dir=os.getcwd(), dictionary_path=None):
    dataset(working_dir=working_dir, dictionary_path=dictionary_path, mode='collect_dataset')


def train_model(working_dir=os.getcwd(), clf=None):
    wav_path = os.path.join(working_dir, 'wav_tg_all')
    sd = StressDetector(wav_path, FEATURES)
    sd.get_features('./data/complete_features.tsv')
    evaluation = sd.train(clf())
    print(f'F1 Score: {np.mean(evaluation["f1"])}')


def extend_dictionary(working_dir=os.getcwd(), dictionary_path='/home/sivh/thesis/zaliznyak.txt'):
    dataset(working_dir=working_dir, dictionary_path=dictionary_path, mode='extend_dictionary')