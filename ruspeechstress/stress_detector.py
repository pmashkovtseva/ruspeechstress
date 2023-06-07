#!/usr/bin/env python3
# Author: Vera Bernhard // minor changes by Polina Mashkovtseva
# Date: 21.06.2021 // 25.05.2023

import os
import json

import joblib
import pandas as pd
import numpy as np
from pandas import DataFrame

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, f1_score
from sklearn.model_selection import ShuffleSplit, GridSearchCV
from sklearn.preprocessing import StandardScaler

from typing import Tuple, Dict, List

from .utils import plot_confusion_matrix
from tqdm import tqdm
from .sound import Sound, LANG_SAMPA_VOWELS

# selected features after feature engineering
FEATURES = ['pos', 'nr_syll', 'nucl_dur', 'syll_dur', 'syll_dur_norm',  'nucl_dur_left', 'nucl_dur_right', 'rms',
            'int_peak', 'rms_norm', 'int_peak_norm', 'rms_left', 'rms_right', 'int_peak_left', 'int_peak_right',
            'spect_b1', 'spect_b2', 'spect_b3', 'spect_b1_left', 'spect_b2_left', 'spect_b3_left', 'spect_b1_right',
            'spect_b2_right', 'spect_b3_right', 'trajectory', 'f0_max', 'f0_mean', 'f0_meanST', 'f0_max_styl',
            'f0_max_norm', 'f0_mean_norm', 'f0_max_styl_norm', 'f0_meanST_norm', 'intersyllab', 'f0_max_left',
            'f0_max_right', 'f0_mean_left', 'f0_mean_right', 'f0_max_styl_left', 'f0_max_styl_right', 'f0_meanST_left',
            'f0_meanST_right']


# Class to:
# - extract prosodic features on all recordings single words within a directory
# - train and evaluate sklearn classifiers on the resulted data
# - classify a single recording with trained classifier

class StressDetector:

    def __init__(self, tg_wav_path: str, features: List) -> None:
        self.tg_wav_path = tg_wav_path
        self.sound_list = []
        self.features = features
        self.wav_files = self.get_files_dir(self.tg_wav_path, 'wav')
        self.vowels_length = {v: [] for v in LANG_SAMPA_VOWELS}
        self.features_list = features
        self.scaler = None
        self.classifier = None

    def preprocess(self) -> None:
        """ Runs following preprocessing steps on all .wav files in self.tg_wav_path
            opens already existing syllable alignment .TextGrid which must be in same directory as .wav,
            runs prosogram, improves pitch

        Attention: Expects all .wav to be named in form деньги_1.wav (word_index.wav)

        """
        bar = tqdm(self.wav_files)
        for f in bar:
            try:
                ortho = os.path.basename(f).split('_')[0]
                s = Sound(ortho, f)
                bar.set_description(f'Preprocessing {s.basename.ljust(30)}')
                s.preprocess()
                if s.pros_data is not None:
                    self.sound_list.append(s)
            except IndexError:
                print(f"SKIPPED: file name '{os.path.basename(f)} is not of form word_index.wav")

    def read_in(self) -> None:
        """
            requires webmaus (syllabification) and prosogramm (prosodic features extraction) already to be ran,
            reads in existing textgrid and prosogram output for all files in tg_wav_path
        """
        bar = tqdm(self.wav_files)
        for f in bar:
            try:
                ortho = os.path.basename(f).split('_')[0]
                s = Sound(ortho, f)
                bar.set_description(f'Reading in {s.basename.ljust(30)}')
                s.read_in()
                if s.pros_data is not None:
                    self.sound_list.append(s)

            except IndexError:
                print(
                    f"SKIPPED: file name '{os.path.basename(f)} is not of form word_index.wav")

    def load_classifier(self, classifier: str, scaler: str) -> None:
        """loads the pretrained model and classifier

        Args:
            classifier (str): path to model
            scaler (str): path to scaler
        """
        self.classifier = joblib.load(classifier)
        self.scaler = joblib.load(scaler)

    @staticmethod
    def get_files_dir(path: str, type: str) -> List:
        """returns of a list of all files of a certain type e.g. .wav in a given direcotry

        Args:
            path (str): directory
            type (str): type, e.g. .wav

        Returns:
            List: files of type
        """
        return [os.path.join(path, f) for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and f.endswith(type)]

    def get_features(self, tsv_file: str = None) -> pd.DataFrame:
        """ Calculates all features 

        Args:
            tsv_file (str, optional): file with all features of all data. Defaults to None.

        Returns:
            pd.DataFrame: all features 
        """
        if not tsv_file:
            self.features = pd.DataFrame()
            bar = tqdm(self.sound_list)
            for sound in bar:
                bar.set_description(
                    f'Feature extraction {sound.basename.ljust(30)}')
                self.features = self.features.append(
                    sound.get_features(), ignore_index=True)
            self.features.fillna(0, inplace=True)
        else:
            self.features = pd.read_csv(tsv_file, sep='\t', header=0)
            print(f"Successfully read in all features from '{tsv_file}'")
        return self.features

    def get_vowel_length(self, file_name: str) -> None:
        """ Caculates average vowel length on entire training data
            which is then used for duration normalisation

            Attention: has to be run after get_features()
        """
        for sound in self.sound_list:
            for index, row in sound.features.iterrows():
                vowel = sound.vowel_tier_list[index].label
                self.vowels_length[vowel].append(row['nucl_dur_norm'])
        for vowel, dur_list in self.vowels_length.items():
            if dur_list:
                self.vowels_length[vowel] = np.mean(dur_list)
            else:
                self.vowels_length[vowel] = 0.0
        with open(file_name, 'w', encoding='utf-8') as outfile:
            json.dump(self.vowels_length, outfile)

    def agree_syllable_nr(self) -> int:
        """ Counts the number of files where Prosogram detected all syllables

        Returns:
            int: nr of files where Prosogram found all syllables
        """
        syll_agree_c = 0
        for s in self.sound_list:
            if s.syll_nr_agreeing:
                syll_agree_c += 1
            else:
                print(f'SYLL DISAGREE {s.basename}')
        return syll_agree_c

    def test_features(self, classifier, name: str, features: List, outfile: str = None) -> Dict:
        """ Trains a given classifier with one feature of feature list at a time
            and saves f1 scores for each model in .tsv format

        Args:
            classifier: sklearn classifier
            name (str): name of classifier
            features (List): list of features
            outfile (str): name of outilfe

        Returns:
            Dict: dict with f1 score per feature
        """
        avg_performance = pd.DataFrame({'feature': features})
        with tqdm(total=len(features)) as pbar:
            for feat in features:
                pbar.set_description(f'Training {name} with {feat}')
                performance_dict = self.train(classifier, [feat])
                i = avg_performance.index[avg_performance['feature']
                                          == feat]
                avg_performance.at[i, 'f1'] = np.mean(performance_dict['f1'])
                pbar.update()
        if outfile:
            avg_performance.to_csv(outfile, sep='\t')

        else:
            return avg_performance

    def test_feature_groups(self, classifier, name: str, feature_groups: Dict, outfile: str = None) -> DataFrame:
        """ Trains a given classifier with each feature group of feature_groups at a time
            and saves f1 scores for each model in .tsv format

        Args:
            classifier ([type]): sklearn classifier
            name (str): name of classifier
            feature_groups (Dict): list of feature groups
            outfile (str): name of outilfe

        Returns:
            Dict: dict with f1 score per feature group
        """
        avg_performance = pd.DataFrame({'feature': feature_groups.keys()})
        with tqdm(total=len(feature_groups)) as pbar:
            for feat_group, features in feature_groups.items():
                pbar.set_description(f'Training {name} with {feat_group}')
                performance_dict = self.train(classifier, features)
                i = avg_performance.index[avg_performance['feature']
                                          == feat_group]
                avg_performance.at[i, 'f1'] = np.mean(performance_dict['f1'])
                pbar.update()
        if outfile:
            avg_performance.to_csv(outfile, sep='\t')
        else:
            return avg_performance

    def test_classifiers(self, classifiers: List, names: List, outfile=None, predict_post=True) -> DataFrame:
        """ Trains a list of classifiers with the features saved within this class

        Args:
            classifiers (List): list of sklearn classifiers
            names (List): names of sklearn classifiers
            outfile ([type], optional): if given, writes average F1 score per classifier in fiel. Defaults to None.
            predict_post (bool, optional): if set true, uses postprocessing to predict. Defaults to True.

        Returns:
            [type]: [description]
        """
        avg_performance = pd.DataFrame({'classifier': names})

        for name, clf in zip(names, classifiers):
            print(f'Currently training {name}')
            if predict_post:
                performance_dict = self.train(clf)
            else:
                performance_dict = self.train(
                    clf, predict_post=False)
            i = avg_performance.index[avg_performance['classifier'] == name]
            avg_performance.at[i, 'f1'] = np.mean(performance_dict['f1'])
            avg_performance.at[i, 'accuracy'] = np.mean(
                performance_dict['accuracy'])
            avg_performance.at[i, 'precision_0'] = np.mean(
                performance_dict['precision'][0])
            avg_performance.at[i, 'precision_1'] = np.mean(
                performance_dict['precision'][1])
            avg_performance.at[i, 'recall_0'] = np.mean(
                performance_dict['recall'][0])
            avg_performance.at[i, 'recall_1'] = np.mean(
                performance_dict['recall'][1])

        if outfile:
            avg_performance.to_csv(outfile, sep='\t')

        else:
            return avg_performance

    def train(self, clf, features: List = None, predict_post=True, matrix=False) -> Dict:
        """ Trains a given classifier on the feature list and returns
            different performance metrics

        Args:
            features (List): list of features to train
            clf: skleanr classifier
            predict_post (bool, optional): if set true, uses postprocessing to predict. Defaults to True.
            matrix (bool, optional): if set true, produces a normalized confusion matrix. Defaults to False.

        Returns:
            Dict: list of performance metric (precision, recall, accuracy, f1) for each fold of cross-validation
        """
        if not features:
            features = self.features_list
        performance_dict = {'precision': [[], []], 'recall': [[], []], 'accuracy': [], 'f1': []}

        total_true = []
        total_pred = []

        for train, test, test_files in self._split_dataset():
            d_train = train[features]
            t_train = train['stressed']
            if len(features) > 1:
                scaler = StandardScaler().fit(d_train)
                d_train = scaler.transform(d_train)
            clf.fit(d_train, t_train)
            test_true = []
            test_pred = []
            test_pred_proba = []

            for file in test_files:
                # returns all syllables of a word
                word = self.features.loc[self.features['file'] == file]
                test_true += word['stressed'].tolist()
                total_true += word['stressed'].tolist()
                word = word[features]
                if len(features) > 1:
                    word = scaler.transform(word)
                if predict_post:
                    probs, classes, _ = self.predict_post(clf, word)
                else:
                    probs, classes = self.predict(clf, word)
                test_pred += classes
                total_pred += classes
                test_pred_proba += probs

            accuracy = accuracy_score(test_true, test_pred)
            precision, recall, _, _ = precision_recall_fscore_support(
                test_true, test_pred)
            f1 = f1_score(test_true, test_pred)

            performance_dict['accuracy'].append(accuracy)
            performance_dict['f1'].append(f1)
            performance_dict['precision'][0].append(precision[0])
            performance_dict['recall'][0].append(recall[0])
            performance_dict['precision'][1].append(precision[1])
            performance_dict['recall'][1].append(recall[1])

        if matrix:
            plot_confusion_matrix(total_pred, total_true, [
                'unstressed', 'stressed'], 'conf.png')

        return performance_dict

    def grid_search(self, pipelines: List, params: List[Dict]) -> List[Dict]:
        """Peforms gridsearch on given classifiers and params list

        Args:
            pipelines (List): list of sklearn pipelines/classifiers
            params (List[Dict]): list of dicts with different hyperparameters

        Returns:
            List[Dict]: list of best best hyperparameters
        """
        best_params = []
        for i in range(len(pipelines)):
            grid = GridSearchCV(
                estimator=pipelines[i],
                param_grid=params[i],
                n_jobs=-1,  # uses all possible processor
                scoring='f1',
                cv=10,  # cross-validation
                verbose=2
            ).fit(self.features[self.features_list], self.features['stressed'])

            best_params.append((grid.best_score_, grid.best_params_))

        return best_params

    def _split_dataset(self):
        """Splits the training data 10-fold cross validated into 25/75
        while it makes sure that syllables of the same word """
        files = pd.unique(self.features['file'])
        rs = ShuffleSplit(test_size=0.25, n_splits=10, random_state=42)
        for train_index, test_index in rs.split(files):
            train_files = files[np.array(train_index)]
            test_files = files[np.array(test_index)]
            train = self.features.loc[self.features['file'].isin(
                train_files)]
            test = self.features.loc[self.features['file'].isin(
                test_files)]
            yield train, test, test_files

    def train_all(self, classifier, name: str, save=False) -> None:
        """ trains classifier on all training data,
            saves resulting scaler and model within class

        Args:
            classifier: classifier
            name (str): name of classifier
            save (bool, optional): if given: pickels model. Defaults to False.
        """

        train = self.features[self.features_list]
        target = self.features['stressed']
        scaler = StandardScaler().fit(train)
        train_scaled = scaler.transform(train)
        print(f'Currently Training {name} on all data')
        clf = classifier.fit(train_scaled, target)

        self.scaler = scaler
        self.classifier = clf
        self.clf_name = name

        if save:
            joblib.dump(scaler, 'models/scaler.pkl')
            joblib.dump(clf, f'models/classifier_{name}.pkl')

    def predict_post(self, clf, word: np.array) -> Tuple[List, List]:
        """ Classifies each syllable in a word, taking into account that there
            can only be one stress per word

        Args:
            clf ([type]): classifier
            word (np.array): word represent with feature vectors

        Returns:
            Tuple[List, List]: array of classification
            probabilites and an array of classes
        """

        # get probability for stressed syllable only
        classes = clf.predict(word).tolist()
        probs = clf.predict_proba(word)[:, 1].tolist()
        nr_stressed_syl = classes.count(1)
        # more or less than 1 stressed syllable detected
        if nr_stressed_syl != 1:
            # no stressed syllable found
            # --> chose syllable with highest probability to be stressed
            if nr_stressed_syl == 0:
                pred_stress_pos = np.argmax(probs)
                classes[pred_stress_pos] = 1

            # too many stressed syllables
            # --> chose of stressed syllabe the one with highest probability to be stressed
            if nr_stressed_syl > 1:
                max_prob = 0
                pred_stress_pos = 0
                syll = 0
                for cla, prob in zip(classes, probs):
                    if cla == 1 and prob > max_prob:
                        max_prob = prob
                        pred_stress_pos = syll
                    syll += 1
                classes = [0] * len(probs)
                classes[pred_stress_pos] = 1
            return probs, classes, pred_stress_pos

        # exactly 1 stressed syllable detected
        else:
            return probs, classes, classes.index(1)

    def predict(self, clf, word: np.array) -> Tuple[List, List]:
        """ predicts stress pattern of a word without post_processing

        Args:
            clf: classifier
            word (np.array): word respresented as feature vectors

        Returns:
            Tuple[List, List]: (probilities for stressed, stress pattern)
        """
        probs = clf.predict_proba(
            word)[:, 1].tolist()
        classes = clf.predict(word).tolist()

        return probs, classes

    def classify(self, wav_file: str, word: str) -> List:
        """

        Args:
            wav_file (str): path to audio file
            word (str): orthographic transcription of word
            feedback (bool, optional): If given: prints feedback. Defaults to False.

        Returns:
            List: (predicted stress pattern)
        """
        sound = Sound(word, wav_file)
        sound.preprocess()
        all_features = sound.get_features()
        feat_scaled = self.scaler.transform(all_features[self.features_list])
        _, classes, pred_stress_pos = self.predict_post(
            self.classifier, feat_scaled)

        pred = classes

        return pred


def main():
    pass


if __name__ == "__main__":
    main()
