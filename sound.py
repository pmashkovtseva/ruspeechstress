#!/usr/bin/env python3
# Author: Vera Bernhard // minor changes by Polina Mashkovtseva
# Date: 01.06.2021 // 31.05.2023

import os
import subprocess
import json
import re

from praatio import tgio
from typing import Tuple

import pandas as pd


# Class to process a recording of an isolated word:
# - Detects syllable within a words
# - Extracts duration, pitch, loudness and spectral features per syllabe


# http://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runMAUSGetInventar?LANGUAGE=rus-RU
LANG_SAMPA_VOWELS = ['o~', 'E~', 'e~', 'a~', 'u', 'o', 'i', 'e', 'a', '1', '@U', 'EI', 'aU', 'aI']


class Sound:
    def __init__(self, ortho_word: str, sound_file: str) -> None:
        self.ortho = ortho_word
        if not os.path.isfile(sound_file):
            raise FileNotFoundError(
                f"'{sound_file}' can be acessed, check path")
        else:
            self.sound = os.path.abspath(sound_file)
        self.basename = os.path.basename(sound_file).replace('.wav', '')  # e.g. деньги_1
        self.tg_wav_path = os.path.dirname(self.sound)
        self.nr_syll, self.stress_pos = self.get_stats()

        self.tg = None

        # Prosogram Output
        self.pros_data = None
        self.pros_speaker_data = None

        # Extracted from Webmaus .TextGrid
        self.syll_tier_list = []
        self.vowel_tier_list = []

        self.features = None

        self.time_stamps = {}
        self.stress_index = -1

    def get_stats(self) -> Tuple[int, int]:
        """ extract the position of stress and nr of syllable
            from training_words.tsv

        Raises:
            KeyError: if word does not exist in training_words.tsv

        Returns:
            Tuple[int, int]: number of syllables, position of stress (counted from right to left)
        """

        all_words = pd.read_csv(
            'data/training_words.tsv', sep='\t', header=0)
        row = all_words.loc[all_words['word'] == self.ortho]
        if row.empty:
            raise KeyError(
                f"No stats on '{self.ortho}' found, check spelling of word")
        else:
            return row['nr_syll'].values[0], row['stress_pos'].values[0]

    def preprocess(self) -> None:
        """ opens already existing syllable alignment .TextGrid which must be in same directory as .wav,
                    runs prosogram, improves pitch
        """

        self.run_prosogram()
        self._improve_pitch()
        self._extract_segm_tg()

    def read_in(self) -> None:
        """requires webmaus (syllabification) and prosogramm (prosodic features extraction) already to be ran,
            reads in existing textgrid and prosogram output taken from tg_wav_path"""
        self.tg = self._get_tg_file()
        self._read_prosogram_result()
        self._extract_segm_tg()

    def run_prosogram(self) -> None:
        """ runs Prosogram on .wav and .TextGrid file

            if Prosogram fails with 'Using external segmentation in tier segm'
            strategy 'Automatic: acoustic syllables' is used

        Raises:
            subprocess.CalledProcessError: if the prosogram fallback strategy does not work
        """

        # rename tiers of .TextGrid to fit Prosogram
        self.tg = self._get_tg_file()
        self._rename_segm_tier()

        try:
            subprocess.run(['praat', '--run', 'scripts/praat/run_prosogram.praat',
                            self.sound, '0', '0'],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT,
                           check=True)

            output = subprocess.check_output(['praat', '--run', 'scripts/praat/run_prosogram.praat',
                                              self.sound, '0', '0'])

            if '*** ERROR ***' in output.decode('utf-8'):
                try:
                    subprocess.run(['praat', '--run', 'scripts/praat/run_prosogram.praat',
                                    self.sound, '0', '1'],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.STDOUT,
                                   check=True)
                    print(f"PROSOGRAM:\tsuccessfully run on '{self.basename}'")
                except subprocess.CalledProcessError:
                    print(f"ERROR - PROSOGRAM: fails on '{self.basename}'")
                    pass
            else:
                print(f"PROSOGRAM:\tsuccessfully run on '{self.basename}'")

        except subprocess.CalledProcessError:
            try:
                subprocess.run(['praat', '--run', 'scripts/praat/run_prosogram.praat',
                                self.sound, '0', '1'],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.STDOUT,
                               check=True)
                print(f"PROSOGRAM:\tsuccessfully run on '{self.basename}'")
            except subprocess.CalledProcessError:
                print(f"ERROR - PROSOGRAM: fails on '{self.basename}'")

        finally:
            self._read_prosogram_result()

    def _get_tg_file(self) -> str:
        """ returns the path of the .TextGrid file assuming that it
            it is in same directory as .wav and has same name as .wav

        Returns:
            str: path of .TextGrid path
        """
        return os.path.join(
            self.tg_wav_path, self.basename + '.TextGrid'
        )

    def _rename_segm_tier(self) -> None:
        """ Changes names of TextGrid tiers to fit Prosogram
            using change_tiername.sh
        """
        subprocess.run(['bash', 'scripts/bash/change_tiername.sh',
                       self.tg])

    def _improve_pitch(self) -> None:
        """ Recalculates pitch and reruns Prosogram with new .Pitch file

        Raises:
            subprocess.CalledProcessError: if calculating new .Pitch file fails
            subprocess.CalledProcessError: if rerunning Prosogram with new .Pitch fails
        """
        try:
            subprocess.run(['praat', '--run', 'scripts/praat/stylize_pitch.praat',
                            os.path.dirname(self.sound) + '/', self.basename],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT,
                           check=True)

        except subprocess.CalledProcessError:
            print(f'ERROR - PITCH: stylize pitch fails with {self.basename}')

        try:

            subprocess.run(['praat', '--run', 'scripts/praat/run_prosogram.praat',
                            self.sound, '1', '0'],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.STDOUT,
                           check=True)

        except subprocess.CalledProcessError:
            try:
                subprocess.run(['praat', '--run', 'scripts/praat/run_prosogram.praat',
                                self.sound, '1', '1'],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.STDOUT,
                               check=True)
            except subprocess.CalledProcessError:
                print(f'ERROR - PROSOGRAM: Recalculate pitch fails with {self.basename}')
        # finally:
        #     print(
        #         f"PROSOGRAM:\tsuccessfully recalculated pitch for '{self.basename}'")

    @ property
    def syll_nr_agreeing(self) -> bool:
        """ checks whether prosogram has found the same number of syllables as predicted

        Raises:
            AttributeError: if self.pros_data does not exist, i.e. Prosogram was not run yet or failed

        Returns:
            bool:
        """
        try:
            if self.pros_data.shape[0] == self.nr_syll:
                return True

            else:
                return False

        # in case there is no _data.txt and hence no pros_data.shape
        except AttributeError:
            raise AttributeError(
                f"{self.basename} has not attribute 'pros_data': \n Prosogram not run")

    def _read_prosogram_result(self) -> None:
        """ Reads in _data.txt and _profile_data.txt created by Prosogram

        Raises:
            FileNotFoundError: if no _data.txt exists i.e. Prosogram failed
        """
        try:
            df = pd.read_csv(
                os.path.join(self.tg_wav_path, self.basename + "_data.txt"), sep="\t")
            self.pros_data = df

            df = pd.read_csv(
                os.path.join(self.tg_wav_path, self.basename + "_profile_data.txt"), sep="\t")
            self.pros_speaker_data = df

        # in case that Prosogram has failed and hence there is no _data.txt
        except FileNotFoundError:
            print(f'\nignoring {self.basename} as Prosogram failed to process syllables')

    def _extract_segm_tg(self) -> None:
        """ Extracts segments from .TextGrid (Webmaus forced syllable alignment):
            - syllable intervals
            - vowel intervals
            --> will be needed for backup strategy where Prosogram fails to detect syllables
        """
        tg_open = tgio.openTextgrid(self.tg)
        for interval in tg_open.tierDict['syll'].entryList:
            if interval.label != '<p:>':
                self.syll_tier_list.append(interval)

        for interval in tg_open.tierDict['segm'].entryList:
            if interval.label != '<p:>' and interval.label in LANG_SAMPA_VOWELS:
                self.vowel_tier_list.append(interval)

    def _prepare_features(self) -> None:
        """ generate basic features per syllable
            - pos: position of syllable
            - pros:
                -1: if Prosogram did not represent syllable
                -[0-3]: index of row in Prosogram data where syllable is represented
            - stressd: 1: stressd, 0: unstressed
            - file: .wav filenama, e.g. деньги_1.wav
            - vowel: vowel included in nucles, e.g. u
            - nr_syll: number of syllables
            - start: start time of syllable nucleus
            - end: end time of syllable nucleus
        """
        data = []
        # add whether syllable is stressed or not
        for syll in range(self.nr_syll):
            # check if current syllable is stressed
            if self.nr_syll-self.stress_pos == syll:
                self.stress_index = syll
                data.append(1)
            else:
                data.append(0)
        self.features = pd.DataFrame({'stressed': data})

        self.features['pos'] = self.features.index
        self.features['pros_repr'] = self.features.index
        self.features['pros'] = 1
        self.features['file'] = self.basename
        self.features['vowel'] = [
            vowel_tier.label for vowel_tier in self.vowel_tier_list]
        self.features['nr_syll'] = self.nr_syll

        if not self.syll_nr_agreeing:
            # check which syllable is represented in prosogram output,
            # -1: if no information in prosogram output
            # index: for which line in prosogram output represents syll
            for index, row in self.pros_data.iterrows():
                start_time_nucl = float(row['nucl_t1'])
                end_time_nucl = float(row['nucl_t2'])

                for i, interval in enumerate(self.syll_tier_list):
                    # syllable is detected by Prosogram
                    # --> take time stamps of nucleus from Prosogram
                    if interval.start <= start_time_nucl <= interval.end:
                        self.features.at[i, 'pros_repr'] = index
                        self.features.at[i, 'start'] = start_time_nucl
                        self.features.at[i, 'end'] = end_time_nucl

                    # syllable is not detected by Prosogram
                    # --> take time stamps of nucleus from Webmaus .TextGrid
                    else:
                        self.features.at[i, 'pros_repr'] = -1
                        self.features.at[i,
                                         'start'] = self.vowel_tier_list[i].start
                        self.features.at[i,
                                         'end'] = self.vowel_tier_list[i].end
                        self.features.at[i, 'pros'] = 0
        else:
            for index, row in self.pros_data.iterrows():
                start_time_nucl = float(row['nucl_t1'])
                end_time_nucl = float(row['nucl_t2'])
                self.features.at[index, 'start'] = start_time_nucl
                self.features.at[index, 'end'] = end_time_nucl

        # prepare spectral information
        subprocess.check_output(['praat', '--run', 'scripts/praat/get_intensity_bands.praat',
                                 self.basename+'.wav', self.tg_wav_path])

    def get_features(self) -> pd.DataFrame:
        """ Calculates absolute, normalised and contextual features of all syllables in word
            in the acoustic domains of:
            - Duration
            - Loudness
            - Pitch
            - Spectrum

        Returns:
            pd.DataFrame: returns all features of word as a DataFrame
        """
        self._prepare_features()

        # get absolute features
        for index, row in self.features.iterrows():
            self._get_dur_features(index, row)
            self._get_int_features(index, row)
            self._get_pitch_features(index, row)
            self._get_spect_features(index, row)

        # get normalized features
        self._norm_dur_feat()
        self._norm_int_feat()
        self._norm_pitch_feat()

        # get context-aware features
        for index, row in self.features.iterrows():
            # is not first syllable of word
            if index > 0:
                left_syl = self.features.iloc[index-1]
                self._cont_dur_feat(index, row, 'left', left_syl)
                self._cont_int_feat(index, row, 'left', left_syl)
                self._cont_spect_feat(index, row, 'left', left_syl)
                self._cont_pitch_feat(index, row, 'left', left_syl)

            # is first syllable of word -> has not left neighbour
            else:
                self._cont_dur_feat(index, row, 'left')
                self._cont_int_feat(index, row, 'left')
                self._cont_spect_feat(index, row, 'left')
                self._cont_pitch_feat(index, row, 'left')

            # is not last syllable of word
            if index < self.nr_syll-1:
                right_syl = self.features.iloc[index+1]
                self._cont_dur_feat(index, row, 'right', right_syl)
                self._cont_int_feat(index, row, 'right', right_syl)
                self._cont_spect_feat(index, row, 'right', right_syl)
                self._cont_pitch_feat(index, row, 'right', right_syl)
            # is last syllable of word -> has no right neighbour
            else:
                self._cont_dur_feat(index, row, 'right')
                self._cont_int_feat(index, row, 'right')
                self._cont_spect_feat(index, row, 'right')
                self._cont_pitch_feat(index, row, 'right')

        # remove helper files needed for spectral features
        try:
            os.remove(os.path.join(self.tg_wav_path,
                                   self.basename + '_filt_500.wav'))
            os.remove(os.path.join(self.tg_wav_path,
                                   self.basename + '_filt_1000.wav'))
            os.remove(os.path.join(self.tg_wav_path,
                                   self.basename + '_filt_2000.wav'))
        except FileNotFoundError:
            pass

        return self.features

    def _get_dur_features(self, index: int, row: pd.Series) -> None:
        """ Calculates absolute durational features per syllable:
            - nucl_dur: duration of nucleus
            - syll_dur: duration of syllable

        Args:
            index (int): position of current syllable in word
            row (pd.Series): feature vector of current syllable
        """
        self.features.at[index, 'nucl_dur'] = row['end'] - row['start']

        pros_index = row['pros_repr']
        # Prosogram has detected syllable
        if pros_index > -1:
            # find corresponding line in pros_data
            pros_row = self.pros_data.iloc[pros_index]
            syll_dur = pros_row['syll_dur']
        # Prosogram has not found syllable
        # take duration of syllable from webmaus .TextGrid
        else:
            syll_dur = self.syll_tier_list[index].end - \
                self.syll_tier_list[index].start

        self.features.at[index, 'syll_dur'] = syll_dur

    def _get_int_features(self, index: int, row: pd.Series) -> None:
        """Calculates absolute loudness features per syllable:
            - rms: root mean square amplitude
            - int peak: maximum intensity

        Args:
            index (int): position of current syllable in word
            row (pd.Series): feature vector of current syllable
        """

        pros_index = row['pros_repr']

        # root mean square energy
        rms = subprocess.check_output(
            ['praat', '--run', 'scripts/praat/get_loudness.praat', self.tg_wav_path,
             self.basename, str(row['start']), str(row['end']), '1', '0', '0'])
        rms = float(rms.decode('utf-8').strip(' Pascal\n'))
        self.features.at[index, 'rms'] = rms

        # from prosogram: intensity peak
        if pros_index > -1:
            int_peak = self.pros_data.iloc[pros_index]['int_peak']
            self.features.at[index, 'int_peak'] = int_peak

        # fallback: intensity peak
        else:
            int_peak = subprocess.check_output(
                ['praat', '--run', 'scripts/praat/get_loudness.praat', self.tg_wav_path,
                    self.basename, str(row['start']), str(row['end']), '0', '1', '0'])
            if int_peak.decode('utf-8').strip(' dB\n') == '--undefined--':
                int_peak = 0.0
            else:
                int_peak = float(int_peak.decode('utf-8').strip(' dB\n'))
            self.features.at[index, 'int_peak'] = int_peak

    def _get_spect_features(self, index: int, row: pd.Series) -> None:
        """ Calculates absolute spectral features per syllable:
            - spect_b1: mean intensity between 500-1000 Hz
            - spect_b2: mean intensity between 1000-2000 Hz
            - spect_b3: mean intensity between 2000-4000 Hz

        Args:
            index (int): position of current syllable in word
            row (pd.Series): feature vector of current syllable
        """
        syl_int = subprocess.check_output(
            ['praat', '--run', 'scripts/praat/get_spectral.praat', self.tg_wav_path,
             self.basename, str(row['start']), str(row['end'])])
        syl_int = [float(x) if x != '--undefined--' else 0.0 for x in syl_int.decode('utf-8').split(' dB\n')[:-1]]
        self.features.at[index, 'spect_b1'] = syl_int[0]
        self.features.at[index, 'spect_b2'] = syl_int[1]
        self.features.at[index, 'spect_b3'] = syl_int[2]

    def _get_pitch_features(self, index: int, row: pd.Series) -> None:
        """ Calculates pitch features per syllable:
            - trajectory: sum of absolute pitch interval of tonal segments in nucleus after stylization by Prosogram
            - f0_max: maximal f0 in Hz
            - f0_mean: mean f0 in Hz
            - f0_meanST: mean f0 in emitones
            - hipitch: highest pitch stylized by Prosogram in Hz
            - intersyllab: difference in semitones between end of previous nucleus and start of current one

        Args:
            index (int): position of current syllable in word
            row (pd.Series): feature vector of current syllable
        """
        pros_index = row['pros_repr']

        # Prosogram has detected syllable
        if pros_index > -1:
            trajectory = self.pros_data.iloc[pros_index]['trajectory']
            f0_max = self.pros_data.iloc[pros_index]['f0_max']
            f0_mean = self.pros_data.iloc[pros_index]['f0_mean']

            self.features.at[index, 'trajectory'] = trajectory
            self.features.at[index, 'f0_max'] = f0_max
            self.features.at[index, 'f0_mean'] = f0_mean
            self.features.at[index,
                             'intersyllab'] = self.pros_data.iloc[pros_index]['intersyllab']
            self.features.at[index,
                             'f0_meanST'] = self.pros_data.iloc[pros_index]['f0_meanST']
            self.features.at[index,
                             'f0_max_styl'] = self.pros_data.iloc[pros_index]['hipitch']

        # Prosogram has not found syllable
        else:
            self.features.at[index, 'trajectory'] = 0.0
            self.features.at[index, 'intersyllab'] = 0.0
            self.features.at[index, 'f0_max_styl'] = 0.0

            # Fallback
            f0_output = subprocess.check_output(
                ['praat', '--run', 'scripts/praat/get_pitch.praat', self.tg_wav_path,
                 self.basename, str(row['start']), str(row['end'])])
            f0_output = re.split('\n| Hz| semitones re 1 Hz',
                                 f0_output.decode('utf-8'))

            f0_max, f0_mean, f0_meanST = [x for x in f0_output if x != '']

            try:
                self.features.at[index, 'f0_max'] = float(f0_max)

            except ValueError:
                self.features.at[index, 'f0_max'] = 0.0

            try:
                self.features.at[index, 'f0_mean'] = float(f0_mean)
            except ValueError:
                self.features.at[index, 'f0_mean'] = 0.0

            try:
                self.features.at[index, 'f0_meanST'] = float(f0_meanST)
            except ValueError:
                self.features.at[index, 'f0_meanST'] = 0.0

    def _norm_int_feat(self) -> None:
        """ Normalises the loudness features with
            average peak intensity and root mean square energy
        """
        start_word = str(self.features.iloc[0]['start'])
        end_word = str(self.features.iloc[0]['end'])

        avg_int = subprocess.check_output(
            ['praat', '--run', 'scripts/praat/get_loudness.praat', self.tg_wav_path,
             self.basename, start_word, end_word, '0', '0', '1'])
        if avg_int.decode('utf-8').strip(' dB\n') == '--undefined--':
            avg_int = 0.0
        else:
            avg_int = float(avg_int.decode('utf-8').strip(' dB\n'))
        avg_rms = self.features['rms'].mean()

        self.features['rms_norm'] = self.features['rms']-avg_rms
        self.features['int_peak_norm'] = self.features['int_peak']-avg_int

    def _norm_dur_feat(self) -> None:
        """ Normalises the durational features with
            average nucleus resp. syllable duration and average intrinstic vowel duration
            calculated on all training data
        """
        avg_nucl_dur = self.features['nucl_dur'].mean()
        avg_syll_dur = self.features['syll_dur'].mean()

        # normalise for speech rate
        self.features['nucl_dur_norm'] = self.features['nucl_dur'] / avg_nucl_dur
        self.features['syll_dur_norm'] = self.features['syll_dur'] / avg_syll_dur

        try:
            with open('data/vowel_length.json') as f:
                vowel_length = json.load(f)
                # normalise for intrinsic vowel length
                for index, row in self.features.iterrows():
                    vowel_avg = vowel_length[row['vowel']]
                    if vowel_avg > 0:
                        self.features.at[index,
                                         'nucl_dur_vnorm'] = row['nucl_dur_norm'] / vowel_avg
                    else:
                        self.features.at[index,
                                         'nucl_dur_vnorm'] = row['nucl_dur_norm']
        except FileNotFoundError:
            for index, row in self.features.iterrows():
                self.features.at[index, 'nucl_dur_vnorm'] = 0.00

    def _norm_pitch_feat(self) -> None:
        """ Normalises pitch features with
            average pitch in Hz and Semitonse
        """
        mean_pitch = self.pros_speaker_data.iloc[0]['F0MeanHz']
        mean_pitch_st = self.pros_speaker_data.iloc[0]['F0MeanInST']

        # measures in Hz
        self.features['f0_max_norm'] = self.features['f0_max'] - mean_pitch
        self.features['f0_mean_norm'] = self.features['f0_mean'] - mean_pitch
        self.features['f0_max_styl_norm'] = self.features['f0_max_styl'] - mean_pitch

        # measures in Semitones
        self.features['f0_meanST_norm'] = self.features['f0_meanST'] - \
            mean_pitch_st

    def _cont_dur_feat(self, index: int, row: pd.Series, side: str, neighbour=None) -> None:
        """ Calculates contextual durational features of a syllable
            by taking difference to left or right neighbouring syllable

            if there is no neighbour as it is first or last syllable,
            average of all syllables of word is taking as neighbour value

        Args:
            index (int): position of current syllable in word
            row (pd.Series): feature vector of current syllable
            side (str): {'left'/'right'} is considered
            neighbour ([type], optional): feature vector of neighbouring syllable Defaults to None.
        """
        avg_nucl_norm = self.features['nucl_dur_norm'].mean()
        avg_nucl_vnorm = self.features['nucl_dur_vnorm'].mean()
        avg_syll_norm = self.features['syll_dur_norm'].mean()

        if isinstance(neighbour, pd.Series):
            self.features.at[index,
                             f'nucl_dur_{side}'] = row['nucl_dur_norm'] - neighbour['nucl_dur_norm']
            self.features.at[index,
                             f'nucl_dur_v_{side}'] = row['nucl_dur_vnorm'] - neighbour['nucl_dur_vnorm']
            self.features.at[index,
                             f'syll_dur_{side}'] = row['syll_dur_norm'] - neighbour['syll_dur_norm']

        else:
            self.features.at[index,
                             f'nucl_dur_{side}'] = row['nucl_dur_norm'] - avg_nucl_norm
            self.features.at[index,
                             f'nucl_dur_v_{side}'] = row['nucl_dur_vnorm'] - avg_nucl_vnorm
            self.features.at[index,
                             f'syll_dur_{side}'] = row['syll_dur_norm'] - avg_syll_norm

    def _cont_int_feat(self, index: int, row: pd.Series, side: str, neighbour=None) -> None:
        """ Calculates contextual loudness features of a syllable
            by taking difference to left or right neighbouring syllable

            if there is no neighbour as it is first or last syllable,
            average of all syllables of word is taking as neighbour value

        Args:
            index (int): position of current syllable in word
            row (pd.Series): feature vector of current syllable
            side (str): {'left'/'right'} is considered
            neighbour ([type], optional): feature vector of neighbouring syllable Defaults to None.
        """
        avg_int = self.features['int_peak_norm'].mean()
        avg_rms = self.features['rms_norm'].mean()

        if isinstance(neighbour, pd.Series):
            self.features.at[index,
                             f'int_peak_{side}'] = row['int_peak_norm'] - neighbour['int_peak_norm']

            self.features.at[index,
                             f'rms_{side}'] = row['rms_norm'] - neighbour['rms_norm']

        else:
            self.features.at[index,
                             f'int_peak_{side}'] = row['int_peak_norm'] - avg_int
            self.features.at[index,
                             f'rms_{side}'] = row['rms_norm'] - avg_rms

    def _cont_spect_feat(self, index: int, row: pd.Series, side: str, neighbour=None) -> None:
        """  Calculates contextual spectral features of a syllable
            by taking difference to left or right neighbouring syllable

            if there is no neighbour as it is first or last syllable,
            average of all syllables of word is taking as neighbour value

        Args:
            index (int): position of current syllable in word
            row (pd.Series): feature vector of current syllable
            side (str): {'left'/'right'} is considered
            neighbour ([type], optional): feature vector of neighbouring syllable Defaults to None.
        """
        avg_b1 = self.features['spect_b1'].mean()
        avg_b2 = self.features['spect_b2'].mean()
        avg_b3 = self.features['spect_b3'].mean()

        if isinstance(neighbour, pd.Series):
            self.features.at[index,
                             f'spect_b1_{side}'] = row['spect_b1'] - neighbour['spect_b1']
            self.features.at[index,
                             f'spect_b2_{side}'] = row['spect_b2'] - neighbour['spect_b2']
            self.features.at[index,
                             f'spect_b3_{side}'] = row['spect_b3'] - neighbour['spect_b3']

        else:
            self.features.at[index,
                             f'spect_b1_{side}'] = row['spect_b1'] - avg_b1
            self.features.at[index,
                             f'spect_b2_{side}'] = row['spect_b2'] - avg_b2
            self.features.at[index,
                             f'spect_b3_{side}'] = row['spect_b3'] - avg_b3

    def _cont_pitch_feat(self, index: int, row: pd.Series, side: str, neighbour=None) -> None:
        """  Calculates contextual pitch features of a syllable
            by taking difference to left or right neighbouring syllable

            if there is no neighbour as it is first or last syllable,
            average of all syllables of word is taking as neighbour value

        Args:
            index (int): position of current syllable in word
            row (pd.Series): feature vector of current syllable
            side (str): {'left'/'right'} is considered
            neighbour ([type], optional): feature vector of neighbouring syllable Defaults to None.
        """
        avg_f0_max = self.features['f0_max_norm'].mean()
        avg_f0_mean = self.features['f0_mean_norm'].mean()
        avg_f0_max_styl = self.features['f0_max_styl_norm'].mean()
        avg_f0_meanST = self.features['f0_meanST_norm'].mean()

        if isinstance(neighbour, pd.Series):
            self.features.at[index,
                             f'f0_max_{side}'] = row['f0_max_norm'] - neighbour['f0_max_norm']
            self.features.at[index,
                             f'f0_mean_{side}'] = row['f0_mean_norm'] - neighbour['f0_mean_norm']
            self.features.at[index,
                             f'f0_max_styl_{side}'] = row['f0_max_styl_norm'] - neighbour['f0_max_styl_norm']
            self.features.at[index,
                             f'f0_meanST_{side}'] = row['f0_meanST_norm'] - neighbour['f0_meanST_norm']
        else:
            self.features.at[index,
                             f'f0_max_{side}'] = row['f0_max_norm'] - avg_f0_max
            self.features.at[index,
                             f'f0_mean_{side}'] = row['f0_mean_norm'] - avg_f0_mean
            self.features.at[index,
                             f'f0_max_styl_{side}'] = row['f0_max_styl_norm'] - avg_f0_max_styl
            self.features.at[index,
                             f'f0_meanST_{side}'] = row['f0_meanST_norm'] - avg_f0_meanST
