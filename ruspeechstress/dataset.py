import math
import os
import re
import subprocess

import whisper
from tqdm import tqdm
from os import listdir
from pydub import AudioSegment
import wave
import requests
import xml.etree.ElementTree as Et
import csv
import textgrid
import rusyllab
from .stress_detector import StressDetector, FEATURES


class FileSizeError(Exception):
    pass


class NoDictProvided(Exception):
    pass


def clear_old_files(working_dir, directory='wav_tg_all'):
    if not os.path.exists(os.path.join(working_dir, directory)):
        pass
    else:
        files = os.listdir(os.path.join(working_dir, directory))
        for file in files:
            os.remove(os.path.join(os.path.join(working_dir, directory), file))


def get_files(working_dir):
    file_pairs = {}
    for file in listdir(working_dir):
        text = os.path.join(working_dir, file[:-4] + '.txt')
        wav = os.path.join(working_dir, file)
        if file.endswith('.wav') and os.path.exists(text):
            file_pairs.update({wav: text})
        elif file.endswith('.wav') and not os.path.exists(text):
            file_pairs.update({wav: None})
    return file_pairs


def preprocess_audio(input_files):
    for wav in input_files.keys():
        with wave.open(wav, "rb") as wave_file:
            frame_rate = wave_file.getframerate()
        sound = AudioSegment.from_file(wav, format='wav', frame_rate=frame_rate)
        sound.set_channels(1).set_frame_rate(16000).export(wav, format='wav')


def get_size(input_files):
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    for wav, text in input_files.items():
        file_stats = os.stat(wav)
        if file_stats.st_size == 0:
            return "0B"
        i = int(math.floor(math.log(file_stats.st_size, 1024)))
        p = math.pow(1024, i)
        s = round(file_stats.st_size / p, 2)
        with open(text, 'r', encoding='utf-8') as t:
            transcription = t.read().split()
        if s > 2 and size_name[i] == "GB":
            print(f'Your input file {wav} is too big for processing (>2GB).')
            raise FileSizeError
        if len(transcription) > 3000:
            print(f'Your input file {text} is too big for processing (>3000 words).')
            raise FileSizeError


def transcribe_audio(input_files):
    for wav in input_files.keys():
        if input_files[wav] is None:
            model = whisper.load_model("base")
            result = model.transcribe(wav)
            with open(wav[:-4] + ".txt", "w", encoding="utf-8") as o:
                o.write(result["text"].lower())


def preprocess_text(input_files):
    for text_file in input_files.values():
        with open(text_file, 'r', encoding='utf-8') as tr:
            text = tr.read()
        capitalized = {}
        for i in range(len(text.split())):
            if text.split()[i - 1][-1] not in '.?!…' and text.split()[i][0].isupper():
                capitalized.update({i: text.split()[i].strip('.?!…')})
        text_clear = []
        for i in range(len(text.split())):
            word_to_compare = text.split()[i].strip('.?!…').capitalize()
            if i in capitalized.keys() and word_to_compare == capitalized[i]:
                text_clear.append(word_to_compare)
            else:
                text_clear.append(text.split()[i].lower())
        with open(text_file, 'w', encoding='utf-8') as nw:
            nw.write(' '.join(text_clear))


def webmaus(input_files, out_path):
    url = "https://clarin.phonetik.uni-muenchen.de/BASWebServices/services/runPipeline"
    for wav, text in tqdm(input_files.items()):
        files = {
            'TEXT': open(text, 'rb'),
            'PIPE': (None, 'G2P_MAUS_PHO2SYL'),
            'LANGUAGE': (None, 'rus-RU'),
            'SIGNAL': open(wav, 'rb'),
            'OUTFORMAT': (None, 'TextGrid'),
        }
        res = requests.post(url, files=files)
        if res.status_code == 200:
            tree = Et.fromstring(res.text)
            res2 = requests.get(tree.find('downloadLink').text)
            open(out_path + os.path.basename(wav)[:-4] + '.TextGrid', 'wb').write(
                res2.content)


def build_dictionary(dictionary_path):
    with open(dictionary_path, 'r', encoding='utf-8') as f:
        paradigms = f.read().splitlines()
    stressed_dict = {}
    for paradigm in paradigms:
        if '#' in paradigm:
            paradigm = paradigm.replace('#', ',')
        for word in paradigm.split(','):
            stressed_dict.update({word.replace("'", "").replace("`", ""): word})
    return stressed_dict


def get_number_of_syllables(word):
    s = list(word)
    chunks = []
    for i in range(len(s)):
        if s[i] in 'уеэоаыяиёУЕЭОАЫЯИЁ' and i + 1 < len(s):
            chunks.append(s[i] + s[i + 1])
        elif s[i] in 'уеэоаыяиёУЕЭОАЫЯИЁ':
            chunks.append(s[i])
    stress_position = 0
    for i in range(len(chunks)):
        if "'" in chunks[i]:
            stress_position = i + 1
    if stress_position != 0:
        return stress_position


def collect_dataset(working_dir, input_files, dictionary, output_path, mode='collect_dataset'):
    n = 0
    all_data = []
    words_all = []
    words_dict = []
    words_new = []
    number_of_word_recordings_all = 0
    number_of_word_recordings_dict = 0
    number_of_word_recordings_new = 0
    for wav in input_files.keys():
        if os.path.exists(wav[:-4] + '.TextGrid'):
            transcription = textgrid.TextGrid.fromFile(wav[:-4] + '.TextGrid')
            for i in range(len(transcription[0])):
                word = transcription[0][i]
                if mode == 'collect_dataset':
                    if word.mark != "":
                        number_of_word_recordings_all += 1
                        if word.mark not in words_all:
                            words_all.append(word.mark)
                    if word.mark != "" and word.mark in dictionary.keys() and len(
                            transcription[2][i].mark.split('.')) > 1 \
                            and get_number_of_syllables(dictionary[word.mark]) is not None:
                        nr_syll = str(len(transcription[2][i].mark.split('.')))
                        stress_pos = str(len(transcription[2][i].mark.split('.')) - get_number_of_syllables(
                            dictionary[word.mark]) + 1)
                        if word.mark not in words_dict:
                            all_data.append([str(n), word.mark, stress_pos, nr_syll])
                            words_dict.append(word.mark)
                            n += 1
                        start_time = word.minTime * 1000
                        end_time = word.maxTime * 1000
                        sound = AudioSegment.from_file(wav, format='wav', frame_rate=16000)
                        extract = sound[start_time:end_time]
                        counter_wav = 1
                        output_filename = word.mark + '_' + str(counter_wav) + '.wav'
                        output_wav = os.path.join(os.path.join(working_dir, output_path), output_filename)
                        while os.path.exists(output_wav):
                            output_wav = output_wav.replace(f'_{counter_wav}.', f'_{counter_wav + 1}.')
                            counter_wav += 1
                        extract.export(output_wav, format='wav')
                        number_of_word_recordings_dict += 1
                elif mode == 'extend_dictionary':
                    if word.mark != "":
                        number_of_word_recordings_all += 1
                        if word.mark not in words_all:
                            words_all.append(word.mark)
                    if word.mark in dictionary.keys() and word.mark not in words_dict:
                        words_dict.append(word.mark)
                    elif word.mark != "" and word.mark not in ['обо', 'изо', 'подо',
                                                               'нибудь'] and word.mark not in dictionary.keys(
                    ) and re.search(r'у|е|э|о|а|ы|я|и|ё|У|Е|Э|О|А|Ы|Я|И|Ё', word.mark):
                        nr_syll = str(len(transcription[2][i].mark.split('.')))
                        stress_pos = 0
                        if word.mark not in words_new:
                            all_data.append([str(n), word.mark, stress_pos, nr_syll])
                            words_new.append(word.mark)
                            n += 1
                        start_time = word.minTime * 1000
                        end_time = word.maxTime * 1000
                        sound = AudioSegment.from_file(wav, format='wav', frame_rate=16000)
                        extract = sound[start_time:end_time]
                        counter_wav = 1
                        output_filename = word.mark + '_' + str(counter_wav) + '.wav'
                        output_wav = os.path.join(os.path.join(working_dir, output_path), output_filename)
                        while os.path.exists(output_wav):
                            output_wav = output_wav.replace(f'_{counter_wav}.', f'_{counter_wav + 1}.')
                            counter_wav += 1
                        extract.export(output_wav, format='wav')
                        number_of_word_recordings_new += 1
    with open('data/training_words.tsv', 'w', newline='') as tw:
        tsv_output = csv.writer(tw, delimiter='\t')
        tsv_output.writerow(['', 'word', 'stress_pos', 'nr_syll'])
        for row in all_data:
            tsv_output.writerow(row)
    return number_of_word_recordings_all, number_of_word_recordings_dict, number_of_word_recordings_new, len(
        words_all), len(words_dict), len(words_new)


def get_words(working_dir, out_path):
    subprocess.call(["Rscript", "scripts/get_words.R", working_dir + '/', out_path + '/'])


def clear_files(input_path):
    for file in os.listdir(input_path):
        wav_file = file.strip('.TextGrid') + '.wav'
        if file.endswith('.TextGrid') and not os.path.exists(
                os.path.join(input_path, wav_file)):
            os.remove(os.path.join(input_path, file))


def dataset(dataset_dir, dictionary_path=None, mode=None, model_path=None, scaler_path=None):
    working_dir = os.path.dirname(dataset_dir)
    out_path = os.path.join(dataset_dir, 'wav_tg_all')
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    clear_old_files(dataset_dir)
    print("Getting files and preprocessing audio and text...")
    files = get_files(dataset_dir)
    preprocess_audio(files)
    get_size(files)
    transcribe_audio(files)
    files = get_files(dataset_dir)
    preprocess_text(files)
    print("Creating TextGrid files...")
    # webmaus(files, working_dir)
    if dictionary_path is None:
        raise NoDictProvided('No dictionary provided.')
    dictionary = build_dictionary(dictionary_path)
    print("Collecting dataset...")
    if mode == 'collect_dataset':
        recordings_all, recordings_dict, recordings_new, words_all, words_dict, words_new = collect_dataset(working_dir,
                                                                                                            files,
                                                                                                            dictionary,
                                                                                                            out_path,
                                                                                                            mode='collect_dataset')
        print(recordings_all, recordings_dict, recordings_new, words_all, words_dict, words_new)
    elif mode == 'extend_dictionary':
        recordings_all, recordings_dict, recordings_new, words_all, words_dict, words_new = collect_dataset(working_dir,
                                                                                                            files,
                                                                                                            dictionary,
                                                                                                            out_path,
                                                                                                            mode='extend_dictionary')
    get_words(dataset_dir, out_path)
    clear_files(out_path)
    sd = StressDetector(out_path, FEATURES)
    if mode == 'collect_dataset':
        sd.preprocess()
        sd.get_features().to_csv('data/complete_features.tsv', sep='\t')
        with open('data/complete_features.tsv') as cf:
            number_of_syllables = len(cf.read().split('\n'))
        print(f"Dataset creation is finished. Out of {recordings_all} recordings of {words_all} words,"
              f" {recordings_dict} recordings of {words_dict} words were processed. {number_of_syllables - 2} syllables"
              f" were"
              f" extracted.")
    elif mode == 'extend_dictionary':
        sd.load_classifier(model_path, scaler_path)
        words_auto = 0
        words_to_add = []
        lines_to_add = []
        for file in tqdm(os.listdir(out_path)):
            if file.endswith('.wav'):
                word_ortho = os.path.basename(file).split('_')[0]
                syllables = rusyllab.split_words(f"{word_ortho}".split())
                if 'ё' in word_ortho and word_ortho not in words_to_add:
                    words_to_add.append(''.join(word_ortho.replace('ё', 'ё' + "'")))
                    words_auto += 1
                elif len(syllables) == 1:
                    for i in list(word_ortho):
                        if i in 'уеэоаыяиёУЕЭОАЫЯИЁ':
                            words_to_add.append(''.join(word_ortho.replace(i, i + "'")))
                else:
                    stresses = sd.classify(os.path.join(out_path, file), word_ortho)
                    word_for_dict = []
                    for syllable, result in zip(syllables, stresses):
                        if result == 1:
                            for i in list(syllable):
                                if i in 'уеэоаыяиёУЕЭОАЫЯИЁ':
                                    word_for_dict.append(''.join(syllable.replace(i, i + "'")))
                        else:
                            word_for_dict.append(syllable)
                    words_to_add.append(''.join(word_for_dict))
        for item in words_to_add:
            lines_to_add.append(item.replace("'", "") + '#' + ''.join(item))
        else:
            with open(dictionary_path, 'a', encoding='utf-8') as d:
                for item in sorted(set(lines_to_add)):
                    d.write(item + '\n')
        print(f"Done. Out of {words_all} words, {words_dict} of them were already in the dictionary. {words_new} words"
              f" were added, {words_all - words_dict - words_new} ignored (as not having primary "
              f"stress).")
