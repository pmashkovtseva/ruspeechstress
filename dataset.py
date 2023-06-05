import math
import os
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


class FileSizeError(Exception):
    pass


def get_files(working_dir, input_directory='./dataset'):
    file_pairs = {}
    dir = os.path.join(working_dir, input_directory)
    for file in listdir(dir):
        text = os.path.join(dir, file[:-4] + '.txt')
        wav = os.path.join(dir, file)
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


def webmaus(input_files, working_dir, out_path='dataset/'):
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
            open(os.path.join(working_dir, out_path + os.path.basename(wav)[:-4] + '.TextGrid'), 'wb').write(res2.content)


def build_dictionary(working_dir, input_path='data/dictionary.txt'):
    with open(os.path.join(working_dir, input_path), 'r', encoding='utf-8') as f:
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


def collect_dataset(working_dir, input_files, dictionary, output_path='./wav_tg_all'):
    n = 0
    all_data = []
    words = []
    for wav in input_files.keys():
        if os.path.exists(wav[:-4] + '.TextGrid'):
            transcription = textgrid.TextGrid.fromFile(wav[:-4] + '.TextGrid')
            for i in range(len(transcription[0])):
                word = transcription[0][i]
                if word.mark != "" and word.mark not in ['обо', 'изо',
                                                         'подо'] and word.mark in dictionary.keys() and len(
                    transcription[2][i].mark.split('.')) > 1 and get_number_of_syllables(
                    dictionary[word.mark]) is not None:
                    nr_syll = str(len(transcription[2][i].mark.split('.')))
                    stress_pos = str(len(transcription[2][i].mark.split('.')) - get_number_of_syllables(
                        dictionary[word.mark]) + 1)
                    if word.mark not in words:
                        all_data.append([str(n), word.mark, stress_pos, nr_syll])
                        words.append(word.mark)
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
    with open('data/training_words.tsv', 'w', newline='') as tw:
        tsv_output = csv.writer(tw, delimiter='\t')
        tsv_output.writerow(['', 'word', 'stress_pos', 'nr_syll'])
        for row in all_data:
            tsv_output.writerow(row)


def get_words(working_dir):
    subprocess.call(["Rscript", os.path.join(working_dir, "scripts/get_words.R")])


def clear_files(working_dir, input_path='./wav_tg_all'):
    for file in os.listdir(os.path.join(working_dir, input_path)):
        wav_file = file.strip('.TextGrid') + '.wav'
        if file.endswith('.TextGrid') and not os.path.exists(os.path.join(os.path.join(working_dir, input_path), wav_file)):
            os.remove(os.path.join(input_path, file))


def main():
    working_dir = 'REPLACE WITH YOUR WORKING DIRECTORYr'
    print("Getting files and preprocessing audio and text...")
    files = get_files(working_dir)
    preprocess_audio(files)
    get_size(files)
    transcribe_audio(files)
    files = get_files(working_dir)
    preprocess_text(files)
    print("Creating TextGrid files...")
    webmaus(files, working_dir)
    dictionary = build_dictionary(working_dir)
    print("Collecting dataset...")
    collect_dataset(working_dir, files, dictionary)
    get_words(working_dir)
    clear_files(working_dir)
    print("Done.")


if __name__ == '__main__':
    main()
