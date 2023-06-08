# ruspeechstress

this work is based on the code presented in [(Bernhard et al. 2022)](https://github.com/vera-bernhard/stress-detector).

requirements:

* 6.1.39 <= Praat => 6.2.09

* 3.8 <= Python => 3.10

* R == 4.3.0

* ffmpeg == 6.0

* [rusyllab](https://github.com/Koziev/rusyllab) 0.0.4

* requirements listed in `reqirements.txt`

## usage

```python
pip install ruspeechstress
```

```python
from ruspeechstress import create_dataset, train_model, extend_dictionary

create_dataset(dataset_dir="path to .wav and .txt files", dictionary_path="path to dictionary.txt")
train_model(dataset_dir="path to .wav and .txt files", features path="path to .tsv-features collected on previous step", clf='name of sklearn classifier')
extend_dictionary(dataset_dir="path to .wav and .txt files", dictionary_path="path to dictionary.txt", classifier='path to .pkl classifier', scaler='path to .pkl scaler')
```
