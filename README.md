# ruspeechstress

this work is based on the code presented in [(Bernhard et al. 2022)](https://github.com/vera-bernhard/stress-detector).

requirements:

* 6.1.39 <= Praat => 6.2.09

* 3.8 <= Python => 3.10

* R == 4.3.0

how to collect dataset and train your models:
1. download repo files to your local machine
2. download [dictionary.txt](https://drive.google.com/file/d/1ADm_03fx4NF9eMQBg4gdWUGoEBZFbH4N/view?usp=sharing) (credits to [(Gusev 2009)](https://github.com/IlyaGusev/russ)) and put it in the ```data``` folder
3. put input files - audio recording and text transcriptions - in the ```dataset``` folder
4. change ```working_dir``` in the *main* function in ```dataset.py``` with your working directory path and run the script
5. from (Bernhard et al. 2022):
```python
from stress_detector import StressDetector, FEATURES
from sklearn.tree import DecisionTreeClassifier

wav_path = './wav_tg_all'
sd = StressDetector(wav_path, FEATURES)
sd.read_in()
sd.get_features().to_csv('./data/complete_features.tsv', sep='\t')
sd.get_features('./data/complete_features.tsv')
clf = DecisionTreeClassifier()
evaluation = sd.train(clf)
print('F1 Score: {}'.format(np.mean(evaluation['f1'])))
```
