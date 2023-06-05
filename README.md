# ruspeechstress

this work is based on the code presented in [(Bernhard et al. 2022)](https://github.com/vera-bernhard/stress-detector).

requirements:

* 6.1.39 <= Praat => 6.2.09

* 3.8 <= Python => 3.10

* R == 4.3.0

how to collect dataset:
1. download this folder to your local machine
2. download [dictionary.txt](https://drive.google.com/file/d/1ADm_03fx4NF9eMQBg4gdWUGoEBZFbH4N/view?usp=sharing) (credits to [(Gusev 2009)](https://github.com/IlyaGusev/russ) and put it in the ```data``` folder
3. put input files - audio recording and text transcriptions - in the ```dataset``` folder
4. change ```working_dir``` in the *main* function in ```dataset.py``` with your working directory path and run the script
