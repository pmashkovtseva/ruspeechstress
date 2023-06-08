form Input directory name
    comment Enter wav file
    sentence wavFile
    comment Path to save filtered wavs
    sentence path
endform

Read from file: path$ + "/" +  wavFile$
baseName$ =  wavFile$ - ".wav"
Filter (formula): "if x<500 or x>1000 then 0 else self fi; rectangular band filter"
filtObj$ = baseName$ + "_filt_500.wav"
Rename: filtObj$
Save as WAV file: path$ + "/" + filtObj$

selectObject: "Sound " + baseName$
Filter (formula): "if x<1000 or x>2000 then 0 else self fi; rectangular band filter"
filtObj$ = baseName$ + "_filt_1000.wav"
Rename: filtObj$
Save as WAV file: path$ + "/" + filtObj$

selectObject: "Sound " + baseName$
Filter (formula): "if x<2000 or x>4000 then 0 else self fi; rectangular band filter"
filtObj$ = baseName$ + "_filt_2000.wav"
Rename: filtObj$
Save as WAV file: path$ + "/" + filtObj$
