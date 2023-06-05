
# Adaptiation from Schwab and Corretge (http://www.praatvocaltoolkit.com/) 
# based on Hirst (2007)

form Input directory name
    comment Enter directory where soundfile lives:
    sentence soundDir
    comment Enter name of sound file
    sentence sound
endform


name$ = sound$ - ".wav"

# new pitch file
calcPitch$ = name$ + ".Pitch"

# Reads in sound (wav) 
Read from file... 'soundDir$'/'name$'.wav

# deletes any existing output file
deleteFile$ = soundDir$ + calcPitch$
filedelete 'deleteFile$'

select Sound 'name$'

selsnd_m = selected("Sound")

nocheck noprogress To Pitch: 0, 40, 600

if extractWord$(selected$(), "") = "Pitch"
    voicedframes = Count voiced frames

    if voicedframes > 0
        q1 = Get quantile: 0, 0, 0.25, "Hertz"
        q3 = Get quantile: 0, 0, 0.75, "Hertz"
        minF0 = round(q1 * 0.75)
        maxF0 = round(q3 * 2.5)
    else
        minF0 = 40
        maxF0 = 600
    endif

    Remove
else
    minF0 = 40
    maxF0 = 600
endif

if minF0 < 3 / (object[selsnd_m].nx * object[selsnd_m].dx)
    minF0 = ceiling(3 / (object[selsnd_m].nx * object[selsnd_m].dx))
endif

selectObject: selsnd_m

noprogress To Pitch: 0.01, minF0, maxF0

Save as text file: soundDir$ + "/" + calcPitch$
