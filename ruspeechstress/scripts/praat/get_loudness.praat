form Input directory name
    comment Enter wav path
    sentence path
    comment Enter basname
    sentence baseName
    comment Starting point of syllable
    real start
    comment Ending point of syllable
    real end
    comment Calculate root mean square enegery or not
    boolean rms
    comment Calculate peak intensity or not
    boolean peak_int
    comment Calculate mean intensity or not
    boolean mean_int

endform

Read from file: path$ + "/" + baseName$ + ".wav"
if rms = 1
    Extract part: start, end, "rectangular", 1, "no"
    Get root-mean-square...  0 0
elsif peak_int = 1
    To Intensity... 100 0 yes
    Get maximum... start end parabolic
elsif mean_int = 1
    To Intensity... 100 0 yes
    Get mean... start end
endif
