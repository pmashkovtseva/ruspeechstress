form Input directory name
    comment Enter wav path
    sentence path
    comment Enter basname
    sentence baseName
    comment Starting point of syllable
    real start
    comment Ending point of syllable
    real end
endform

Read from file: path$ + "/" + baseName$ + ".Pitch"
Get maximum... start end Hertz Parabolic
Get mean... start end Hertz
Get mean... start end semitones re 1 Hz