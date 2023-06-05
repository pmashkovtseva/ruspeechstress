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

Read from file: path$ + "/" + baseName$ + "_filt_500.wav"
Extract part: start, end, "rectangular", 1, "no"
Get intensity (dB)


Read from file: path$ + "/" + baseName$ + "_filt_1000.wav"
Extract part: start, end, "rectangular", 1, "no"
Get intensity (dB)

Read from file: path$ + "/" + baseName$ + "_filt_2000.wav"
Extract part: start, end, "rectangular", 1, "no"
Get intensity (dB)