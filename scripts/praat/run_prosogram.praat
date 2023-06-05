form Prosogram 
    sentence input_directory
    boolean recalc_pitch
    boolean opt_strategy

endform

# TODO: adjust path to Prosogram according to your environment
if recalc_pitch = 1
    if opt_strategy = 0
        runScript: "../prosogram/prosogram.praat", "Recalculate pitch for entire sound file", input_directory$, 0, 0, 0, 450, "Full (saved in file)", "0.005", "Using external segmentation in tier ""segm""", "G=0.16-0.32/T^2 (adaptive), DG=30, dmin=0.050", "Wide, light", 3, "*1, 2, 3", "Times 10", 0, 100, "Fill page with strips", "PNG 300 dpi", "<input_directory>/<basename>_"

    else
        runScript: "../prosogram/prosogram.praat", "Recalculate pitch for entire sound file", input_directory$, 0, 0, 0, 450, "Full (saved in file)", "0.005", "Automatic: acoustic syllables", "G=0.16-0.32/T^2 (adaptive), DG=30, dmin=0.050", "Wide, light", 3, "*1, 2, 3", "Times 10", 0, 100, "Fill page with strips", "PNG 300 dpi", "<input_directory>/<basename>_"
    endif

else

    if opt_strategy = 0
       runScript: "../prosogram/prosogram.praat", "Calculate intermediate data files & Prosodic profile (no graphics files)", input_directory$, 0, 0, 0, 450, "Full (saved in file)", "0.005", "Using external segmentation in tier ""segm""", "G=0.16-0.32/T^2 (adaptive), DG=30, dmin=0.050", "Wide, rich, with pitch range", 3, "*1, 2, 3", "Times 10", 0, 100, "Fill page with strips", "PNG 300 dpi", "<input_directory>/<basename>_"
    
    else       
        runScript: "../prosogram/prosogram.praat", "Calculate intermediate data files & Prosodic profile (no graphics files)", input_directory$, 0, 0, 0, 450, "Full (saved in file)", "0.005", "Automatic: acoustic syllables", "G=0.16-0.32/T^2 (adaptive), DG=30, dmin=0.050", "Wide, rich, with pitch range", 3, "*1, 2, 3", "Times 10", 0, 100, "Fill page with strips", "PNG 300 dpi", "<input_directory>/<basename>_"
    endif
endif