# Author: George Moroz, minor changes by Polina Mashkovtseva
# Date: 31.05.2023

args = commandArgs(trailingOnly=TRUE)

install.packages('tidyverse', dependencies=TRUE)
install.packages('phonfieldwork', dependencies=TRUE)
library(tidyverse)
library(phonfieldwork)

textgrid_folder <- args[1]
results_folder <- args[2]

# set the working directory to the folder with textgrids
setwd(textgrid_folder)

# check if the results folder exists, create it if it doesn't
if (!dir.exists(results_folder)) {
  dir.create(results_folder)
}

# generate a unique filename with a counter
generate_unique_filename <- function(filename, counter = 1) {
  extension <- tools::file_ext(filename)
  base_name <- tools::file_path_sans_ext(filename)
  unique_filename <- paste0(base_name, "_", counter, ".", extension)

  if (file.exists(file.path(results_folder, unique_filename))) {
    counter <- counter + 1
    unique_filename <- generate_unique_filename(filename, counter)
  }

  return(unique_filename)
}

# extract data
map(list.files(pattern = ".TextGrid"), function(tg) {
  df <- textgrid_to_df(tg)
  df |>
    filter(tier == 1,
           content != "") ->
    to_extract

  map(seq_along(to_extract$id), function(i) {
    df |>
      filter(time_start >= to_extract$time_start[i],
             time_end <= to_extract$time_end[i]) |>
      mutate(time_end = time_end - min(time_start),
             time_start = time_start - min(time_start),
             duration = max(time_end) - min(time_start),
             filename = generate_unique_filename(str_c(to_extract$content[i], ".TextGrid"))) ->
      to_textgrid

    map(sort(unique(to_textgrid$tier)), function(j) {
      to_textgrid |>
        filter(tier == j) ->
        tier

      if (!file.exists(str_c(results_folder, unique(tier$filename)))) {
        create_empty_textgrid(duration = unique(tier$duration),
                              tier_name = "fake",
                              path = results_folder,
                              result_file_name = unique(tier$filename))
      }
      df_to_tier(df = tier,
                 textgrid = str_c(results_folder, unique(tier$filename)),
                 tier_name = unique(tier$tier_name))
    })
  })
})

map(list.files(results_folder, pattern = ".TextGrid", full.names = TRUE), function(i) {
  remove_textgrid_tier(i, "fake")
})
