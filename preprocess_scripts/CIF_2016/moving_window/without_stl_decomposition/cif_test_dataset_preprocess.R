output_dir = "./datasets/text_data/CIF_2016/moving_window/without_stl_decomposition/"
suppressWarnings(dir.create(output_dir, recursive=TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/CIF_2016/cif-dataset.txt"

output_file_12 = paste(output_dir, "cif12test.txt", sep="")
output_file_6 = paste(output_dir, "cif6test.txt", sep="")

unlink(output_file_12)
unlink(output_file_6)

cif_df = read.csv(file = input_file, sep = ';', header = FALSE)

names(cif_df)[4 : ncol(cif_df)] = paste('x', (1 : (ncol(cif_df) - 3)), sep =
'_')
names(cif_df)[1] = "Series"
names(cif_df)[2] = "maxPredHorizon"

input_size_multiple = 1.25

input_size_6 = 7
input_size_12 = input_size_multiple * 12
  
for (idr in 1 : nrow(cif_df)) {
    time_series = cif_df[idr,]
    max_forecast_horizon = cif_df[idr,]$maxPredHorizon
    series_number = as.character(time_series$Series)
    time_series = as.numeric(time_series[4 : (ncol(time_series))])
    time_series = time_series[! is.na(time_series)]
    mean = mean(time_series)
    time_series = time_series / mean
    time_series_log = log(time_series)
    time_series_length = length(time_series_log)

    if (max_forecast_horizon == 6){
      input_size = input_size_6
      output_file = output_file_6
    }else{
      input_size = input_size_12
      output_file = output_file_12
    }
    input_windows = embed(time_series_log[1 : time_series_length], input_size)[, input_size : 1]
    
    if(is.null(dim(input_windows))){
      no_of_windows = 1  
    }else{
      no_of_windows = dim(input_windows)[1]
    }

    sav_df = matrix(NA, ncol = (3 + input_size), nrow = no_of_windows)
    sav_df = as.data.frame(sav_df)
    
    sav_df[, 1] = paste(idr - 1, '|i', sep = '')
    sav_df[, 2 : (input_size + 1)] = input_windows
    
    sav_df[, (input_size + 2)] = '|#'
    sav_df[, (input_size + 3)] = rep(mean, no_of_windows)
    
    write.table(sav_df, file = output_file, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}