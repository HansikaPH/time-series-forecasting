output_dir = "./datasets/text_data/Tourism/moving_window/without_stl_decomposition/"
suppressWarnings(dir.create(output_dir, recursive = TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/Tourism/tourism_data.csv"

tourism_dataset <- readLines(input_file)
tourism_dataset <- strsplit(tourism_dataset, ',')

max_forecast_horizon = 24
seasonality_period = 12
input_size_multiple = 1.25
input_size = round(seasonality_period * input_size_multiple)


output_path = paste(output_dir, "tourism_test_", sep = '/')
output_path = paste(output_path, max_forecast_horizon, sep = '')
output_path = paste(output_path, 'i', input_size, sep = '')

output_path = paste(output_path, 'txt', sep = '.')
unlink(output_path)

for (idr in 1 : length(tourism_dataset)) {
    time_series = unlist(tourism_dataset[idr], use.names = FALSE)
    mean = mean(as.numeric(time_series[1 : length(time_series)]))
    time_series = as.numeric(time_series[1 : length(time_series)]) / mean
    time_series_log = log(time_series + 1)
    time_series_length = length(time_series_log)

    input_windows = embed(time_series_log[1 : time_series_length], input_size)[, input_size : 1]
    
    if(is.null(dim(input_windows))){
      no_of_windows = 1  
    }else{
      no_of_windows = dim(input_windows)[1]
    }

    sav_df = matrix(NA, ncol = (3 + input_size), nrow = no_of_windows)
    sav_df = as.data.frame(sav_df)

    sav_df[, 1] = paste(idr, '|i', sep = '')
    sav_df[, 2 : (input_size + 1)] = input_windows

    sav_df[, (input_size + 2)] = '|#'
    sav_df[, (input_size + 3)] = rep(mean, no_of_windows)

    write.table(sav_df, file = output_path, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}#through all series from one file