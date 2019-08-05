output_dir = "./datasets/text_data/Tourism/non_moving_window/without_stl_decomposition/"
suppressWarnings(dir.create(output_dir, recursive = TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/Tourism/tourism_data.csv"

tourism_dataset <- readLines(input_file)
tourism_dataset <- strsplit(tourism_dataset, ',')

max_forecast_horizon = 24
seasonality_period = 12

output_path = paste(output_dir, "tourism_test_", sep = '/')
output_path = paste(output_path, max_forecast_horizon, sep = '')
output_path = paste(output_path, 'txt', sep = '.')
unlink(output_path)

for (idr in 1 : length(tourism_dataset)) {

    time_series = unlist(tourism_dataset[idr], use.names = FALSE)
    mean = mean(as.numeric(time_series[1 : length(time_series)]))
    time_series = as.numeric(time_series[1 : length(time_series)])/mean
    time_series_log = log(time_series + 1)
    time_series_length = length(time_series_log)

    level_value = mean #last "trend" point in the input window is the "level" (the value used for the normalization)


    # preallocate data frame
    sav_df = matrix(NA, ncol = (3 + time_series_length), nrow = 1)
    sav_df = as.data.frame(sav_df)


    sav_df[, 1] = paste(idr, '|i', sep = '')
    normalized_values = time_series_log

    sav_df[, 2 : (time_series_length + 1)] = t(normalized_values[1 : time_series_length])


    sav_df[, (2 + time_series_length)] = '|#'
    sav_df[, (3 + time_series_length)] = level_value

    write.table(sav_df, file = output_path, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}