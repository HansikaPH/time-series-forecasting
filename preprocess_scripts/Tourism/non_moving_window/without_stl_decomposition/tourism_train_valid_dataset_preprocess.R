output_dir = "./datasets/text_data/Tourism/non_moving_window/without_stl_decomposition/"
suppressWarnings(dir.create(output_dir, recursive = TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/Tourism/tourism_data.csv"

tourism_dataset <- readLines(input_file)
tourism_dataset <- strsplit(tourism_dataset, ',')

max_forecast_horizon = 24
seasonality_period = 12

for (validation in c(TRUE, FALSE)) {
	output_path = paste(output_dir, "tourism_24", sep = '/')
	if (validation) {
		output_path = paste(output_path, 'v', sep = '')
	}

	output_path = paste(output_path, 'txt', sep = '.')
	unlink(output_path)
    for (idr in 1 : length(tourism_dataset)) {
        time_series = unlist(tourism_dataset[idr], use.names = FALSE)
        mean = mean(as.numeric(time_series[1 : length(time_series)]))
        time_series = as.numeric(time_series[1 : length(time_series)])/mean
        time_series_log = log(time_series + 1)
        time_series_length = length(time_series_log)


        if (! validation) {
            time_series_length = time_series_length - max_forecast_horizon
            time_series_log = time_series_log[1 : time_series_length]
        }

        level_value = mean #last "trend" point in the input window is the "level" (the value used for the normalization)

        if (validation) {
            # preallocate data frame
            sav_df = matrix(NA, ncol = (4 + time_series_length), nrow = 1)
            sav_df = as.data.frame(sav_df)

            sav_df[, (time_series_length + 3)] = '|#'
            sav_df[, (time_series_length + 4)] = level_value
        }else {
            # preallocate data frame
            sav_df = matrix(NA, ncol = (2 + time_series_length), nrow = 1)
            sav_df = as.data.frame(sav_df)
        }

        sav_df[, 1] = paste(idr, '|i', sep = '')
        normalized_values = t(time_series_log)

        sav_df[, 2 : (time_series_length - max_forecast_horizon + 1)] = normalized_values[1 : (time_series_length - max_forecast_horizon)]

        sav_df[, (2 + time_series_length - max_forecast_horizon)] = '|o'

        sav_df[, (3 + time_series_length - max_forecast_horizon) : (2 + time_series_length)] = normalized_values[(length(normalized_values) - max_forecast_horizon + 1) : length(normalized_values)]

        write.table(sav_df, file = output_path, row.names = F, col.names = F, sep = " ", quote = F, append=TRUE)
    }
}