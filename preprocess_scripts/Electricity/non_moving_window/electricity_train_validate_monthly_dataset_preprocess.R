library(forecast)

OUTPUT_DIR = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/Electricity/non_moving_window/"

file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/Electricity/original_electricity_data.csv"
electricity_dataset <- readLines(file)
electricity_dataset <- strsplit(electricity_dataset, ',')

max_forecast_horizon = 12
seasonality_period = 12


for (validation in c(TRUE, FALSE)) {
	OUTPUT_PATH = paste(OUTPUT_DIR, "electricity_stl_12", sep = '/')
	if (validation) {
		OUTPUT_PATH = paste(OUTPUT_PATH, 'v', sep = '')
	}

	OUTPUT_PATH = paste(OUTPUT_PATH, 'txt', sep = '.')
	unlink(OUTPUT_PATH)
    for (idr in 1 : length(electricity_dataset)) {
        time_series = unlist(electricity_dataset[idr], use.names = FALSE)
        time_series_log = log(as.numeric(time_series[1 : length(time_series)]) + 1)
        time_series_length = length(time_series_log)


        if (! validation) {
            time_series_length = time_series_length - max_forecast_horizon
            time_series_log = time_series_log[1 : time_series_length]
        }
        # apply stl
        stl_result = tryCatch({
            sstl = stl(ts(time_series_log, frequency = seasonality_period), "period")
            seasonal_vect = as.numeric(sstl$time.series[, 1])
            levels_vect = as.numeric(sstl$time.series[, 2])
            values_vect = as.numeric(sstl$time.series[, 2] + sstl$time.series[, 3])# this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
            cbind(seasonal_vect, levels_vect, values_vect)
        }, error = function(e) {
            seasonal_vect = rep(0, length(time_series_length))#stl() may fail, and then we would go on with the seasonality vector=0
            levels_vect = time_series_log
            values_vect = time_series_log
            cbind(seasonal_vect, levels_vect, values_vect)
        })

        level_value = stl_result[time_series_length - max_forecast_horizon, 2] #last "trend" point in the input window is the "level" (the value used for the normalization)

        if (validation) {
            # preallocate data frame
            sav_df = matrix(NA, ncol = (4 + time_series_length + max_forecast_horizon), nrow = 1)
            sav_df = as.data.frame(sav_df)

            sav_df[, (time_series_length + 3)] = '|#'
            sav_df[, (time_series_length + 4)] = level_value
            sav_df[, (time_series_length + 5) : ncol(sav_df)] = (stl_result[(time_series_length - max_forecast_horizon + 1) : time_series_length,1])
        }else {
            # preallocate data frame
            sav_df = matrix(NA, ncol = (2 + time_series_length), nrow = 1)
            sav_df = as.data.frame(sav_df)
        }

        sav_df[, 1] = paste(idr, '|i', sep = '')
        normalized_values = t(stl_result[, 3]) - level_value

        sav_df[, 2 : (time_series_length - max_forecast_horizon + 1)] = normalized_values[1 : (time_series_length - max_forecast_horizon)]

        sav_df[, (2 + time_series_length - max_forecast_horizon)] = '|o'

        sav_df[, (3 + time_series_length - max_forecast_horizon) : (2 + time_series_length)] = normalized_values[(length(normalized_values) - max_forecast_horizon + 1) : length(normalized_values)]

        write.table(sav_df, file = OUTPUT_PATH, row.names = F, col.names = F, sep = " ", quote = F, append=TRUE)
    }
}