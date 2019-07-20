library(forecast)

OUTPUT_DIR = "/home/hhew0002/qg35/time-series-forecasting/datasets/text_data/M4/non_moving_window/without_stl_decomposition/"

file = "/home/hhew0002/qg35/time-series-forecasting/datasets/text_data/M4/Monthly-train.csv"
m4_dataset <- readLines(file)
m4_dataset <- strsplit(m4_dataset, ',')

max_forecast_horizon = 18
seasonality_period = 12

# INPUT_SIZE_MULTIP = 1.25  # using some reasoning and backesting, I decided to make input size a bit (here by 25%) larger than the maximum prediction horizon

for (validation in c(TRUE, FALSE)) {
    for (idr in 2 : length(m4_dataset)) {
        if (idr - 1 <= 10016 && idr - 1 >= 1) { #Macro Series
            OUTPUT_PATH = paste(OUTPUT_DIR, "m4_monthly_macro_", sep = '/')
        }
        else if (idr - 1 <= 20991 && idr - 1 > 10016) {
            OUTPUT_PATH = paste(OUTPUT_DIR, "m4_monthly_micro_", sep = '/')
        }
        else if (idr - 1 <= 26719 && idr - 1 > 20991) {
            OUTPUT_PATH = paste(OUTPUT_DIR, "m4_monthly_demo_", sep = '/')
        }
        else if (idr - 1 <= 36736 && idr - 1 > 26719) {
            OUTPUT_PATH = paste(OUTPUT_DIR, "m4_monthly_industry_", sep = '/')
        }
        else if (idr - 1 <= 47723 && idr - 1 > 36736) {
            OUTPUT_PATH = paste(OUTPUT_DIR, "m4_monthly_finance_", sep = '/')
        }
        else if (idr - 1 > 47723) {
            OUTPUT_PATH = paste(OUTPUT_DIR, "m4_monthly_other_", sep = '/')
        }
        OUTPUT_PATH = paste(OUTPUT_PATH, max_forecast_horizon, sep = '')
        if (validation) {
            OUTPUT_PATH = paste(OUTPUT_PATH, 'v', sep = '')
        }
        OUTPUT_PATH = paste(OUTPUT_PATH, 'txt', sep = '.')

        time_series = unlist(m4_dataset[idr], use.names = FALSE)
        mean = mean(as.numeric(time_series[2 : length(time_series)]))
        time_series = as.numeric(time_series[2 : length(time_series)])/mean
        time_series_log = log(time_series)
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

        if (validation) {
            # preallocate data frame
            sav_df = matrix(NA, ncol = (4 + time_series_length + max_forecast_horizon), nrow = 1)
            sav_df = as.data.frame(sav_df)

            sav_df[, (time_series_length + 3)] = '|#'
            sav_df[, (time_series_length + 4)] = mean
            sav_df[, (time_series_length + 5) : ncol(sav_df)] = (stl_result[(time_series_length - max_forecast_horizon + 1) : time_series_length,1])

        }else {
            # preallocate data frame
            sav_df = matrix(NA, ncol = (2 + time_series_length), nrow = 1)
            sav_df = as.data.frame(sav_df)
        }

        sav_df[, 1] = paste(idr - 1, '|i', sep = '')
        normalized_values = t(time_series_log)

        sav_df[, 2 : (time_series_length - max_forecast_horizon + 1)] = normalized_values[1 : (time_series_length - max_forecast_horizon)]

        sav_df[, (2 + time_series_length - max_forecast_horizon)] = '|o'

        sav_df[, (3 + time_series_length - max_forecast_horizon) : (2 + time_series_length)] = normalized_values[(length(normalized_values) - max_forecast_horizon + 1) : length(normalized_values)]

        write.table(sav_df, file = OUTPUT_PATH, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
    }
}