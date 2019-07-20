library(forecast)

OUTPUT_DIR = "/home/hhew0002/qg35/time-series-forecasting/datasets/text_data/M4/moving_window/without_stl_decomposition/"

file = "/home/hhew0002/qg35/time-series-forecasting/datasets/text_data/M4/Monthly-train.csv"
m4_dataset <- readLines(file)
m4_dataset <- strsplit(m4_dataset, ',')

max_forecast_horizon = 18
seasonality_period = 12

for (idr in 2 : length(m4_dataset)) {
    if (idr - 1 <= 10016 && idr - 1 >= 1) { #Macro Series
        input_size = 15
        OUTPUT_PATH = paste(OUTPUT_DIR, "m4_test_monthly_macro", sep = '/')
    }
    else if (idr - 1 <= 20991 && idr - 1 > 10016) {
        input_size = 15
        OUTPUT_PATH = paste(OUTPUT_DIR, "m4_test_monthly_micro", sep = '/')
    }
    else if (idr - 1 <= 26719 && idr - 1 > 20991) {
        input_size = 15
        OUTPUT_PATH = paste(OUTPUT_DIR, "m4_test_monthly_demo", sep = '/')
    }
    else if (idr - 1 <= 36736 && idr - 1 > 26719) {
        input_size = 15
        OUTPUT_PATH = paste(OUTPUT_DIR, "m4_test_monthly_industry", sep = '/')
    }
    else if (idr - 1 <= 47723 && idr - 1 > 36736) {
        input_size = 15
        OUTPUT_PATH = paste(OUTPUT_DIR, "m4_test_monthly_finance", sep = '/')
    }
    else if (idr - 1 > 47723) {
        input_size = 5
        OUTPUT_PATH = paste(OUTPUT_DIR, "m4_test_monthly_other", sep = '/')
    }

    OUTPUT_PATH = paste(OUTPUT_PATH, max_forecast_horizon, sep = '')
    OUTPUT_PATH = paste(OUTPUT_PATH, 'i', input_size, sep = '')
    OUTPUT_PATH = paste(OUTPUT_PATH, 'txt', sep = '.')

    time_series = unlist(m4_dataset[idr], use.names = FALSE)
    mean = mean(as.numeric(time_series[2 : length(time_series)]))
    time_series = as.numeric(time_series[2 : length(time_series)])/mean
    time_series_log = log(time_series)
    time_series_length = length(time_series_log)

    stl_result = tryCatch({
        sstl = stl(ts(time_series_log, frequency = seasonality_period), "period")
        seasonal_vect = as.numeric(sstl$time.series[, 1])
        levels_vect = as.numeric(sstl$time.series[, 2])
        values_vect = as.numeric(sstl$time.series[, 2] + sstl$time.series[, 3]) # this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
        cbind(seasonal_vect, levels_vect, values_vect)
    }, error = function(e) {
        seasonal_vect = rep(0, length(time_series_length))   #stl() may fail, and then we would go on with the seasonality vector=0
        levels_vect = time_series_log
        values_vect = time_series_log
        cbind(seasonal_vect, levels_vect, values_vect)
    })

    seasonality = tryCatch({
        forecast = stlf(ts(stl_result[, 1] , frequency = seasonality_period), "period", h = max_forecast_horizon)
        seasonality_vector = as.numeric(forecast$mean)
        c(seasonality_vector)
    }, error = function(e) {
        seasonality_vector = rep(0, max_forecast_horizon)   #stl() may fail, and then we would go on with the seasonality vector=0
        c(seasonality_vector)
    })

    input_windows = embed(time_series_log[1 : time_series_length], input_size)[, input_size : 1]
    level_values = stl_result[input_size : time_series_length, 2]

    sav_df = matrix(NA, ncol = (3 + input_size + max_forecast_horizon), nrow = length(level_values))
    sav_df = as.data.frame(sav_df)

    sav_df[, 1] = paste(idr - 1, '|i', sep = '')
    sav_df[, 2 : (input_size + 1)] = input_windows

    sav_df[, (input_size + 2)] = '|#'
    sav_df[, (input_size + 3)] = rep(mean, length(level_values))

    seasonality_windows = matrix(rep(t(seasonality),each=length(level_values)),nrow=length(level_values))
    sav_df[(input_size + 4) : ncol(sav_df)] = seasonality_windows

    write.table(sav_df, file = OUTPUT_PATH, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}