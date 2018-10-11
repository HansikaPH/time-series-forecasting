library(forecast)
library(tibble)

OUTPUT_DIR = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/M4/moving_window/"

file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/M4/Monthly-train.csv"
m4_dataset <- readLines(file)
m4_dataset <- strsplit(m4_dataset, ',')

max_forecast_horizon = 18
seasonality_period = 12

INPUT_SIZE_MULTIP = 1.25  # using some reasoning and backesting, I decided to make input size a bit (here by 25%) larger than the maximum prediction horizon
input_size = as.integer(INPUT_SIZE_MULTIP * seasonality_period)

for (validation in c(TRUE, FALSE)) {
    for (idr in 2 : 2) {
        if (idr - 1 <= 10016 && idr - 1 >= 1) { #Macro Series
            OUTPUT_PATH = paste(OUTPUT_DIR, "m4_stl_monthly_macro", sep = '/')
        }
        else if (idr - 1 <= 20991 && idr - 1 > 10016) {
            OUTPUT_PATH = paste(OUTPUT_DIR, "m4_stl_monthly_micro", sep = '/')
        }
        else if (idr - 1 <= 26719 && idr - 1 > 20991) {
            OUTPUT_PATH = paste(OUTPUT_DIR, "m4_stl_monthly_demo", sep = '/')
        }
        else if (idr - 1 <= 36736 && idr - 1 > 26719) {
            OUTPUT_PATH = paste(OUTPUT_DIR, "m4_stl_monthly_industry", sep = '/')
        }
        else if (idr - 1 <= 47723 && idr - 1 > 36736) {
            OUTPUT_PATH = paste(OUTPUT_DIR, "m4_stl_monthly_finance", sep = '/')
        }
        else if (idr - 1 > 47723) {
            OUTPUT_PATH = paste(OUTPUT_DIR, "m4_stl_monthly_other", sep = '/')
        }
        OUTPUT_PATH = paste(OUTPUT_PATH, max_forecast_horizon, sep = '')
        OUTPUT_PATH = paste(OUTPUT_PATH, 'i', input_size, sep = '')
        if (validation) {
            OUTPUT_PATH = paste(OUTPUT_PATH, 'v', sep = '')
        }
        OUTPUT_PATH = paste(OUTPUT_PATH, 'txt', sep = '.')

        time_series = unlist(m4_dataset[idr], use.names = FALSE)
        time_series_log = log(as.numeric(time_series[2 : length(time_series)]))
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

        input_windows = embed(stl_result[1 : (time_series_length - max_forecast_horizon), 3], input_size)[, input_size : 1]
        output_windows = embed(stl_result[- (1 : input_size) , 3], max_forecast_horizon)[, max_forecast_horizon : 1]

        level_values = stl_result[input_size : (time_series_length - max_forecast_horizon), 2]
        print(input_windows)
        print(output_windows)

        input_windows = input_windows - level_values
        output_windows = output_windows - level_values



        print(level_values)

        if (validation) {
            # create the seasonality metadata
            seasonality_windows = embed(stl_result[- (1 : input_size) , 1], max_forecast_horizon)[, max_forecast_horizon : 1]

            print(seasonality_period)

            sav_df = matrix(NA, ncol = (4 + input_size + max_forecast_horizon * 2), nrow = length(level_values))
            sav_df = as.data.frame(sav_df)
            sav_df[, (input_size + max_forecast_horizon + 3)] = '|#'
            sav_df[, (input_size + max_forecast_horizon + 4)] = level_values
            sav_df[, (input_size + max_forecast_horizon + 5) : ncol(sav_df)] = seasonality_windows
        }else {
            sav_df = matrix(NA, ncol = (2 + input_size + max_forecast_horizon), nrow = length(level_values))
            sav_df = as.data.frame(sav_df)
        }

        sav_df[, 1] = paste(idr - 1, '|i', sep = '')
        sav_df[, 2 : (input_size + 1)] = input_windows

        sav_df[, (input_size + 2)] = '|o'
        sav_df[, (input_size + 3) : (input_size + max_forecast_horizon + 2)] = output_windows

        write.table(sav_df, file = OUTPUT_PATH, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)


        # for (inn in input_size : (time_series_length - max_forecast_horizon)) {
        #     level = stl_result[inn, 2]#last "trend" point in the input window is the "level" (the value used for the normalization)
        #     sav_df = data.frame(id = paste(idr, '|i', sep = ''));
        #
        #     sav_df[, paste('r', (1 : input_size), sep = '')] = stl_result[(inn - input_size + 1) : inn, 3] - level #inputs: past values normalized by the level
        #
        #     sav_df[, 'o'] = '|o'
        #     sav_df[, paste('o', (1 : max_forecast_horizon), sep = '')] = stl_result[(inn + 1) : (inn + max_forecast_horizon), 3] - level #outputs: future values normalized by the level.
        #
        #     if (validation) {
        #         sav_df[, 'nyb'] = '|#' #Not Your Business :-) Anything after '|#' is treated as a comment by CNTK's (unitil next bar)
        #         #What follows is data that CNTK is not supposed to "see". We will use it in the validation R script.
        #         sav_df[, 'level'] = level
        #         sav_df[, paste('s', (1 : max_forecast_horizon), sep = '')] = stl_result[(inn + 1) : (inn + max_forecast_horizon), 1]
        #     }
        #     write.table(sav_df, file = OUTPUT_PATH, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
        # }#steps
    }
}