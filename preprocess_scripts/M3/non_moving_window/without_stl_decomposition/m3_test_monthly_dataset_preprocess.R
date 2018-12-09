library(forecast)

OUTPUT_DIR = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/M3/non_moving_window/without_stl_decomposition/"

file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/M3/Train_Dataset.csv"
m3_dataset <- readLines(file)
m3_dataset <- strsplit(m3_dataset, ',')

max_forecast_horizon = 18

# INPUT_SIZE_MULTIP = 1.25  # using some reasoning and backesting, I decided to make input size a bit (here by 25%) larger than the maximum prediction horizon

for (idr in 1 : length(m3_dataset)) {
    if (idr <= 474 && idr >= 1) { #Macro Series
        OUTPUT_PATH = paste(OUTPUT_DIR, "m3_test_monthly_micro_", sep = '/')
    }
    else if (idr <= 808 && idr > 474) {
        OUTPUT_PATH = paste(OUTPUT_DIR, "m3_test_monthly_industry_", sep = '/')
    }
    else if (idr <= 1120 && idr > 808) {
        OUTPUT_PATH = paste(OUTPUT_DIR, "m3_test_monthly_macro_", sep = '/')
    }
    else if (idr <= 1265 && idr > 1120) {
        OUTPUT_PATH = paste(OUTPUT_DIR, "m3_test_monthly_finance_", sep = '/')
    }
    else if (idr <= 1376 && idr > 1265) {
        OUTPUT_PATH = paste(OUTPUT_DIR, "m3_test_monthly_demo_", sep = '/')
    }
    else if (idr > 1376) {
        OUTPUT_PATH = paste(OUTPUT_DIR, "m3_test_monthly_other_", sep = '/')
    }
    OUTPUT_PATH = paste(OUTPUT_PATH, max_forecast_horizon, sep = '')

    OUTPUT_PATH = paste(OUTPUT_PATH, 'txt', sep = '.')

    time_series = unlist(m3_dataset[idr], use.names = FALSE)
    time_series_log = log(as.numeric(time_series[2 : length(time_series)]))
    time_series_length = length(time_series_log)

    level_value = mean(time_series_log) #last "trend" point in the input window is the "level" (the value used for the normalization)


    # preallocate data frame
    sav_df = matrix(NA, ncol = (3 + time_series_length), nrow = 1)
    sav_df = as.data.frame(sav_df)


    sav_df[, 1] = paste(idr, '|i', sep = '')
    normalized_values = time_series_log - level_value

    sav_df[, 2 : (time_series_length + 1)] = t(normalized_values[1 : time_series_length])


    sav_df[, (2 + time_series_length)] = '|#'
    sav_df[, (3 + time_series_length)] = level_value

    write.table(sav_df, file = OUTPUT_PATH, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}