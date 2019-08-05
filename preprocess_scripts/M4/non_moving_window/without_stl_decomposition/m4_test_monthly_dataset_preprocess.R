output_dir = "./datasets/text_data/M4/non_moving_window/without_stl_decomposition/"
suppressWarnings(dir.create(output_dir, recursive=TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/M4/Monthly-train.csv"

m4_dataset <- readLines(file)
m4_dataset <- strsplit(m4_dataset, ',')

max_forecast_horizon = 18
seasonality_period = 12

unlink(paste(output_dir, "m4_test_monthly_*", sep=""))

for (idr in 2 : length(m4_dataset)) {
    if (idr - 1 <= 10016 && idr - 1 >= 1) { #Macro Series
        output_path = paste(output_dir, "m4_test_monthly_macro_", sep = '/')
    }
    else if (idr - 1 <= 20991 && idr - 1 > 10016) {
        output_path = paste(output_dir, "m4_test_monthly_micro_", sep = '/')
    }
    else if (idr - 1 <= 26719 && idr - 1 > 20991) {
        output_path = paste(output_dir, "m4_test_monthly_demo_", sep = '/')
    }
    else if (idr - 1 <= 36736 && idr - 1 > 26719) {
        output_path = paste(output_dir, "m4_test_monthly_industry_", sep = '/')
    }
    else if (idr - 1 <= 47723 && idr - 1 > 36736) {
        output_path = paste(output_dir, "m4_test_monthly_finance_", sep = '/')
    }
    else if (idr - 1 > 47723) {
        output_path = paste(output_dir, "m4_test_monthly_other_", sep = '/')
    }
    output_path = paste(output_path, max_forecast_horizon, sep = '')

    output_path = paste(output_path, 'txt', sep = '.')

    time_series = unlist(m4_dataset[idr], use.names = FALSE)
    mean = mean(as.numeric(time_series[2 : length(time_series)]))
    time_series = as.numeric(time_series[2 : length(time_series)])/mean
    time_series_log = log(time_series)
    time_series_length = length(time_series_log)

    level_value = mean 


    # preallocate data frame
    sav_df = matrix(NA, ncol = (3 + time_series_length), nrow = 1)
    sav_df = as.data.frame(sav_df)

    sav_df[, 1] = paste(idr - 1, '|i', sep = '')
    normalized_values = time_series_log

    sav_df[, 2 : (time_series_length + 1)] = t(normalized_values[1 : time_series_length])


    sav_df[, (2 + time_series_length)] = '|#'
    sav_df[, (3 + time_series_length)] = level_value

    write.table(sav_df, file = output_path, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}