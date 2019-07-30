output_dir = "./datasets/text_data/M4/moving_window/without_stl_decomposition/"
suppressWarnings(dir.create(output_dir, recursive=TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/M4/Monthly-train.csv"

m4_dataset <- readLines(input_file)
m4_dataset <- strsplit(m4_dataset, ',')

max_forecast_horizon = 18
seasonality_period = 12

unlink(paste(output_dir, "m4_test_monthly_*", sep=""))

for (idr in 2 : length(m4_dataset)) {
    if (idr - 1 <= 10016 && idr - 1 >= 1) { #Macro Series
        input_size = 15
        output_path = paste(output_dir, "m4_test_monthly_macro", sep = '/')
    }
    else if (idr - 1 <= 20991 && idr - 1 > 10016) {
        input_size = 15
        output_path = paste(output_dir, "m4_test_monthly_micro", sep = '/')
    }
    else if (idr - 1 <= 26719 && idr - 1 > 20991) {
        input_size = 15
        output_path = paste(output_dir, "m4_test_monthly_demo", sep = '/')
    }
    else if (idr - 1 <= 36736 && idr - 1 > 26719) {
        input_size = 15
        output_path = paste(output_dir, "m4_test_monthly_industry", sep = '/')
    }
    else if (idr - 1 <= 47723 && idr - 1 > 36736) {
        input_size = 15
        output_path = paste(output_dir, "m4_test_monthly_finance", sep = '/')
    }
    else if (idr - 1 > 47723) {
        input_size = 5
        output_path = paste(output_dir, "m4_test_monthly_other", sep = '/')
    }

    output_path = paste(output_path, max_forecast_horizon, sep = '')
    output_path = paste(output_path, 'i', input_size, sep = '')
    output_path = paste(output_path, 'txt', sep = '.')

    time_series = unlist(m4_dataset[idr], use.names = FALSE)
    mean = mean(as.numeric(time_series[2 : length(time_series)]))
    time_series = as.numeric(time_series[2 : length(time_series)])/mean
    time_series_log = log(time_series)
    time_series_length = length(time_series_log)

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

    write.table(sav_df, file = output_path, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}