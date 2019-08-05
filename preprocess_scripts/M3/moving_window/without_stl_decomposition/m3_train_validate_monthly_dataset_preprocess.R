output_dir = "./datasets/text_data/M3/moving_window/without_stl_decomposition/"
suppressWarnings(dir.create(output_dir, recursive=TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/M3/Train_Dataset.csv"

m3_dataset <- readLines(input_file)
m3_dataset <- strsplit(m3_dataset, ',')

max_forecast_horizon = 18
seasonality_period = 12

unlink(paste(output_dir, "m3_monthly_*", sep=""))

for (validation in c(TRUE, FALSE)) {
    for (idr in 1 : length(m3_dataset)) {
        if (idr <= 474 && idr >= 1) { 
            input_size = 13
            output_path = paste(output_dir, "m3_monthly_micro_", sep = '/')
        }
        else if (idr <= 808 && idr > 474) {
            input_size = 13
            output_path = paste(output_dir, "m3_monthly_industry_", sep = '/')
        }
        else if (idr <= 1120 && idr > 808) {
            input_size = 12
            output_path = paste(output_dir, "m3_monthly_macro_", sep = '/')
        }
        else if (idr <= 1265 && idr > 1120) {
            input_size = 13
            output_path = paste(output_dir, "m3_monthly_finance_", sep = '/')
        }
        else if (idr <= 1376 && idr > 1265) {
            input_size = 13
            output_path = paste(output_dir, "m3_monthly_demo_", sep = '/')
        }
        else if (idr > 1376) {
            input_size = 13
            output_path = paste(output_dir, "m3_monthly_other_", sep = '/')
        }
        output_path = paste(output_path, max_forecast_horizon, sep = '')
        output_path = paste(output_path, 'i', input_size, sep = '')
        if (validation) {
            output_path = paste(output_path, 'v', sep = '')
        }
        output_path = paste(output_path, 'txt', sep = '.')

        time_series = unlist(m3_dataset[idr], use.names = FALSE)
        mean = mean(as.numeric(time_series[2 : length(time_series)]))
        time_series = (as.numeric(time_series[2 : length(time_series)]))/mean
        time_series_log = log(time_series)
        time_series_length = length(time_series_log)


        if (! validation) {
            time_series_length = time_series_length - max_forecast_horizon
            time_series_log = time_series_log[1 : time_series_length]
        }

        input_windows = embed(time_series_log[1 : (time_series_length - max_forecast_horizon)], input_size)[, input_size : 1]
        output_windows = embed(time_series_log[- (1 : input_size)], max_forecast_horizon)[, max_forecast_horizon : 1]

        if(is.null(dim(input_windows))){
          no_of_windows = 1  
        }else{
          no_of_windows = dim(input_windows)[1]
        }
        
        if (validation) {
            sav_df = matrix(NA, ncol = (4 + input_size + max_forecast_horizon), no_of_windows)
            sav_df = as.data.frame(sav_df)
            sav_df[, (input_size + max_forecast_horizon + 3)] = '|#'
            sav_df[, (input_size + max_forecast_horizon + 4)] = rep(mean, no_of_windows)
        }else {
            sav_df = matrix(NA, ncol = (2 + input_size + max_forecast_horizon), nrow = no_of_windows)
            sav_df = as.data.frame(sav_df)
        }

        sav_df[, 1] = paste(idr, '|i', sep = '')
        sav_df[, 2 : (input_size + 1)] = input_windows

        sav_df[, (input_size + 2)] = '|o'
        sav_df[, (input_size + 3) : (input_size + max_forecast_horizon + 2)] = output_windows

        write.table(sav_df, file = output_path, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
    }
}