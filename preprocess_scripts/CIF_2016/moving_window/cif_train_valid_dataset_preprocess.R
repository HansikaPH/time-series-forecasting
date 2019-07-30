# Data preparation script

output_dir = "./datasets/text_data/CIF_2016/moving_window/"
suppressWarnings(dir.create(output_dir, recursive = TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/CIF_2016/cif-dataset.txt"

cif_df = read.csv(file = input_file, sep = ';', header = FALSE)

names(cif_df)[4 : ncol(cif_df)] = paste('x', (1 : (ncol(cif_df) - 3)), sep = '_')
names(cif_df)[1] = "Series"
names(cif_df)[2] = "maxPredHorizon"

output_path_6 = paste(output_dir, "stl_6", sep = '/')
output_path_12 = paste(output_dir, "stl_12", sep = '/')

input_size_multiple = 1.25  # using some reasoning and backesting, I decided to make input size a bit (here by 25%) larger than the maximum prediction horizon

input_size_12 = 12 * input_size_multiple
input_size_6 = 7

output_path_6 = paste(output_path_6, 'i', input_size_6, sep = '')
output_path_12 = paste(output_path_12, 'i', input_size_12, sep = '')

#The validation file constains the training file, although only last record per each series in the validation file is used for calculating the metrics. 
#This is becasue we are using the recurrent networks with potentially long memory (LSTMs), so all the records are needed for "warm-up" or establishment of the state.  
for (validation in c(TRUE, FALSE)) {#
    output_path_final6 = output_path_6
    output_path_final12 = output_path_12
    if (validation) {
        output_path_final6 = paste(output_path_final6, 'v', sep = '')
        output_path_final12 = paste(output_path_final12, 'v', sep = '')
    }
    output_path_final6 = paste(output_path_final6, 'txt', sep = '.')
    output_path_final12 = paste(output_path_final12, 'txt', sep = '.')

    unlink(output_path_final6)
    unlink(output_path_final12)
    
    for (idr in 1 : nrow(cif_df)) {
        time_series = cif_df[idr,]
        series_number = as.character(time_series$Series)

        max_forecast_horizon = time_series$maxPredHorizon
        time_series = as.numeric(time_series[4 : (ncol(time_series))])
        time_series = time_series[! is.na(time_series)]
        time_series_log = log(time_series)
        
        time_series_length = length(time_series_log)
        if (! validation) {
          time_series_length = time_series_length - max_forecast_horizon
            time_series_log = time_series_log[1 : time_series_length]
        }

        stl_result = tryCatch({
            sstl = stl(ts(time_series_log, frequency = 12), "period")
            seasonal_vect = as.numeric(sstl$time.series[, 1])
            nn_levels = as.numeric(sstl$time.series[, 2])
            nn_vect = as.numeric(sstl$time.series[, 2] + sstl$time.series[, 3]) # this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
            cbind(seasonal_vect, nn_levels, nn_vect)
        }, error = function(e) {
            seasonal_vect = rep(0, length(time_series_log))   #stl() may fail, and then we would go on with the seasonality vector=0
            nn_levels = time_series_log
            nn_vect = time_series_log
            cbind(seasonal_vect, nn_levels, nn_vect)
        })
        
        if (max_forecast_horizon == 6) {
          input_size = input_size_6
        }else {
          input_size = input_size_12
        }
        
        input_windows = embed(stl_result[1 : (time_series_length - max_forecast_horizon), 3], input_size)[, input_size : 1]
        output_windows = embed(stl_result[- (1 : input_size) , 3], max_forecast_horizon)[, max_forecast_horizon : 1]
        level_values = stl_result[input_size : (time_series_length - max_forecast_horizon), 2]
        input_windows = input_windows - level_values
        output_windows = output_windows - level_values
        if (validation) {
          # create the seasonality metadata
          seasonality_windows = embed(stl_result[- (1 : input_size) , 1], max_forecast_horizon)[, max_forecast_horizon : 1]
          sav_df = matrix(NA, ncol = (4 + input_size + max_forecast_horizon * 2), nrow = length(level_values))
          sav_df = as.data.frame(sav_df)
          sav_df[, (input_size + max_forecast_horizon + 3)] = '|#'
          sav_df[, (input_size + max_forecast_horizon + 4)] = level_values
          sav_df[, (input_size + max_forecast_horizon + 5) : ncol(sav_df)] = seasonality_windows
        }else {
          sav_df = matrix(NA, ncol = (2 + input_size + max_forecast_horizon), nrow = length(level_values))
          sav_df = as.data.frame(sav_df)
        }
        
        sav_df[, 1] = paste(idr, '|i', sep = '')
        sav_df[, 2 : (input_size + 1)] = input_windows
        
        sav_df[, (input_size + 2)] = '|o'
        sav_df[, (input_size + 3) : (input_size + max_forecast_horizon + 2)] = output_windows
        
        if (max_forecast_horizon == 6) {
          write.table(sav_df, file = output_path_final6, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
        } else {
          write.table(sav_df, file = output_path_final12, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
        }
    }
}

