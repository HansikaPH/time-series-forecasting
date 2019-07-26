# Data preparation script

output_dir = "./datasets/text_data/CIF_2016/non_moving_window/without_stl_decomposition/"
suppressWarnings(dir.create(output_dir, recursive = TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/CIF_2016/cif-dataset.txt"

cif_df = read.csv(file = input_file, sep = ';', header = FALSE)

names(cif_df)[4 : ncol(cif_df)] = paste('x', (1 : (ncol(cif_df) - 3)), sep = '_')
names(cif_df)[1] = "Series"
names(cif_df)[2] = "maxPredHorizon"

output_path6 = paste(output_dir, "cif_6", sep = '/')
output_path12 = paste(output_dir, "cif_12", sep = '/')

#The validation file contains the training file, although only last record per each series in the validation file is used for calculating the metrics. 
#This is becasue we are using the recurrent networks with potentially long memory (LSTMs), so all the records are needed for "warm-up" or establishment of the state.  
for (validation in c(TRUE, FALSE)) {#
    output_path_final6 = output_path6
    output_path_final12 = output_path12
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
        mean = mean(time_series)
        time_series = time_series / mean
        time_series_log = log(time_series)
        time_series_length = length(time_series_log)
        if (! validation) {
            time_series_length = time_series_length - max_forecast_horizon
            time_series_log = time_series_log[1 : time_series_length]
        }

        sav_df = data.frame(id = paste(idr, '|i', sep = '')); #sav_df is the set of input values in the current window
        level = mean
        normalized_values = time_series_log
        sav_df = cbind(sav_df, t(normalized_values[1 : (time_series_length - max_forecast_horizon)])) #inputs: past values normalized by the level
        sav_df[, 'o'] = '|o'
        sav_df = cbind(sav_df, t(normalized_values[(time_series_length - max_forecast_horizon + 1) : length(normalized_values)])) #outputs: future values normalized by the level.
        if (validation) {
            sav_df[, 'nyb'] = '|#'
            sav_df[, 'level'] = level
        }

        if (max_forecast_horizon == 6) {
            write.table(sav_df, file = output_path_final6, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
        } else {
            write.table(sav_df, file = output_path_final12, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
        }
    }#through all series from one file
}

