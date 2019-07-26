#Data preparation script
output_dir = "./datasets/text_data/CIF_2016/non_moving_window/"
suppressWarnings(dir.create(output_dir, recursive=TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/CIF_2016/cif-dataset.txt"

cif_df = read.csv(file = input_file, sep = ';', header = FALSE)

names(cif_df)[4 : ncol(cif_df)] = paste('x', (1 : (ncol(cif_df) - 3)), sep = '_')
names(cif_df)[1] = "Series"
names(cif_df)[2] = "maxPredHorizon"

output_path6 = paste(output_dir, "cif_test_6.txt", sep = '/')
output_path12 = paste(output_dir, "cif_test_12.txt", sep = '/')

# delete the existing files before creating new 
unlink(output_path6)
unlink(output_path12)

for (idr in 1 : nrow(cif_df)) {
    time_series = cif_df[idr,]
    series_number = as.character(time_series$Series)

    max_forecast_horizon = time_series$maxPredHorizon

    time_series = as.numeric(time_series[4 : (ncol(time_series))])
    time_series = time_series[! is.na(time_series)]
    time_series_log = log(time_series)
    time_series_length = length(time_series_log)
    stl_result = tryCatch({
        sstl = stl(ts(time_series_log, frequency = 12), "period")
        seasonal_vect = as.numeric(sstl$time.series[, 1])
        nn_levels = as.numeric(sstl$time.series[, 2])
        nn_vect = as.numeric(sstl$time.series[, 2] + sstl$time.series[, 3]) # this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
        cbind(seasonal_vect, nn_levels, nn_vect)
    }, error = function(e) {
        seasonal_vect = rep(0, time_series_length)   #stl() may fail, and then we would go on with the seasonality vector=0
        nn_levels = time_series_log
        nn_vect = time_series_log
        cbind(seasonal_vect, nn_levels, nn_vect)
    })

    seasonality = tryCatch({
        forecast = stlf(ts(stl_result[, 1] , frequency = 12), "period", h = max_forecast_horizon)
        seasonality_vector = as.numeric(forecast$mean)
        c(seasonality_vector)
    }, error = function(e) {
        seasonality_vector = rep(0, max_forecast_horizon)   
        c(seasonality_vector)
    })
    sav_df = data.frame(id = paste(idr, '|i', sep = '')); 
    level = stl_result[time_series_length, 2]
    normalized_values = stl_result[, 3] - level
    sav_df = cbind(sav_df, t(normalized_values[1 : time_series_length])) #
    sav_df[, 'nyb'] = '|#' 
    
    sav_df[, 'level'] = level
    sav_df = cbind(sav_df, t(seasonality))
    
    if (max_forecast_horizon==6) {
      write.table(sav_df, file=output_path6, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)
    } else {
      write.table(sav_df, file=output_path12, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)
    }
    
}#through all series from one file
