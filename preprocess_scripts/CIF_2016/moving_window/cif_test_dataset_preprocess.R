#Data preparation script
library(forecast)
output_dir = "./datasets/text_data/CIF_2016/moving_window/"
suppressWarnings(dir.create(output_dir, recursive=TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/CIF_2016/cif-dataset.txt"

output_file_12 = paste(output_dir, "cif12test.txt", sep="")
output_file_6 = paste(output_dir, "cif6test.txt", sep="")

unlink(output_file_12)
unlink(output_file_6)

cif_df = read.csv(file = input_file, sep = ';', header = FALSE)

names(cif_df)[4:ncol(cif_df)] = paste('x', (1:(ncol(cif_df) - 3)), sep =
                                        '_')
names(cif_df)[1] = "Series"
names(cif_df)[2] = "maxPredHorizon"

input_size_multiple = 1.25
input_size_6 = 7
input_size_12 = input_size_multiple * 12

#Processing for prediction horizon 12
for (idr in 1:nrow(cif_df)) {
  time_series = cif_df[idr, ]
  max_forecast_horizon = cif_df[idr,]$maxPredHorizon
  series_number = as.character(time_series$Series)
  time_series = as.numeric(time_series[4:(ncol(time_series))])
  time_series = time_series[!is.na(time_series)]
  time_series_log = log(time_series)
  time_series_length = length(time_series_log)

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

  seasonality = tryCatch({
    forecast = stlf(ts(stl_result[,1] , frequency = 12), "period", h = max_forecast_horizon)
    seasonality_vector = as.numeric(forecast$mean)
    c(seasonality_vector)
  }, error = function(e) {
      seasonality_vector = rep(0, 12)   #stl() may fail, and then we would go on with the seasonality vector=0
      c(seasonality_vector)
    })
  
  if (max_forecast_horizon == 6){
    input_size = input_size_6
    output_file = output_file_6
  }else{
    input_size = input_size_12
    output_file = output_file_12
  }

  input_windows = embed(stl_result[1 : time_series_length , 3], input_size)[, input_size : 1]
  level_values = stl_result[input_size : time_series_length, 2]
  input_windows = input_windows - level_values
  
  sav_df = matrix(NA, ncol = (3 + input_size + max_forecast_horizon), nrow = length(level_values))
  sav_df = as.data.frame(sav_df)
  
  sav_df[, 1] = paste(idr - 1, '|i', sep = '')
  sav_df[, 2 : (input_size + 1)] = input_windows
  
  sav_df[, (input_size + 2)] = '|#'
  sav_df[, (input_size + 3)] = level_values
  
  seasonality_windows = matrix(rep(t(seasonality), each = length(level_values)), nrow = length(level_values))
  sav_df[(input_size + 4) : ncol(sav_df)] = seasonality_windows
  
  write.table(sav_df, file = output_file, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}