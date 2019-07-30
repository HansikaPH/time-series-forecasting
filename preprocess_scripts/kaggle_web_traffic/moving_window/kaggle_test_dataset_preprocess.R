library(forecast)

output_dir = "./datasets/text_data/kaggle_web_traffic/moving_window/"
suppressWarnings(dir.create(output_dir, recursive = TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt"

file <-read.csv(file=input_file, sep=',',header = FALSE)
kaggle_dataset <-as.data.frame(file)

max_forecast_horizon=59
seasonality_period=7
input_size_multiple=1.25
input_size = round(seasonality_period * input_size_multiple)

output_path=paste(output_dir,"kaggle_test_",sep='/')
output_path=paste(output_path,max_forecast_horizon,sep='')
output_path=paste(output_path,'i', input_size, sep='')

output_path=paste(output_path,'txt',sep='.')
unlink(output_path)

numeric_dataset = as.matrix(as.data.frame(lapply(kaggle_dataset, as.numeric)))
numeric_dataset = numeric_dataset + 1
numeric_dataset_log = log(numeric_dataset)
time_series_length = ncol(numeric_dataset_log)

for (idr in 1: nrow(numeric_dataset_log)) {
  time_series_log = numeric_dataset_log[idr,]

  stl_result= tryCatch({
    sstl=stl(ts(time_series_log,frequency=seasonality_period),"period")
    seasonal_vect=as.numeric(sstl$time.series[,1])
    levels_vect=as.numeric(sstl$time.series[,2])
    values_vect=as.numeric(sstl$time.series[,2]+sstl$time.series[,3]) # this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
    cbind(seasonal_vect,levels_vect,values_vect)
  }, error = function(e) {
    seasonal_vect=rep(0,time_series_length)   #stl() may fail, and then we would go on with the seasonality vector=0
    levels_vect=time_series_log
    values_vect=time_series_log
    cbind(seasonal_vect, levels_vect, values_vect)
  })

  seasonality = tryCatch({
    forecast = stlf(ts(stl_result[,1] , frequency = seasonality_period), "period",h=max_forecast_horizon)
    seasonality_vector = as.numeric(forecast$mean)
    cbind(seasonality_vector)
  }, error = function(e) {
    seasonality_vector  = rep(0, max_forecast_horizon)   #stl() may fail, and then we would go on with the seasonality vector=0
    cbind(seasonality_vector)
  })

  input_windows = embed(stl_result[1 : time_series_length , 3], input_size)[, input_size : 1]
  level_values = stl_result[input_size : time_series_length, 2]
  input_windows = input_windows - level_values

  sav_df = matrix(NA, ncol = (3 + input_size + max_forecast_horizon), nrow = length(level_values))
  sav_df = as.data.frame(sav_df)

  sav_df[, 1] = paste(idr, '|i', sep = '')
  sav_df[, 2 : (input_size + 1)] = input_windows

  sav_df[, (input_size + 2)] = '|#'
  sav_df[, (input_size + 3)] = level_values

  seasonality_windows = matrix(rep(t(seasonality),each=length(level_values)),nrow=length(level_values))
  sav_df[(input_size + 4) : ncol(sav_df)] = seasonality_windows

  write.table(sav_df, file = output_path, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}#through all series from one file