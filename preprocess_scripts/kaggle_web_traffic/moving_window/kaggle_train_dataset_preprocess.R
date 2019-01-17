args <- commandArgs(trailingOnly = TRUE)

OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/kaggle_web_traffic/moving_window"

file <-read.csv(file="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt",sep=',',header = FALSE)
kaggle_dataset <-as.data.frame(file)

max_forecast_horizon=59
seasonality_period=7
INPUT_SIZE_MULTIP=1.25
input_size = round(seasonality_period * INPUT_SIZE_MULTIP)

OUTPUT_PATH=paste(OUTPUT_DIR,"kaggle_stl_",sep='/')
OUTPUT_PATH=paste(OUTPUT_PATH,max_forecast_horizon,sep='')
OUTPUT_PATH=paste(OUTPUT_PATH,'i',input_size,sep='')

OUTPUT_PATH=paste(OUTPUT_PATH,'txt',sep='.')
unlink(OUTPUT_PATH)

numeric_dataset = as.matrix(as.data.frame(lapply(kaggle_dataset, as.numeric)))
numeric_dataset = numeric_dataset + 1

numeric_dataset_log = log(numeric_dataset)

time_series_length = ncol(numeric_dataset_log)
time_series_length = time_series_length - max_forecast_horizon
numeric_dataset_log = numeric_dataset_log[,1 : time_series_length]

for (idr in 1: nrow(numeric_dataset_log)) {
  time_series_log = numeric_dataset_log[idr, ]
  
  stl_result= tryCatch({
    sstl=stl(ts(time_series_log,frequency=seasonality_period),"period")
    seasonal_vect=as.numeric(sstl$time.series[,1])
    levels_vect=as.numeric(sstl$time.series[,2])
    values_vect=as.numeric(sstl$time.series[,2]+sstl$time.series[,3]) # this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
    cbind(seasonal_vect,levels_vect,values_vect)
  }, error = function(e) { 
    seasonal_vect=rep(0,length(time_series_log))   #stl() may fail, and then we would go on with the seasonality vector=0
    levels_vect=time_series_log
    values_vect=time_series_log
    cbind(seasonal_vect, levels_vect, values_vect)
  })

  input_windows = embed(stl_result[1 : (time_series_length - max_forecast_horizon), 3], input_size)[, input_size : 1]
  output_windows = embed(stl_result[- (1 : input_size) , 3], max_forecast_horizon)[, max_forecast_horizon : 1]
  level_values = stl_result[input_size : (time_series_length - max_forecast_horizon), 2]
  input_windows = input_windows - level_values
  output_windows = output_windows - level_values

  sav_df = matrix(NA, ncol = (2 + input_size + max_forecast_horizon), nrow = length(level_values))
  sav_df = as.data.frame(sav_df)

  sav_df[, 1] = paste(idr, '|i', sep = '')
  sav_df[, 2 : (input_size + 1)] = input_windows

  sav_df[, (input_size + 2)] = '|o'
  sav_df[, (input_size + 3) : (input_size + max_forecast_horizon + 2)] = output_windows

  write.table(sav_df, file = OUTPUT_PATH, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}#through all series from one file