OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/kaggle_web_traffic/non_moving_window/without_stl_decomposition"

file <-read.csv(file="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt",sep=',',header = FALSE)
nn5_dataset <-as.data.frame(file)

max_forecast_horizon=59

OUTPUT_PATH=paste(OUTPUT_DIR,"kaggle_",sep='/')
OUTPUT_PATH=paste(OUTPUT_PATH,max_forecast_horizon,sep='')

OUTPUT_PATH=paste(OUTPUT_PATH,'txt',sep='.')
unlink(OUTPUT_PATH)

numeric_dataset = as.matrix(as.data.frame(lapply(nn5_dataset, as.numeric)))
#numeric_dataset = numeric_dataset + 1

#numeric_dataset_log = log(numeric_dataset)

time_series_length = ncol(numeric_dataset)
time_series_length = time_series_length - max_forecast_horizon
#numeric_dataset_log = numeric_dataset_log[,1 : time_series_length]

for (idr in 1: nrow(numeric_dataset)) {
  time_series = numeric_dataset[idr, ]
  mean = mean(time_series)
  time_series = time_series/mean
  time_series_log = log(time_series[1 : time_series_length] + 1)
  
  #
  level=mean #mean "trend" in the input window is the "level" (the value used for the normalization)
  sav_df=data.frame(id=paste(idr,'|i',sep=''));
  normalized_values = time_series_log

  sav_df=cbind(sav_df, t(normalized_values[1: (time_series_length - max_forecast_horizon)]))

  sav_df[,'o']='|o'
  sav_df=cbind(sav_df, t(normalized_values[(time_series_length - max_forecast_horizon + 1) :length(normalized_values)])) #outputs: future values normalized by the level.

  write.table(sav_df, file=OUTPUT_PATH, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)

  print(idr)
}#through all series from one file