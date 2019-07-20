OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/non_moving_window/without_stl_decomposition"

file <-read.csv(file="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/nn5_dataset.txt",sep=',',header = FALSE)
nn5_dataset <-as.data.frame(file)

max_forecast_horizon=56

OUTPUT_PATH56=paste(OUTPUT_DIR,"nn5_",sep='/')
OUTPUT_PATH56=paste(OUTPUT_PATH56,max_forecast_horizon,sep='')

OUTPUT_PATH56=paste(OUTPUT_PATH56,'txt',sep='.')
unlink(OUTPUT_PATH56)

numeric_dataset = as.matrix(as.data.frame(lapply(nn5_dataset, as.numeric)))

time_series_length = ncol(numeric_dataset)
time_series_length = time_series_length - max_forecast_horizon

for (idr in 1: nrow(numeric_dataset)) {
  #mean scaling
  mean = mean(numeric_dataset[idr,])
  time_series = numeric_dataset[idr,]/mean
  time_series_log = log(time_series + 1)
  time_series_log = time_series_log[1 : time_series_length]

  sav_df=data.frame(id=paste(idr,'|i',sep=''));

  sav_df=cbind(sav_df, t(time_series_log[1: (time_series_length - max_forecast_horizon)]))

  sav_df[,'o']='|o'
  sav_df=cbind(sav_df, t(time_series_log[(time_series_length - max_forecast_horizon + 1) :length(time_series_log)])) #outputs: future values normalized by the level.

  write.table(sav_df, file=OUTPUT_PATH56, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)

  print(idr)
}#through all series from one file