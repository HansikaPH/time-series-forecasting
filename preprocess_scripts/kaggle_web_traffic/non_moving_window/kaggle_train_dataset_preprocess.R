output_dir = "./datasets/text_data/kaggle_web_traffic/non_moving_window/"
suppressWarnings(dir.create(output_dir, recursive = TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt"

file <-read.csv(file=input_file, sep=',',header = FALSE)
nn5_dataset <-as.data.frame(file)

max_forecast_horizon=59

output_path=paste(output_dir,"kaggle_stl_",sep='/')
output_path=paste(output_path,max_forecast_horizon,sep='')

output_path=paste(output_path,'txt',sep='.')
unlink(output_path)

numeric_dataset = as.matrix(as.data.frame(lapply(nn5_dataset, as.numeric)))
numeric_dataset_log = log(numeric_dataset + 1)
time_series_length = ncol(numeric_dataset_log) - max_forecast_horizon
numeric_dataset_log = numeric_dataset_log[,1 : time_series_length]

for (idr in 1: nrow(numeric_dataset_log)) {
  time_series_log = numeric_dataset_log[idr,]
  
  stl_result= tryCatch({
    sstl=stl(ts(time_series_log,frequency=7),"period")
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

  level=stl_result[time_series_length - max_forecast_horizon, 2] #last "trend" point in the input window is the "level" (the value used for the normalization)
  sav_df=data.frame(id=paste(idr,'|i',sep=''));
  normalized_values = stl_result[, 3] - level

  sav_df=cbind(sav_df, t(normalized_values[1: (time_series_length - max_forecast_horizon)]))
  sav_df[,'o']='|o'
  sav_df=cbind(sav_df, t(normalized_values[(time_series_length - max_forecast_horizon + 1) :length(normalized_values)])) #outputs: future values normalized by the level.

  write.table(sav_df, file=output_path, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)
}#through all series from one file