output_dir = "./datasets/text_data/kaggle_web_traffic/non_moving_window/without_stl_decomposition/"
suppressWarnings(dir.create(output_dir, recursive = TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt"

file <-read.csv(file=input_file, sep=',',header = FALSE)
kaggle_dataset <-as.data.frame(file)

max_forecast_horizon=59

output_path=paste(output_dir,"kaggle_",sep='/')
output_path=paste(output_path,max_forecast_horizon,sep='')

output_path=paste(output_path,'v.txt',sep='')
unlink(output_path)

numeric_dataset = as.matrix(as.data.frame(lapply(kaggle_dataset, as.numeric)))
time_series_length = ncol(numeric_dataset)

for (idr in 1: nrow(numeric_dataset)) {
  time_series = numeric_dataset[idr,]
  mean = mean(time_series)
  time_series = time_series/mean
  time_series_log = log(time_series + 1)
  
  level=mean 
  sav_df=data.frame(id=paste(idr,'|i',sep=''));
  normalized_values = time_series_log
  sav_df=cbind(sav_df, t(normalized_values[1: (time_series_length - max_forecast_horizon)])) #inputs: past values normalized by the level

  sav_df[,'o']='|o'
  sav_df=cbind(sav_df, t(normalized_values[(time_series_length - max_forecast_horizon + 1) :length(normalized_values)])) #outputs: future values normalized by the level.
  sav_df[,'nyb']='|#' 
  sav_df[,'level']=level
  
  # write.table(sav_df, file=output_path, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)
}#through all series from one file