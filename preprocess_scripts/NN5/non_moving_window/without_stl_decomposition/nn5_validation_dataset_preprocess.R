output_dir = "./datasets/text_data/NN5/non_moving_window/without_stl_decomposition/"
suppressWarnings(dir.create(output_dir, recursive=TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/NN5/nn5_dataset.txt"

file <-read.csv(file=input_file,sep=',',header = FALSE)
nn5_dataset <-as.data.frame(file)

max_forecast_horizon=56

output_path=paste(output_dir,"nn5_",sep='/')
output_path=paste(output_path,max_forecast_horizon,sep='')

output_path=paste(output_path,'v.txt',sep='')
unlink(output_path)

numeric_dataset = as.matrix(as.data.frame(lapply(nn5_dataset, as.numeric)))

time_series_length = ncol(numeric_dataset)

for (idr in 1: nrow(numeric_dataset)) {
  mean = mean(numeric_dataset[idr, ])
  time_series = numeric_dataset[idr, ]/mean
  time_series_log = log(time_series + 1)

  level=mean 
  sav_df=data.frame(id=paste(idr,'|i',sep=''));
  sav_df=cbind(sav_df, t(time_series_log[1: (time_series_length - max_forecast_horizon)])) #inputs: past values normalized by the level

  sav_df[,'o']='|o'
  sav_df=cbind(sav_df, t(time_series_log[(time_series_length - max_forecast_horizon + 1) :length(time_series_log)])) #outputs: future values normalized by the level.

  sav_df[,'nyb']='|#' 
  sav_df[,'level']=level

  write.table(sav_df, file=output_path, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)
}#through all series from one file