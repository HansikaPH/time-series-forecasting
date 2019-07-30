output_dir = "./datasets/text_data/kaggle_web_traffic/moving_window/without_stl_decomposition/"
suppressWarnings(dir.create(output_dir, recursive = TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt"

file <-read.csv(file=input_file,sep=',',header = FALSE)
nn5_dataset <-as.data.frame(file)

max_forecast_horizon=59
seasonality_period=7
input_size_multiple=1.25
input_size = round(seasonality_period * input_size_multiple)

output_path=paste(output_dir,"kaggle_test_",sep='/')
output_path=paste(output_path,max_forecast_horizon,sep='')
output_path=paste(output_path,'i', input_size, sep='')

output_path=paste(output_path,'txt',sep='.')
unlink(output_path)

numeric_dataset = as.matrix(as.data.frame(lapply(nn5_dataset, as.numeric)))
time_series_length = ncol(numeric_dataset)

for (idr in 1: nrow(numeric_dataset)) {
  time_series = numeric_dataset[idr,]
  mean = mean(time_series)
  time_series = time_series/mean
  time_series_log = log(time_series + 1)

  input_windows = embed(time_series_log[1 : time_series_length], input_size)[, input_size : 1]

  if(is.null(dim(input_windows))){
    no_of_windows = 1  
  }else{
    no_of_windows = dim(input_windows)[1]
  }
  
  sav_df = matrix(NA, ncol = (3 + input_size), nrow = no_of_windows)
  sav_df = as.data.frame(sav_df)

  sav_df[, 1] = paste(idr, '|i', sep = '')
  sav_df[, 2 : (input_size + 1)] = input_windows

  sav_df[, (input_size + 2)] = '|#'
  sav_df[, (input_size + 3)] = rep(mean, no_of_windows)

  write.table(sav_df, file = output_path, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}#through all series from one file