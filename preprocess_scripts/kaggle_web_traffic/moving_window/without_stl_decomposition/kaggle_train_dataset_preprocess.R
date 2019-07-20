args <- commandArgs(trailingOnly = TRUE)

OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/kaggle_web_traffic/moving_window/without_stl_decomposition/"

file <-read.csv(file="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt",sep=',',header = FALSE)
kaggle_dataset <-as.data.frame(file)

max_forecast_horizon=59
seasonality_period=7
INPUT_SIZE_MULTIP=1.25
input_size = round(seasonality_period * INPUT_SIZE_MULTIP)

OUTPUT_PATH=paste(OUTPUT_DIR,"kaggle_",sep='/')
OUTPUT_PATH=paste(OUTPUT_PATH,max_forecast_horizon,sep='')
OUTPUT_PATH=paste(OUTPUT_PATH,'i',input_size,sep='')

OUTPUT_PATH=paste(OUTPUT_PATH,'txt',sep='.')
unlink(OUTPUT_PATH)

numeric_dataset = as.matrix(as.data.frame(lapply(kaggle_dataset, as.numeric)))
# numeric_dataset = numeric_dataset + 1

# numeric_dataset_log = log(numeric_dataset)

time_series_length = ncol(numeric_dataset)
time_series_length = time_series_length - max_forecast_horizon
# numeric_dataset_log = numeric_dataset_log[,1 : time_series_length]

for (idr in 1: nrow(numeric_dataset)) {
  time_series = numeric_dataset[idr,]
  mean = mean(time_series)
  time_series = time_series/mean
  time_series_log = log(time_series[1 : time_series_length] + 1)

  input_windows = embed(time_series_log[1 : (time_series_length - max_forecast_horizon)], input_size)[, input_size : 1]
  output_windows = embed(time_series_log[- (1 : input_size)], max_forecast_horizon)[, max_forecast_horizon : 1]

  sav_df = matrix(NA, ncol = (2 + input_size + max_forecast_horizon), nrow = dim(input_windows)[1])
  sav_df = as.data.frame(sav_df)

  sav_df[, 1] = paste(idr, '|i', sep = '')
  sav_df[, 2 : (input_size + 1)] = input_windows

  sav_df[, (input_size + 2)] = '|o'
  sav_df[, (input_size + 3) : (input_size + max_forecast_horizon + 2)] = output_windows

  write.table(sav_df, file = OUTPUT_PATH, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}#through all series from one file