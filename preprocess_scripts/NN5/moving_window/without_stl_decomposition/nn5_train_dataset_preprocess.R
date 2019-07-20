args <- commandArgs(trailingOnly = TRUE)

OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/moving_window/without_stl_decomposition/"

file <-read.csv(file="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/nn5_dataset.txt",sep=',',header = FALSE)
nn5_dataset <-as.data.frame(file)

max_forecast_horizon=56
seasonality_period=7
INPUT_SIZE_MULTIP=1.25
input_size = round(seasonality_period * INPUT_SIZE_MULTIP)

OUTPUT_PATH56=paste(OUTPUT_DIR,"nn5_",sep='/')
OUTPUT_PATH56=paste(OUTPUT_PATH56,max_forecast_horizon,sep='')
OUTPUT_PATH56=paste(OUTPUT_PATH56,'i',input_size,sep='')

OUTPUT_PATH56=paste(OUTPUT_PATH56,'txt',sep='.')
unlink(OUTPUT_PATH56)

numeric_dataset = as.matrix(as.data.frame(lapply(nn5_dataset, as.numeric)))
# numeric_dataset = numeric_dataset + 1

# numeric_dataset_log = log(numeric_dataset)

time_series_length = ncol(numeric_dataset)
time_series_length = time_series_length - max_forecast_horizon
# numeric_dataset_log = numeric_dataset_log[,1 : time_series_length]

for (idr in 1: nrow(numeric_dataset)) {
  mean = mean(numeric_dataset[idr, ])
  time_series = numeric_dataset[idr, ]/mean
  time_series_log = log(time_series + 1)

  for (inn in input_size:(time_series_length-max_forecast_horizon)) {
    sav_df=data.frame(id=paste(idr,'|i',sep=''));

    for (ii in 1:input_size) {
      sav_df[,paste('r',ii,sep='')]=time_series_log[inn-input_size+ii]  #inputs: past values normalized by the level
    }

    sav_df[,'o']='|o'
    for (ii in 1:max_forecast_horizon) {
      sav_df[,paste('o',ii,sep='')]=time_series_log[inn+ii] #outputs: future values normalized by the level.
    }

    write.table(sav_df, file=OUTPUT_PATH56, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)

  } #steps
  print(idr)
}#through all series from one file