OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/non_moving_window/"

file <-read.csv(file="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/nn5_dataset.txt",sep=',',header = FALSE)
nn5_dataset <-as.data.frame(file)

max_forecast_horizon=56

OUTPUT_PATH56=paste(OUTPUT_DIR,"nn5_stl_",sep='/')
OUTPUT_PATH56=paste(OUTPUT_PATH56,max_forecast_horizon,sep='')

OUTPUT_PATH56=paste(OUTPUT_PATH56,'v.txt',sep='')
unlink(OUTPUT_PATH56)

numeric_dataset = as.matrix(as.data.frame(lapply(nn5_dataset, as.numeric)))
numeric_dataset = numeric_dataset + 1

numeric_dataset_log = log(numeric_dataset)

time_series_length = ncol(numeric_dataset_log)

idr=3
# for (idr in 1: nrow(numeric_dataset_log)) {
  time_series_log = numeric_dataset_log[idr, ]

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
  # levels[idr] = level
  print(level)
  # sav_df=data.frame(id=paste(idr,'|i',sep=''));
  normalized_values = stl_result[,3]-level
  # sav_df=cbind(sav_df, t(normalized_values[1: (time_series_length - max_forecast_horizon)])) #inputs: past values normalized by the level
  #
  # sav_df[,'o']='|o'
  # sav_df=cbind(sav_df, t(normalized_values[(time_series_length - max_forecast_horizon + 1) :length(normalized_values)])) #outputs: future values normalized by the level.
  #
  # sav_df[,'nyb']='|#' #Not Your Business :-) Anything after '|#' is treated as a comment by CNTK's (unitil next bar)
  # #What follows is data that CNTK is not supposed to "see". We will use it in the validation R script.
  # sav_df[,'level']=level
  #
  # sav_df = cbind(sav_df, t(stl_result[(time_series_length - max_forecast_horizon + 1) : time_series_length,1]))
  #
  write.table(normalized_values, file="validation.csv", row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)

  # print(idr)
# }#through all series from one file