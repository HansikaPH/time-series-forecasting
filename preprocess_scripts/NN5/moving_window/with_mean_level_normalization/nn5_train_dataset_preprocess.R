args <- commandArgs(trailingOnly = TRUE)

OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/moving_window/with_mean_level_normalization"

file <-read.csv(file="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/nn5_dataset.txt",sep=',',header = FALSE)
nn5_dataset <-as.data.frame(file)

max_forecast_horizon=56

if(length(args) != 0) {
  input_size = as.integer(args[1])
} else{
  INPUT_SIZE_MULTIP=1.25  # using some reasoning and backesting, I decided to make input size a bit (here by 25%) larger than the maximum prediction horizon
  input_size=as.integer(INPUT_SIZE_MULTIP*max_forecast_horizon)
}

OUTPUT_PATH56=paste(OUTPUT_DIR,"nn5_stl_",sep='/')
OUTPUT_PATH56=paste(OUTPUT_PATH56,max_forecast_horizon,sep='')
OUTPUT_PATH56=paste(OUTPUT_PATH56,'i',input_size,sep='')

OUTPUT_PATH56=paste(OUTPUT_PATH56,'txt',sep='.')
unlink(OUTPUT_PATH56)

numeric_dataset = as.matrix(as.data.frame(lapply(nn5_dataset, as.numeric)))
numeric_dataset = numeric_dataset + 1

numeric_dataset_log = log(numeric_dataset)

time_series_length = ncol(numeric_dataset_log)
time_series_length = time_series_length - max_forecast_horizon
numeric_dataset_log = numeric_dataset_log[,1 : time_series_length]

for (idr in 1: nrow(numeric_dataset_log)) {
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

  for (inn in input_size:(time_series_length-max_forecast_horizon)) {
    level=stl_result[, 2] #last "trend" point in the input window is the "level" (the value used for the normalization)
    sav_df=data.frame(id=paste(idr,'|i',sep=''));

    for (ii in 1:input_size) {
      sav_df[,paste('r',ii,sep='')]=stl_result[inn-input_size+ii,3] - level  #inputs: past values normalized by the mean level
    }

    sav_df[,'o']='|o'
    for (ii in 1:max_forecast_horizon) {
      sav_df[,paste('o',ii,sep='')]=stl_result[inn+ii,3] - level #outputs: future values normalized by the mean level.
    }

    write.table(sav_df, file=OUTPUT_PATH56, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)

  } #steps
  print(idr)
}#through all series from one file