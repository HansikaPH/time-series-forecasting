library(forecast)

output_dir = "./datasets/text_data/kaggle_web_traffic/non_moving_window/"
suppressWarnings(dir.create(output_dir, recursive = TRUE)) # create the output directory if not existing
input_file = "./datasets/text_data/kaggle_web_traffic/kaggle_web_traffic_dataset.txt"

file <-read.csv(file=input_file, sep=',',header = FALSE)
kaggle_dataset <-as.data.frame(file)

max_forecast_horizon=59

output_path=paste(output_dir,"kaggle_scaled_test_",sep='/')
output_path=paste(output_path,max_forecast_horizon,sep='')

output_path=paste(output_path,'txt',sep='.')
unlink(output_path)

numeric_dataset = as.matrix(as.data.frame(lapply(kaggle_dataset, as.numeric)))
numeric_dataset_log = log(numeric_dataset + 1)
time_series_length = ncol(numeric_dataset_log)

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

  seasonality = tryCatch({
    forecast = stlf(ts(stl_result[,1] , frequency = 7), "period",h=59)
    seasonality_vector = as.numeric(forecast$mean)
    cbind(seasonality_vector)
  }, error = function(e) {
    seasonality_vector  = rep(0, max_forecast_horizon)   #stl() may fail, and then we would go on with the seasonality vector=0
    cbind(seasonality_vector)
  })

  level=stl_result[time_series_length, 2] #last "trend" point in the whole series is the "level" (the value used for the normalization)
  sav_df=data.frame(id=paste(idr,'|i',sep=''));
  normalized_values = stl_result[,3]-level

  sav_df=cbind(sav_df, t(normalized_values[1: time_series_length])) #inputs: past values normalized by the level
  sav_df[,'nyb']='|#' 
  sav_df[,'level']=level

  sav_df = cbind(sav_df, t(seasonality))
  write.table(sav_df, file=output_path, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)
}