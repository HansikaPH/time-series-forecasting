library(forecast)

OUTPUT_DIR = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN3/non_moving_window/"
TEST_DATA_FILE = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN3/NN3_DATASET.csv"

nn3_dataset <- readLines(TEST_DATA_FILE)
nn3_dataset <- strsplit(nn3_dataset, ',')

max_forecast_horizon=18

for (idr in 1: length(nn3_dataset)) {
  OUTPUT_PATH=paste(OUTPUT_DIR,"nn3_test_",sep='/')
  OUTPUT_PATH=paste(OUTPUT_PATH,max_forecast_horizon,sep='')

  OUTPUT_PATH=paste(OUTPUT_PATH,'txt',sep='.')

  time_series = nn3_dataset[idr]
  time_series = unlist(time_series, use.names=FALSE)
  time_series_log = log(as.numeric(time_series))
  time_series_length = length(time_series_log)

  stl_result= tryCatch({
    sstl=stl(ts(time_series_log,frequency=12),"period")
    seasonal_vect=as.numeric(sstl$time.series[,1])
    levels_vect=as.numeric(sstl$time.series[,2])
    values_vect=as.numeric(sstl$time.series[,2]+sstl$time.series[,3]) # this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
    cbind(seasonal_vect,levels_vect,values_vect)
  }, error = function(e) {
    seasonal_vect=rep(0,length(time_series_length))   #stl() may fail, and then we would go on with the seasonality vector=0
    levels_vect=time_series_log
    values_vect=time_series_log
    cbind(seasonal_vect, levels_vect, values_vect)
  })

  seasonality = tryCatch({
    forecast = stlf(ts(stl_result[,1] , frequency = 12), "period", h = max_forecast_horizon)
    seasonality_vector = as.numeric(forecast$mean)
    c(seasonality_vector)
  }, error = function(e) {
    seasonality_vector = rep(0, max_forecast_horizon)   #stl() may fail, and then we would go on with the seasonality vector=0
    c(seasonality_vector)
  })

  sav_df=data.frame(id=paste(idr,'|i',sep='')); #sav_df is the set of input values in the current window
  level=stl_result[time_series_length,2]
  normalized_values = stl_result[,3]-level
  sav_df=cbind(sav_df, t(normalized_values[1: time_series_length])) #inputs: past values normalized by the level
  sav_df[,'nyb']='|#' #Not Your Business :-) Anything after '|#' is treated as a comment by CNTK's (unitil next bar)
               #What follows is data that CNTK is not supposed to "see". We will use it in the validation R script.
  sav_df[,'level']=level
  sav_df = cbind(sav_df, t(seasonality))

  write.table(sav_df, file=OUTPUT_PATH, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)

  print(idr)
}