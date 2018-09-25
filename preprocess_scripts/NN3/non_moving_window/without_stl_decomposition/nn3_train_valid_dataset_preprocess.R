OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN3/non_moving_window/without_stl_decomposition/"

DATA_FILE = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN3/NN3_DATASET.csv"
nn3_dataset <- readLines(DATA_FILE)
nn3_dataset <- strsplit(nn3_dataset, ',')

max_forecast_horizon = 18

for (validation in c(TRUE,FALSE)) {
  for (idr in 1: length(nn3_dataset)) {
    OUTPUT_PATH=paste(OUTPUT_DIR,"nn3_stl_",sep='/')
    OUTPUT_PATH=paste(OUTPUT_PATH,max_forecast_horizon,sep='')
    if(validation)
    {
      OUTPUT_PATH=paste(OUTPUT_PATH,'v',sep='')
    }

    OUTPUT_PATH=paste(OUTPUT_PATH,'txt',sep='.')

    time_series = nn3_dataset[idr]
    time_series = unlist(time_series, use.names=FALSE)
    time_series_log = log(as.numeric(time_series))
    time_series_length = length(time_series_log)

    if (!validation) {
        time_series_length = time_series_length - max_forecast_horizon
        time_series_log = time_series_log[1 : time_series_length]
    }

    # stl_result= tryCatch({
    #   sstl=stl(ts(time_series_log,frequency=12),"period")
    #   seasonal_vect=as.numeric(sstl$time.series[,1])
    #   levels_vect=as.numeric(sstl$time.series[,2])
    #   values_vect=as.numeric(sstl$time.series[,2]+sstl$time.series[,3]) # this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
    #   cbind(seasonal_vect,levels_vect,values_vect)
    # }, error = function(e) {
    #   seasonal_vect=rep(0,length(time_series_length))   #stl() may fail, and then we would go on with the seasonality vector=0
    #   levels_vect=time_series_log
    #   values_vect=time_series_log
    #   cbind(seasonal_vect, levels_vect, values_vect)
    # })


    level=time_series_log[time_series_length - max_forecast_horizon] #last "trend" point in the input window is the "level" (the value used for the normalization)
    sav_df=data.frame(id=paste(idr,'|i',sep=''));
    normalized_values = time_series_log-level
    sav_df=cbind(sav_df, t(normalized_values[1: (time_series_length - max_forecast_horizon)])) #inputs: past values normalized by the level
    sav_df[,'o']='|o'
    sav_df=cbind(sav_df, t(normalized_values[(time_series_length - max_forecast_horizon + 1) :length(normalized_values)])) #outputs: future values normalized by the level.
    if(validation){
        sav_df[,'nyb']='|#' #Not Your Business :-) Anything after '|#' is treated as a comment by CNTK's (unitil next bar)
                     #What follows is data that CNTK is not supposed to "see". We will use it in the validation R script.
        sav_df[,'level']=level
        # sav_df = cbind(sav_df, t(stl_result[(time_series_length - max_forecast_horizon + 1) : time_series_length,1]))
    }
    write.table(sav_df, file=OUTPUT_PATH, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)

    print(idr)
  }#through all series from one file
}