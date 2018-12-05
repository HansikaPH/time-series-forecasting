OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/CIF_2015/"

DATA_FILE = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/CIF_2015/cif-dataset.txt"
cif_dataset <- readLines(DATA_FILE)
cif_dataset <- strsplit(cif_dataset, ';')

max_forecast_horizon = 18
input_size = 13 # since this is monthly data, to capture the effect of yearly seasonality, the input window size has been made slightly more than 12

files <- dir(path=OUTPUT_DIR, pattern="nn3_stl_")
file.remove(paste(OUTPUT_DIR, files, sep = ''))

for (validation in c(TRUE,FALSE)) {
  for (idr in 1: length(nn3_dataset)) {
    OUTPUT_PATH=paste(OUTPUT_DIR,"nn3_stl_",sep='/')
    OUTPUT_PATH=paste(OUTPUT_PATH,max_forecast_horizon,sep='')
    OUTPUT_PATH=paste(OUTPUT_PATH,'i',input_size,sep='')
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

    for (inn in input_size:(time_series_length-max_forecast_horizon)) {
      level=stl_result[inn, 2] #last "trend" point in the input window is the "level" (the value used for the normalization)
      sav_df=data.frame(id=paste(idr,'|i',sep=''));

      for (ii in 1:input_size) {
        sav_df[,paste('r',ii,sep='')]=stl_result[inn-input_size+ii,3]-level  #inputs: past values normalized by the level
      }

      sav_df[,'o']='|o'
      for (ii in 1:max_forecast_horizon) {
        sav_df[,paste('o',ii,sep='')]=stl_result[inn+ii,3]-level #outputs: future values normalized by the level.
      }

       if(validation){
        sav_df[,'nyb']='|#' #Not Your Business :-) Anything after '|#' is treated as a comment by CNTK's (unitil next bar)
                     #What follows is data that CNTK is not supposed to "see". We will use it in the validation R script.
        sav_df[,'level']=level
        for (ii in 1:max_forecast_horizon) {
            sav_df[,paste('s',ii,sep='')]=stl_result[inn+ii,1]
        }
       }
      write.table(sav_df, file=OUTPUT_PATH, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)

    } #steps
    print(idr)
  }#through all series from one file
}