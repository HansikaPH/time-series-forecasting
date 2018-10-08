OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/M4/moving_window"

file <-read.csv(file="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/M4/Monthly-train.csv",sep=',',header = TRUE)
m4_dataset <-as.data.frame(t(file[,-1]))

max_forecast_horizon=18
seasonality_period = 12

INPUT_SIZE_MULTIP=1.25  # using some reasoning and backesting, I decided to make input size a bit (here by 25%) larger than the maximum prediction horizon
input_size=as.integer(INPUT_SIZE_MULTIP*seasonality_period)

for (validation in c(TRUE,FALSE)) {
  for (idr in 1: nrow(m4_dataset)) {
    if(idr <= 10016 && idr >= 1){ #Macro Series
      OUTPUT_PATH=paste(OUTPUT_DIR,"m4_stl_monthly_macro",sep='/')
    }
    else if(idr <= 20991 && idr > 10016) {
      OUTPUT_PATH=paste(OUTPUT_DIR,"m4_stl_monthly_micro",sep='/')
    }
    else if(idr <= 26719 && idr > 20991) {
      OUTPUT_PATH=paste(OUTPUT_DIR,"m4_stl_monthly_demo",sep='/')
    }
    else if(idr <= 36736 && idr > 26719) {
      OUTPUT_PATH=paste(OUTPUT_DIR,"m4_stl_monthly_industry",sep='/')
    }
    else if(idr <= 47723 && idr > 36736) {
      OUTPUT_PATH=paste(OUTPUT_DIR,"m4_stl_monthly_finance",sep='/')
    }
    else if(idr > 47723) {
      OUTPUT_PATH=paste(OUTPUT_DIR,"m4_stl_monthly_other",sep='/')
    }

    OUTPUT_PATH=paste(OUTPUT_PATH,max_forecast_horizon,sep='')
    OUTPUT_PATH=paste(OUTPUT_PATH,'i',input_size,sep='')
    if(validation)
    {
      OUTPUT_PATH=paste(OUTPUT_PATH,'v',sep='')
    }

    OUTPUT_PATH=paste(OUTPUT_PATH,'txt',sep='.')

    time_series = m4_dataset[idr, ]
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

    # for (inn in input_size:(time_series_length-max_forecast_horizon)) {
    #   level=stl_result[inn, 2] #last "trend" point in the input window is the "level" (the value used for the normalization)
    #   sav_df=data.frame(id=paste(idr,'|i',sep=''));
    #
    #
    #   sav_df[,paste('r',(1:input_size),sep='')] = stl_result[(inn - input_size + 1) : inn, 3] - level #inputs: past values normalized by the level
    #
    #   sav_df[,'o']='|o'
    #   sav_df[,paste('o',(1:max_forecast_horizon),sep='')] = stl_result[(inn + 1) : (inn + max_forecast_horizon), 3] - level #outputs: future values normalized by the level.
    #
    #    if(validation){
    #     sav_df[,'nyb']='|#' #Not Your Business :-) Anything after '|#' is treated as a comment by CNTK's (unitil next bar)
    #                  #What follows is data that CNTK is not supposed to "see". We will use it in the validation R script.
    #     sav_df[,'level']=level
    #     sav_df[,paste('s',(1:max_forecast_horizon),sep='')]=stl_result[(inn+1) : (inn+max_forecast_horizon),1]
    #    }
      write.table(sav_df, file=OUTPUT_PATH, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)

    } #steps
    print(idr)
  }#through all series from one file
}