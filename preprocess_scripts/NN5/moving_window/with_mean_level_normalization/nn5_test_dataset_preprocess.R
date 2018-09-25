library(forecast)

args <- commandArgs(trailingOnly = TRUE)

OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/moving_window/with_mean_level_normalization"

file <-read.csv(file="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/NN5_FINAL_DATASET.csv",sep=',',header = FALSE)
nn5_dataset <-as.data.frame(t(file[,-1]))

max_forecast_horizon=56

sunday = vector()
monday = vector()
tuesday = vector()
wednesday = vector()
thursday= vector()
friday = vector()
saturday = vector()
wrong= vector()

#replacing missing values
for (idr in 1: nrow(nn5_dataset)) {
  oneLine_df=nn5_dataset[idr,]
  numericvalue<-as.numeric(oneLine_df)
  for(i in 1:length(numericvalue)){
    if(i%%7==0){
      sunday = append(sunday,numericvalue[i])
    }else if(i%%7==1){
      monday = append(monday,numericvalue[i])
    }else if(i%%7==2){
      tuesday = append(tuesday,numericvalue[i])
    }else if(i%%7==3){
      wednesday  = append(wednesday,numericvalue[i])
    }else if(i%%7==4){
      thursday= append(thursday,numericvalue[i])
    }else if(i%%7==5){
      friday= append(friday,numericvalue[i])
    }else if(i%%7==6){
      saturday= append(saturday,numericvalue[i])
    }else{
      wrong= append(wrong,numericvalue[i])
    }
  }
  print(idr)
}

sunday_median<-median(sunday,na.rm = TRUE)
monday_median <-median(monday,na.rm = TRUE)
tuesday_median <- median(tuesday,na.rm = TRUE)
wednesday_median <-median(wednesday,na.rm = TRUE)
thursday_median<-median(thursday,na.rm = TRUE)
friday_median<-median(friday,na.rm = TRUE)
saturday_median<-median(saturday,na.rm = TRUE)

#replacing missing values
for (idr in 1: nrow(nn5_dataset)) {
  oneLine_df=nn5_dataset[idr,]
  numericvalue<-as.numeric(oneLine_df)
  for(i in 1:length(numericvalue)){
    if(is.na(oneLine_df[i])){
      if(i%%7==0){
        nn5_dataset[idr,i] =sunday_median
      }else if(i%%7==1){
        nn5_dataset[idr,i]= monday_median
      }else if(i%%7==2){
        nn5_dataset[idr,i]= tuesday_median
      }else if(i%%7==3){
        nn5_dataset[idr,i] =wednesday_median
      }else if(i%%7==4){
        nn5_dataset[idr,i] =thursday_median
      }else if(i%%7==5){
        nn5_dataset[idr,i]= friday_median
      }else if(i%%7==6){
        nn5_dataset[idr,i]= saturday_median
      }
    }
  }
}

if(length(args) != 0) {
  input_size = as.integer(args[1])
} else{
  INPUT_SIZE_MULTIP=1.25  # using some reasoning and backesting, I decided to make input size a bit (here by 25%) larger than the maximum prediction horizon
  input_size=as.integer(INPUT_SIZE_MULTIP*max_forecast_horizon)
}

OUTPUT_PATH56=paste(OUTPUT_DIR,"nn5_test_",sep='/')
OUTPUT_PATH56=paste(OUTPUT_PATH56,max_forecast_horizon,sep='')
OUTPUT_PATH56=paste(OUTPUT_PATH56,'i', input_size, sep='')

OUTPUT_PATH56=paste(OUTPUT_PATH56,'txt',sep='.')
unlink(OUTPUT_PATH56)

numeric_dataset = as.matrix(as.data.frame(lapply(nn5_dataset, as.numeric)))
numeric_dataset = numeric_dataset + 1

numeric_dataset_log = log(numeric_dataset)

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


  seasonality_56 = tryCatch({
    forecast = stlf(ts(stl_result[,1] , frequency = 7), "period",h=56)
    seasonality_56_vector = as.numeric(forecast$mean)
    # seasonality_56_vector = rep(seasonality_56_vector,times=8)
    cbind(seasonality_56_vector)
  }, error = function(e) {
    seasonality_56_vector  = rep(0, max_forecast_horizon)   #stl() may fail, and then we would go on with the seasonality vector=0
    cbind(seasonality_56_vector)
  })

  for (inn in input_size:time_series_length) {
    level=stl_result[inn, 2] #last "trend" point in the input window is the "level" (the value used for the normalization)
    sav_df=data.frame(id=paste(idr,'|i',sep=''));

    for (ii in 1:input_size) {
      sav_df[,paste('r',ii,sep='')]=stl_result[inn-input_size+ii,3] - level  #inputs: past values normalized by the mean_level
    }


    sav_df[,'nyb']='|#' #Not Your Business :-) Anything after '|#' is treated as a comment by CNTK's (unitil next bar)
    #What follows is data that CNTK is not supposed to "see". We will use it in the validation R script.
    sav_df[,'level']=level

    for (ii in 1:56) {
      sav_df[, paste('s', ii, sep = '')] = seasonality_56[ii]
    }

    write.table(sav_df, file=OUTPUT_PATH56, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)

  } #steps
  print(idr)
}#through all series from one file