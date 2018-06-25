file <-read.csv(file="NN5_FINAL_DATASET_WITH_TEST_DATA_2.txt",sep=',',header = FALSE) 
nn5_dataset <-as.data.frame(t(file[,-1]))

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


save56_df = NULL
INPUT_SIZE_MULTIP=1.25
maxForecastHorizon=56

#Processing for prediction horizon 56
for (idr in 1:nrow(nn5_dataset)) {
  oneLine_df= nn5_dataset[idr,]
  y=as.numeric(oneLine_df)
  y=y+1
  n=length(y)
  ylog=log(y)
  
  stlAdj= tryCatch({
    sstl=stl(ts(ylog,frequency=7),"period")
    seasonal_vect=as.numeric(sstl$time.series[,1])
    nnLevels=as.numeric(sstl$time.series[,2])
    nn_vect=as.numeric(sstl$time.series[,2]+sstl$time.series[,3]) # this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
    cbind(seasonal_vect,nnLevels,nn_vect)
  }, error = function(e) { 
    seasonal_vect=rep(0,length(ylog))   #stl() may fail, and then we would go on with the seasonality vector=0
    nnLevels=ylog
    nn_vect=ylog
    cbind(seasonal_vect ,nnLevels,nn_vect)
  })
  
  seasonality_56 = tryCatch({
    seasonality_56 = stlf(ts(ylog , frequency = 7), "period",h=7)
    seasonality_56_vector = as.numeric(seasonality_56$seasonal)
    seasonality_56_vector = rep(seasonality_56_vector,times=8)
    cbind(seasonality_56_vector)
  }, error = function(e) {
    seasonality_56_vector  = rep(0,56)   #stl() may fail, and then we would go on with the seasonality vector=0
    cbind(seasonality_56_vector)
  })
  
  inputSize=as.integer(INPUT_SIZE_MULTIP*maxForecastHorizon)
  inn=inputSize
   for (inn in inputSize:(n)) {
    level = stlAdj[inn, 2] #last "trend" point in the input window is the "level" (the value used for the normalization)
    sav_df = data.frame(id = paste(idr, '|i', sep = ''))
    
    
    for (ii in 1:inputSize) {
      sav_df[, paste('r', ii, sep = '')] = stlAdj[inn - inputSize + ii, 3] - level  #inputs: past values normalized by the level
    }
  
  #stlAdj = tail(stlAdj , n = 70)
  #level = stlAdj[70, 2] #last "trend" point in the input window is the "level" (the value used for the normalization)
  
  #sav_df = data.frame(id = paste(idr, '|i', sep = ''))
  
  #for (ii in 1:70) {
  #  sav_df[, paste('r', ii, sep = '')] = stlAdj[ii, 3] - level  #inputs: past values normalized by the level
 # }
  
  sav_df[, 'nyb'] = '|#' #Not Your Business :-) Anything after '|#' is treated as a comment by CNTK's (unitil next bar)
  #What follows is data that CNTK is not supposed to "see". We will use it in the validation R script.
  
  sav_df[, 'level'] = level
  
  for (ii in 1:56) {
    sav_df[, paste('s', ii, sep = '')] = seasonality_56[ii]
  }
  
  if (is.null(save56_df)) {
    save56_df = sav_df
  } else {
    save56_df = rbind(save56_df, sav_df)
    
  }
 }
  print(idr)
}

write.table(
  save56_df,
  file = "nn5_all_test.txt",
  row.names = F,
  col.names = F,
  sep = " ",
  quote = F
)