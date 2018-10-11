OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/"

file <-read.csv(file="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/NN5_FINAL_DATASET.csv",sep=',',header = FALSE)
nn5_dataset <-as.data.frame(t(file[,-1]))

output_file_name = 'nn5_dataset.txt'
output_file_full_name = paste(OUTPUT_DIR, output_file_name, sep = '')

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

# printing the dataset to the file
write.table(nn5_dataset, output_file_full_name, sep = ",", row.names = FALSE, col.names = FALSE)
