args <- commandArgs(trailingOnly = TRUE)
forecast_file_path = args[1]

cif_df=read.csv(file="/media/hansika/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/cif-results.txt",sep=';',header = FALSE)

text_df_12=read.csv(file="/media/hansika/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/cif12test.txt",sep = " ",header = FALSE)

names(cif_df)[1]="Series"

cif_df <- cif_df[,-1]

# take the transpose of the dataframe
value <- t(text_df_12[1])

indexes <- length(value) - match(unique(value), rev(value)) + 1

uniqueindexes <- unique(indexes)
# print(uniqueindexes)

forcast12_DF <- cif_df[rowSums(is.na(cif_df)) == 0,]
# print(forcast12_DF)

final_forcast_actual <- (forcast12_DF)

forecast_df_12=read.csv(forecast_file_path, header = F, sep = ",")

pred12_df = NULL
pred_12matrix = matrix(nrow = 57, ncol = 12)

for(k in 1 :nrow(forecast_df_12)){
  lstm_output_12 = as.numeric(forecast_df_12[k,])
  finalindex <- uniqueindexes[k]
  oneLine_df_12 = as.numeric(text_df_12[finalindex,]) 
  
  level_value_12 = oneLine_df_12[18]
  seasonal_values_12 = oneLine_df_12[19:length(oneLine_df_12)]
  
  for (ii in 1:12) {
    predicted_value_12 = exp(lstm_output_12[ii]+ level_value_12+ seasonal_values_12[ii])
    pred12_df[ii] =  predicted_value_12 
  }
  pred_12matrix[k,] = pred12_df
}

final_prediction_matrix = (pred_12matrix)
# print(final_prediction_matrix)

sMAPEPerSeries1 <- rowMeans(2*abs(final_prediction_matrix-final_forcast_actual)/(abs(final_prediction_matrix)+abs(final_forcast_actual)), na.rm=TRUE)
# print(sMAPEPerSeries1)

print(mean(sMAPEPerSeries1))
print(sd(sMAPEPerSeries1))