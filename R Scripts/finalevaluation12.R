cif_df=read.csv(file="cif-results-dataset.txt",sep=';',header = FALSE)

text_df_12=read.csv(file="cif12test.txt",sep = " ",header = FALSE)
#text_df_6=read.csv(file="test6file.txt",sep = " ",header = FALSE)

names(cif_df)[1]="Series"

cif_df <- cif_df[,-1]

value <- t(text_df_12[1])

indexes <- length(value) - match(unique(value), rev(value)) + 1

uniqueindexes <- unique(indexes)


forcast12_DF <- cif_df[rowSums(is.na(cif_df)) == 0,]
#forcast6_DF <- cif_df[rowSums(is.na(cif_df)) > 0,]

final_forcast_actual <- (forcast12_DF)

#forecastFilePath12="Output/stl_12i15_50/Out.z"
forecast_df_12=read.csv("cif12evaluation.txt", header = F, sep = " ")

#forecastFilePath6="Output6/stl_6i7_50/Out6.z"
#forecast_df_6=read.csv(forecastFilePath6, header = F, sep = " ")


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


# pred6_df = NULL
# pred_6matrix = matrix(nrow = 15, ncol = 6)
# 
# for(k in 1 :nrow(forecast_df_6)){
#   lstm_output_6 = as.numeric(forecast_df_6[k,])
#   oneLine_df_6 = as.numeric(text_df_6[k,]) 
#   
#   level_value_6 = oneLine_df_6[10]
#   seasonal_values_6 = oneLine_df_6[11:length(oneLine_df_6)]
#   
#   for (ii in 1:6) {
#     predicted_value_6 = exp(lstm_output_6[ii]+ level_value_6+ seasonal_values_6[ii]) 
#     pred6_df[ii] =  predicted_value_6 
#   }
#   pred_6matrix[k,] = pred6_df
# }

# 
# pred_6matrix_1=cbind(pred_6matrix,NA)
# pred_6matrix_2=cbind(pred_6matrix_1,NA)
# pred_6matrix_3=cbind(pred_6matrix_2,NA)
# pred_6matrix_4=cbind(pred_6matrix_3,NA)
# pred_6matrix_5=cbind(pred_6matrix_4,NA)
# pred_6matrix_6=cbind(pred_6matrix_5,NA)
# 

final_prediction_matrix = (pred_12matrix)


sMAPEPerSeries1 <- rowMeans(2*abs(final_prediction_matrix-final_forcast_actual)/(abs(final_prediction_matrix)+abs(final_forcast_actual)), na.rm=TRUE)

mean(sMAPEPerSeries1)
sd(sMAPEPerSeries1)