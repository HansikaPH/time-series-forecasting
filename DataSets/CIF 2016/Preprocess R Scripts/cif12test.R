cif_df = read.csv(file = "/media/hansika/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/cif-dataset.txt", sep = ';', header = FALSE)

names(cif_df)[4:ncol(cif_df)] = paste('x', (1:(ncol(cif_df) - 3)), sep =
                                        '_')
names(cif_df)[1] = "Series"
names(cif_df)[2] = "maxPredHorizon"

SIZE_12 = 12
SIZE_12_MUL = 15

SIZE_6 = 6
SIZE_6_MUL = 7

#cif_df<-cif_df[rows,]

#Select Time series with prediction horizon 12
cif_df_12 = cif_df[cif_df$maxPredHorizon == 12, ]
maxForecastHorizon = 12
INPUT_SIZE_MULTIP = 1.25

firstTime = TRUE
save12_df = NULL

#Processing for prediction horizon 12
for (idr in 1:nrow(cif_df_12)) {
  oneLine_df = cif_df_12[idr, ]
  series = as.character(oneLine_df$Series)
  y = as.numeric(oneLine_df[4:(ncol(oneLine_df))])
  y = y[!is.na(y)]
  ylog = log(y)
  n = length(y)
  
  stlAdj = tryCatch({
    sstl = stl(ts(ylog, frequency = 12), "period")
    seasonal_vect = as.numeric(sstl$time.series[, 1])
    nnLevels = as.numeric(sstl$time.series[, 2])
    nn_vect = as.numeric(sstl$time.series[, 2] + sstl$time.series[, 3]) # this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
    cbind(seasonal_vect, nnLevels, nn_vect)
  }, error = function(e) {
    seasonal_vect = rep(0, length(ylog))   #stl() may fail, and then we would go on with the seasonality vector=0
    nnLevels = ylog
    nn_vect = ylog
    cbind(seasonal_vect, nnLevels, nn_vect)
  })
  
  seasonality_12 = tryCatch({
    seasonality_12 = stlf(ts(ylog , frequency = 12), "period", h = 12)
    seasonality_12_vector = as.numeric(seasonality_12$seasonal)
    cbind(seasonality_12_vector)
  }, error = function(e) {
    seasonality_12_vector = rep(0, 12)   #stl() may fail, and then we would go on with the seasonality vector=0
    cbind(seasonality_12_vector)
  })
  
  #stlAdj = tail(stlAdj , n = 15)
  #level = stlAdj[15, 2] #last "trend" point in the input window is the "level" (the value used for the normalization)
  #sav_df = data.frame(id = paste(idr, '|i', sep = ''))
  
  inputSize = as.integer(INPUT_SIZE_MULTIP * maxForecastHorizon)
  
  print(series)
  inn = inputSize
  for (inn in inputSize:(n)) {
    level = stlAdj[inn, 2] #last "trend" point in the input window is the "level" (the value used for the normalization)
    sav_df = data.frame(id = paste(idr, '|i', sep = ''))
    
    
    for (ii in 1:inputSize) {
      sav_df[, paste('r', ii, sep = '')] = stlAdj[inn - inputSize + ii, 3] - level  #inputs: past values normalized by the level
    }
    
    sav_df[, 'nyb'] = '|#' #Not Your Business :-) Anything after '|#' is treated as a comment by CNTK's (unitil next bar)
    #What follows is data that CNTK is not supposed to "see". We will use it in the validation R script.
    
    sav_df[, 'level'] = level
    
    for (ii in 1:12) {
      sav_df[, paste('s', ii, sep = '')] = seasonality_12[ii]
    }
    
    if (is.null(save12_df)) {
      save12_df = sav_df
    } else {
      save12_df = rbind(save12_df, sav_df)
    }
  }
}

write.table(
  save12_df,
  file = "/media/hansika/Academic/Monash University/Research Project/Codes/time-series-forecasting/DataSets/CIF 2016/cif12test.txt",
  row.names = F,
  col.names = F,
  sep = " ",
  quote = F
)
