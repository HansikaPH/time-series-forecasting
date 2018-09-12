library(forecast)
cif_df = read.csv(file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/CIF_2016/cif-dataset.txt", sep = ';', header = FALSE)

names(cif_df)[4:ncol(cif_df)] = paste('x', (1:(ncol(cif_df) - 3)), sep =
                                        '_')
names(cif_df)[1] = "Series"
names(cif_df)[2] = "maxPredHorizon"

forecast_6_input_size = 7

#Select Time series with prediction horizon 12
cif_df_12 = cif_df[cif_df$maxPredHorizon == 12, ]

#Select Time series with prediction horizon 6
cif_df_6 = cif_df[cif_df$maxPredHorizon == 6, ]

maxForecastHorizon_12 = 12
maxForecastHorizon_6 = 6
INPUT_SIZE_MULTIP = 1.25

save12_df = NULL
save6_df = NULL

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
    forecast = stlf(ts(stlAdj[,1] , frequency = 12), "period", h = 12)
    seasonality_12_vector = as.numeric(forecast$mean)
    c(seasonality_12_vector)
  }, error = function(e) {
      seasonality_12_vector = rep(0, 12)   #stl() may fail, and then we would go on with the seasonality vector=0
      c(seasonality_12_vector)
    })

  inputSize = as.integer(INPUT_SIZE_MULTIP * maxForecastHorizon_12)

  print(series)
  for (inn in inputSize:(n)) {
    level = stlAdj[inn, 2] #last "trend" point in the input window is the "level" (the value used for the normalization)
    sav_df = data.frame(id = paste(idr, '|i', sep = ''))


    for (ii in 1:inputSize) {
      sav_df[, paste('r', ii, sep = '')] = stlAdj[inn - inputSize + ii, 3] - level  #inputs: past values normalized by the level
    }

    sav_df[, 'nyb'] = '|#' #Not Your Business :-) Anything after '|#' is treated as a comment by CNTK's (unitil next bar)
    #What follows is data that CNTK is not supposed to "see". We will use it in the validation R script.

    sav_df[, 'level'] = level

    for (ii in 1:maxForecastHorizon_12) {
      sav_df[, paste('s', ii, sep = '')] = seasonality_12[ii]
    }


    if (is.null(save12_df)) {
    save12_df = sav_df
    } else {
    save12_df = rbind(save12_df, sav_df)
    }
  }
}

#Processing for prediction horizon 6
for (idr in 1:nrow(cif_df_6)) {
  oneLine_df = cif_df_6[idr, ]
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

  seasonality_6 = tryCatch({
    forecast = stlf(ts(stlAdj[,1] , frequency = 12), "period", h = 6)
    seasonality_6_vector = as.numeric(forecast$mean)
    c(seasonality_6_vector)
  }, error = function(e) {
    seasonality_6_vector = rep(0, 6)   #stl() may fail, and then we would go on with the seasonality vector=0
    c(seasonality_6_vector)
  })

  inputSize = forecast_6_input_size

  print(series)
  for (inn in inputSize:(n)) {
    level = stlAdj[inn, 2] #last "trend" point in the input window is the "level" (the value used for the normalization)
    sav_df = data.frame(id = paste(idr, '|i', sep = ''))


    for (ii in 1:inputSize) {
      sav_df[, paste('r', ii, sep = '')] = stlAdj[inn - inputSize + ii, 3] - level  #inputs: past values normalized by the level
    }

    sav_df[, 'nyb'] = '|#' #Not Your Business :-) Anything after '|#' is treated as a comment by CNTK's (unitil next bar)
    #What follows is data that CNTK is not supposed to "see". We will use it in the validation R script.

    sav_df[, 'level'] = level

    for (ii in 1:maxForecastHorizon_6) {
      sav_df[, paste('s', ii, sep = '')] = seasonality_6[ii]
    }

    if (is.null(save6_df)) {
      save6_df = sav_df
    } else {
      save6_df = rbind(save6_df, sav_df)
    }
  }
}


write.table(
  save12_df,
  file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/CIF_2016/moving_window/cif12test.txt",
  row.names = F,
  col.names = F,
  sep = " ",
  quote = F
)

write.table(
  save6_df,
  file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/CIF_2016/moving_window/cif6test.txt",
  row.names = F,
  col.names = F,
  sep = " ",
  quote = F
)
