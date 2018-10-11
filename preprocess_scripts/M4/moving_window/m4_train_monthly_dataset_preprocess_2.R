library(forecast)

OUTPUT_DIR = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/M4/"

file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/M4/Monthly-train.csv"
m4_dataset <- readLines(file)
m4_dataset <- strsplit(m4_dataset, ',')
print(typeof(m4_dataset))

# m4_dataset <- as.data.frame(file[- 1][-])

seasonality_train_output_path = paste(OUTPUT_DIR, "m4_seasonality_train.txt", sep='')
trend_train_output_path = paste(OUTPUT_DIR, "m4_trend_train.txt", sep='')
remainder_train_output_path = paste(OUTPUT_DIR, "m4_remainder_train.txt", sep='')

seasonality_test_output_path = paste(OUTPUT_DIR, "m4_seasonality_test.txt", sep='')
trend_test_output_path = paste(OUTPUT_DIR, "m4_trend_test.txt", sep='')
remainder_test_output_path = paste(OUTPUT_DIR, "m4_remainder_test.txt", sep='')

predicted_seasonality_output_path = paste(OUTPUT_DIR, "m4_seasonality_predicted.txt", sep='')

close(file(seasonality_train_output_path, open = "w"))
close(file(trend_train_output_path, open = "w"))
close(file(remainder_train_output_path, open = "w"))
close(file(seasonality_test_output_path, open = "w"))
close(file(trend_test_output_path, open = "w"))
close(file(remainder_test_output_path, open = "w"))
close(file(predicted_seasonality_output_path, open = "w"))

max_forecast_horizon = 18
seasonality_period = 12

for (idr in 2 : length(m4_dataset)) {

    time_series = unlist(m4_dataset[idr], use.names = FALSE)
    time_series_log_test = log(as.numeric(time_series[2 : length(time_series)]))
    time_series_length_test = length(time_series_log_test)
    time_series_length_train = time_series_length_test - max_forecast_horizon
    time_series_log_train = time_series_log_test[1 : time_series_length_train]

    # apply stl for the training set
    stl_result_train = tryCatch({
        sstl = stl(ts(time_series_log_train, frequency = seasonality_period), "period")
        seasonal_vect = as.numeric(sstl$time.series[, 1])
        levels_vect = as.numeric(sstl$time.series[, 2])
        values_vect = as.numeric(sstl$time.series[, 2] + sstl$time.series[, 3])# this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
        cbind(seasonal_vect, levels_vect, values_vect)
    }, error = function(e) {
        seasonal_vect = rep(0, length(time_series_length_train))#stl() may fail, and then we would go on with the seasonality vector=0
        levels_vect = time_series_log_train
        values_vect = time_series_log_train
        cbind(seasonal_vect, levels_vect, values_vect)
    })
    # write to files
    write.table(t(stl_result_train[, 1]), file = seasonality_train_output_path, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
    write.table(t(stl_result_train[, 2]), file = trend_train_output_path, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
    write.table(t(stl_result_train[, 3]), file = remainder_train_output_path, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)

    # apply stl for the test set
    stl_result_test = tryCatch({
        sstl = stl(ts(time_series_log_test, frequency = seasonality_period), "period")
        seasonal_vect = as.numeric(sstl$time.series[, 1])
        levels_vect = as.numeric(sstl$time.series[, 2])
        values_vect = as.numeric(sstl$time.series[, 2] + sstl$time.series[, 3]) # this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
        cbind(seasonal_vect, levels_vect, values_vect)
    }, error = function(e) {
        seasonal_vect = rep(0, length(time_series_length_test))   #stl() may fail, and then we would go on with the seasonality vector=0
        levels_vect = time_series_log_test
        values_vect = time_series_log_test
        cbind(seasonal_vect, levels_vect, values_vect)
    })

    # write to files
    write.table(t(stl_result_test[, 1]), file = seasonality_test_output_path, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
    write.table(t(stl_result_test[, 2]), file = trend_test_output_path, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
    write.table(t(stl_result_test[, 3]), file = remainder_test_output_path, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)

    # make seasonality predictions for the test set
    seasonality_test = tryCatch({
        forecast = stlf(ts(stl_result_test[, 1] , frequency = seasonality_period), "period", h = max_forecast_horizon)
        seasonality_vector = as.numeric(forecast$mean)
        c(seasonality_vector)
    }, error = function(e) {
        seasonality_vector = rep(0, max_forecast_horizon)   #stl() may fail, and then we would go on with the seasonality vector=0
        c(seasonality_vector)
    })

    #write to file
    write.table(t(seasonality_test), file = predicted_seasonality_output_path, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}