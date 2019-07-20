library(forecast)

OUTPUT_DIR = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/Electricity/moving_window"

file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/Electricity/original_electricity_data.csv"
electricity_dataset <- readLines(file)
electricity_dataset <- strsplit(electricity_dataset, ',')

max_forecast_horizon = 12
seasonality_period = 12
INPUT_SIZE_MULTIP = 1.25
input_size = round(seasonality_period * INPUT_SIZE_MULTIP)

# indices = list(1,    9,   26,   32,   34,   35,   49,   62,   84,  100,  137,
#         166,  167,  185,  240,  272,  289,  337,  378,  393,  395,  404,
#         427,  445,  449,  470,  482,  516,  547,  570,  576,  602,  613,
#         626,  637,  649,  659,  674,  682,  684,  701,  711,  723,  754,
#         762,  768,  793,  796,  798,  801,  811,  830,  866,  882,  900,
#         949,  963,  969, 1028, 1059, 1066, 1075, 1096, 1130, 1136, 1149,
#        1166, 1238, 1242, 1252, 1270, 1302, 1306, 1316, 1330, 1345, 1350,
#        1374, 1437, 1455, 1456, 1498, 1514, 1537, 1571, 1576, 1577, 1587,
#        1730, 1744, 1799, 1812, 1817, 1825, 1826, 1831, 1838, 1842, 1856,
#        1862, 1879, 1885, 1904, 1960, 1977, 1982, 1987, 1997, 2026, 2053,
#        2071, 2083, 2103, 2142, 2179, 2183, 2200, 2210, 2218, 2256, 2334,
#        2337, 2353, 2380, 2404, 2409, 2480, 2484, 2486, 2487, 2491, 2512,
#        2516, 2534, 2634, 2656, 2694, 2708, 2729, 2738, 2761, 2767, 2899,
#        2908, 2952, 3042, 3081, 3121, 3146, 3191, 3282, 3344, 3363, 3393,
#        3397, 3398, 3408, 3443, 3446)

for (idr in 1 : length(electricity_dataset)) {
    time_series = unlist(electricity_dataset[idr], use.names = FALSE)
    time_series_log = log((as.numeric(time_series[1 : length(time_series)]) + 1))
    # time_series_log = BoxCox((as.numeric(time_series[1 : length(time_series)]) + 1), lambda=-0.7)
    # if (!(idr %in% indices)) {
        # time_series_log = BoxCox((as.numeric(time_series[1 : length(time_series)]) + 1), lambda=-0.7)
        OUTPUT_PATH = paste(OUTPUT_DIR, "electricity_test_", sep = '/')
        # }
        # else{
        #
        #     OUTPUT_PATH=paste(OUTPUT_DIR,"electricity_test_level_group2_",sep='/')
        # }
        # OUTPUT_PATH=paste(OUTPUT_DIR,"electricity_test_level_mean_normalized_",sep='/')
        OUTPUT_PATH = paste(OUTPUT_PATH, max_forecast_horizon, sep = '')
        OUTPUT_PATH = paste(OUTPUT_PATH, 'i', input_size, sep = '')

        OUTPUT_PATH = paste(OUTPUT_PATH, 'txt', sep = '.')

        time_series_length = length(time_series_log)

        stl_result = tryCatch({
            sstl = stl(ts(time_series_log, frequency = seasonality_period), "period")
            seasonal_vect = as.numeric(sstl$time.series[, 1])
            levels_vect = as.numeric(sstl$time.series[, 2])
            values_vect = as.numeric(sstl$time.series[, 2] + sstl$time.series[, 3]) # this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
            cbind(seasonal_vect, levels_vect, values_vect)
        }, error = function(e) {
            seasonal_vect = rep(0, length(time_series_log))   #stl() may fail, and then we would go on with the seasonality vector=0
            levels_vect = time_series_log
            values_vect = time_series_log
            cbind(seasonal_vect, levels_vect, values_vect)
        })


        seasonality = tryCatch({
            forecast = stlf(ts(stl_result[, 1] , frequency = seasonality_period), "period", h = max_forecast_horizon)
            seasonality_vector = as.numeric(forecast$mean)
            cbind(seasonality_vector)
        }, error = function(e) {
            seasonality_vector = rep(0, max_forecast_horizon)   #stl() may fail, and then we would go on with the seasonality vector=0
            cbind(seasonality_vector)
        })

        # min-max normalization of data
        # min = min(stl_result[,3])
        # max = max(stl_result[,3])
        # mean = mean(stl_result[,3])
        # series = (stl_result[,3] - min)/(max - min)
        # series = stl_result[,3]/mean
        # input_windows = embed(series[1 : time_series_length], input_size)[, input_size : 1]
        # input_windows = embed(stl_result[1 : time_series_length , 3], input_size)[, input_size : 1]
        level_values = stl_result[input_size : time_series_length, 2]
        input_windows = embed(time_series_log[1 : time_series_length], input_size)[, input_size : 1]
        # means = rowMeans(input_windows)
        input_windows = input_windows - level_values
        # input_windows = input_windows / means

        sav_df = matrix(NA, ncol = (3 + input_size + max_forecast_horizon), nrow = length(level_values))
        sav_df = as.data.frame(sav_df)

        sav_df[, 1] = paste(idr, '|i', sep = '')
        sav_df[, 2 : (input_size + 1)] = input_windows

        sav_df[, (input_size + 2)] = '|#'
        sav_df[, (input_size + 3)] = level_values
        # sav_df[, (input_size + 3)] = means
        #   sav_df[, (input_size + 3)] = rep(mean, length(level_values))
        #   sav_df[, (input_size + 3)] = rep(min, length(level_values))
        #   sav_df[, (input_size + 4)] = rep(max, length(level_values))

        seasonality_windows = matrix(rep(t(seasonality), each = length(level_values)), nrow = length(level_values))
        sav_df[(input_size + 4) : ncol(sav_df)] = seasonality_windows

        write.table(sav_df, file = OUTPUT_PATH, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
    # }
}#through all series from one file