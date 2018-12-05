library(forecast)

args <- commandArgs(trailingOnly = TRUE)

OUTPUT_DIR = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/moving_window/without_stl_decomposition"

file <- read.csv(file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/NN5/nn5_dataset.txt", sep = ',', header = FALSE)
nn5_dataset <- as.data.frame(file)

max_forecast_horizon = 56
seasonality_period = 7
INPUT_SIZE_MULTIP = 1.25
input_size = round(seasonality_period * INPUT_SIZE_MULTIP)

OUTPUT_PATH56 = paste(OUTPUT_DIR, "nn5_test_", sep = '/')
OUTPUT_PATH56 = paste(OUTPUT_PATH56, max_forecast_horizon, sep = '')
OUTPUT_PATH56 = paste(OUTPUT_PATH56, 'i', input_size, sep = '')

OUTPUT_PATH56 = paste(OUTPUT_PATH56, 'txt', sep = '.')
unlink(OUTPUT_PATH56)

numeric_dataset = as.matrix(as.data.frame(lapply(nn5_dataset, as.numeric)))
numeric_dataset = numeric_dataset + 1

numeric_dataset_log = log(numeric_dataset)

time_series_length = ncol(numeric_dataset_log)

for (idr in 1 : nrow(numeric_dataset_log)) {
    time_series_log = numeric_dataset_log[idr,]

    input_windows = embed(time_series_log[1 : time_series_length], input_size)[, input_size : 1]
    level_values = rowMeans(input_windows)
    input_windows = input_windows - level_values

    sav_df = matrix(NA, ncol = (3 + input_size), nrow = length(level_values))
    sav_df = as.data.frame(sav_df)

    sav_df[, 1] = paste(idr, '|i', sep = '')
    sav_df[, 2 : (input_size + 1)] = input_windows

    sav_df[, (input_size + 2)] = '|#'
    sav_df[, (input_size + 3)] = level_values

    write.table(sav_df, file = OUTPUT_PATH56, row.names = F, col.names = F, sep = " ", quote = F, append = TRUE)
}#through all series from one file