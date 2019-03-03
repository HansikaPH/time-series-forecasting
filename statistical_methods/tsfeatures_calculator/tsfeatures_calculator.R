library('tsfeatures')
library('dplyr')

# args = commandArgs(trailingOnly=TRUE)
# input_file = args[1]
# output_file = args[2]
# frequency = args[3]

input_file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/Tourism/tourism_data.csv"
frequency = 12
output_file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/seasonality_strengths/tourism.txt"

# read the data
dataset <- readLines(input_file)
dataset <- strsplit(dataset, ',')

seasonality_strengths = list()
for(i in 1:length(dataset)){
    time_series = dataset[i]
    time_series = as.numeric(unlist(time_series))

    seasonal_strength= tryCatch({
        features = tsfeatures(ts(time_series, freq=as.numeric(frequency)), parallel=TRUE)
        select(features, 'seasonal_strength')
    }, error = function(e) {

    })

    seasonality_strengths[i] = list(seasonal_strength)
}

# print(seasonality_strengths)
write.table(seasonality_strengths, output_file, sep="\n", row.names = FALSE, col.names = FALSE)