library('tsfeatures')
library('dplyr')

input_file = "./datasets/text_data/Solar_Data/df.csv"
frequency = 12
output_file = "./results/seasonality_strengths/electricity.txt"

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