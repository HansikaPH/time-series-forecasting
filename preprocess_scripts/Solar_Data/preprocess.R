library(dplyr)

df = read.csv("/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/Solar_Data/new_formatted_dataset.csv")
output_file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/Solar_Data/df.csv"
out = split(df, df$Customer.ID)

for (i in 1 : length(out)){
  print(i)
  write.table(t(out[[i]]$Sum), output_file, append = TRUE, sep = ",",
              row.names = FALSE, col.names = FALSE)
}