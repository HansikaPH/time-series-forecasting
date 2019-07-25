input_file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/CIF_2016-/m4_seasonality.txt"
OUTPUT_DIR = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/M4/"
m4_dataset <- readLines(input_file)
m4_dataset <- strsplit(m4_dataset, ',')

for (idr in 1 : length(m4_dataset)) {
  if (idr <= 10016 && idr >= 1) { #Macro Series
    OUTPUT_PATH = paste(OUTPUT_DIR, "m4_macro_seasonality.txt", sep = '/')
  }
  else if (idr <= 20991 && idr > 10016) {
    OUTPUT_PATH = paste(OUTPUT_DIR, "m4_micro_seasonality.txt", sep = '/')
  }
  else if (idr <= 26719 && idr > 20991) {
    OUTPUT_PATH = paste(OUTPUT_DIR, "m4_demo_seasonality.txt", sep = '/')
  }
  else if (idr <= 36736 && idr > 26719) {
    OUTPUT_PATH = paste(OUTPUT_DIR, "m4_industry_seasonality.txt", sep = '/')
  }
  else if (idr <= 47723 && idr > 36736) {
    OUTPUT_PATH = paste(OUTPUT_DIR, "m4_finance_seasonality.txt", sep = '/')
  }
  else if (idr > 47723) {
    OUTPUT_PATH = paste(OUTPUT_DIR, "m4_other_seasonality.txt", sep = '/')
  }
  
  time_series = unlist(m4_dataset[idr], use.names = FALSE)
  
  write.table(t(time_series), file = OUTPUT_PATH, row.names = F, col.names = F, sep = ",", quote = F, append = TRUE)
  
}