OUTPUT_DIR = "./datasets/text_data/M4/"

file = "./datasets/text_data/M4/original_m4_dataset.csv"
m4_dataset <- readLines(file)

for (idr in 1 : length(m4_dataset)) {
    if (idr <= 10016 && idr >= 1) { #Macro Series
        OUTPUT_PATH = paste(OUTPUT_DIR, "original_m4_macro.txt", sep = '/')
    }
    else if (idr <= 20991 && idr > 10016) {
        OUTPUT_PATH = paste(OUTPUT_DIR, "original_m4_micro.txt", sep = '/')
    }
    else if (idr <= 26719 && idr > 20991) {
        OUTPUT_PATH = paste(OUTPUT_DIR, "original_m4_demo.txt", sep = '/')
    }
    else if (idr <= 36736 && idr > 26719) {
        OUTPUT_PATH = paste(OUTPUT_DIR, "original_m4_industry.txt", sep = '/')
    }
    else if (idr <= 47723 && idr > 36736) {
        OUTPUT_PATH = paste(OUTPUT_DIR, "original_m4_finance.txt", sep = '/')
    }
    else if (idr > 47723) {
        OUTPUT_PATH = paste(OUTPUT_DIR, "original_m4_other.txt", sep = '/')
    }

    write.table(m4_dataset[idr], file = OUTPUT_PATH, row.names = F, col.names = F, sep = "", quote = F, append = TRUE)
}