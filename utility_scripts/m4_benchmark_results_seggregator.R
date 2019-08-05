OUTPUT_DIR = "./results/ensemble_errors/"

mase_arima_file = "./results/ensemble_errors/merged_cluster_results/all_mase_errors_m4_arima.txt"
smape_arima_file = "./results/ensemble_errors/merged_cluster_results/all_smape_errors_m4_arima.txt"

unlink(paste(OUTPUT_DIR, "all_*_errors_m4_*_arima*", sep=""))

mase_arima <- readLines(mase_arima_file)
smape_arima <- readLines(smape_arima_file)

for (idr in 1 : length(mase_arima)) {
    if (idr <= 10016 && idr >= 1) { #Macro Series
        OUTPUT_PATH_1 = paste(OUTPUT_DIR, "all_smape_errors_m4_macro_arima.txt", sep = '/')
        OUTPUT_PATH_2 = paste(OUTPUT_DIR, "all_mase_errors_m4_macro_arima.txt", sep = '/')
    }
    else if (idr <= 20991 && idr > 10016) {
        OUTPUT_PATH_1 = paste(OUTPUT_DIR, "all_smape_errors_m4_micro_arima.txt", sep = '/')
        OUTPUT_PATH_2 = paste(OUTPUT_DIR, "all_mase_errors_m4_micro_arima.txt", sep = '/')
    }
    else if (idr <= 26719 && idr > 20991) {
        OUTPUT_PATH_1 = paste(OUTPUT_DIR, "all_smape_errors_m4_demo_arima.txt", sep = '/')
        OUTPUT_PATH_2 = paste(OUTPUT_DIR, "all_mase_errors_m4_demo_arima.txt", sep = '/')
    }
    else if (idr <= 36736 && idr > 26719) {
        OUTPUT_PATH_1 = paste(OUTPUT_DIR, "all_smape_errors_m4_industry_arima.txt", sep = '/')
        OUTPUT_PATH_2 = paste(OUTPUT_DIR, "all_mase_errors_m4_industry_arima.txt", sep = '/')
    }
    else if (idr <= 47723 && idr > 36736) {
        OUTPUT_PATH_1 = paste(OUTPUT_DIR, "all_smape_errors_m4_finance_arima.txt", sep = '/')
        OUTPUT_PATH_2 = paste(OUTPUT_DIR, "all_mase_errors_m4_finance_arima.txt", sep = '/')
    }
    else if (idr > 47723) {
        OUTPUT_PATH_1 = paste(OUTPUT_DIR, "all_smape_errors_m4_other_arima.txt", sep = '/')
        OUTPUT_PATH_2 = paste(OUTPUT_DIR, "all_mase_errors_m4_other_arima.txt", sep = '/')
    }

    write.table(smape_arima[idr], file = OUTPUT_PATH_1, row.names = F, col.names = F, sep = "", quote = F, append = TRUE)
    write.table(mase_arima[idr], file = OUTPUT_PATH_2, row.names = F, col.names = F, sep = "", quote = F, append = TRUE)
}