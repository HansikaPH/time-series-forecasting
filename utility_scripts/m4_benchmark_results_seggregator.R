OUTPUT_DIR = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/ensemble_errors/"

mase_ets_file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/ensemble_errors/merged_cluster_results/all_mase_errors_m4_ets.txt"
smape_ets_file = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/results/ensemble_errors/merged_cluster_results/all_smape_errors_m4_ets.txt"

mase_ets <- readLines(mase_ets_file)
smape_ets <- readLines(smape_ets_file)

for (idr in 1 : length(mase_ets)) {
    if (idr <= 10016 && idr >= 1) { #Macro Series
        OUTPUT_PATH_1 = paste(OUTPUT_DIR, "all_smape_errors_m4_macro_ets.txt", sep = '/')
        OUTPUT_PATH_2 = paste(OUTPUT_DIR, "all_mase_errors_m4_macro_ets.txt", sep = '/')
    }
    else if (idr <= 20991 && idr > 10016) {
        OUTPUT_PATH_1 = paste(OUTPUT_DIR, "all_smape_errors_m4_micro_ets.txt", sep = '/')
        OUTPUT_PATH_2 = paste(OUTPUT_DIR, "all_mase_errors_m4_micro_ets.txt", sep = '/')
    }
    else if (idr <= 26719 && idr > 20991) {
        OUTPUT_PATH_1 = paste(OUTPUT_DIR, "all_smape_errors_m4_demo_ets.txt", sep = '/')
        OUTPUT_PATH_2 = paste(OUTPUT_DIR, "all_mase_errors_m4_demo_ets.txt", sep = '/')
    }
    else if (idr <= 36736 && idr > 26719) {
        OUTPUT_PATH_1 = paste(OUTPUT_DIR, "all_smape_errors_m4_industry_ets.txt", sep = '/')
        OUTPUT_PATH_2 = paste(OUTPUT_DIR, "all_mase_errors_m4_industry_ets.txt", sep = '/')
    }
    else if (idr <= 47723 && idr > 36736) {
        OUTPUT_PATH_1 = paste(OUTPUT_DIR, "all_smape_errors_m4_finance_ets.txt", sep = '/')
        OUTPUT_PATH_2 = paste(OUTPUT_DIR, "all_mase_errors_m4_finance_ets.txt", sep = '/')
    }
    else if (idr > 47723) {
        OUTPUT_PATH_1 = paste(OUTPUT_DIR, "all_smape_errors_m4_other_ets.txt", sep = '/')
        OUTPUT_PATH_2 = paste(OUTPUT_DIR, "all_mase_errors_m4_other_ets.txt", sep = '/')
    }

    write.table(smape_ets[idr], file = OUTPUT_PATH_1, row.names = F, col.names = F, sep = "", quote = F, append = TRUE)
    write.table(mase_ets[idr], file = OUTPUT_PATH_2, row.names = F, col.names = F, sep = "", quote = F, append = TRUE)
}