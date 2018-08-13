# Data preparation script
# Slawek Smyl, Feb-Sep 2016
# This script produces 4 files: training and validation files for 6 steps prediction horizon and, similarly, training and validation files for 12 steps prediction horizon.
library(forecast)
OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/CIF_2016/non_moving_window/"
INPUT_FILE="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/CIF_2016/cif-dataset.txt"

cif_df=read.csv(file=INPUT_FILE,sep=';',header = FALSE)

names(cif_df)[4:ncol(cif_df)]=paste('x',(1:(ncol(cif_df)-3)),sep='_')
names(cif_df)[1]="Series"
names(cif_df)[2]="maxPredHorizon"
str(cif_df); #summary(cif_df);

OUTPUT_P6=paste(OUTPUT_DIR,"cif_6_seq2seq_test.txt",sep='/')
OUTPUT_P12=paste(OUTPUT_DIR,"cif_12_seq2seq_test.txt",sep='/')

unlink(OUTPUT_P6);unlink(OUTPUT_P12)

for (idr in 1: nrow(cif_df)) {
  oneLine_df=cif_df[idr,]
  series=as.character(oneLine_df$Series)

    maxForecastHorizon= oneLine_df$maxPredHorizon

    y=as.numeric(oneLine_df[4:(ncol(oneLine_df))])
    y=y[!is.na(y)]
    ylog=log(y)
    #str(y); plot(ylog)
    n=length(y)
    stlAdj= tryCatch({
        sstl=stl(ts(ylog,frequency=12),"period")
            seasonal_vect=as.numeric(sstl$time.series[,1])
            nnLevels=as.numeric(sstl$time.series[,2])
            nn_vect=as.numeric(sstl$time.series[,2]+sstl$time.series[,3]) # this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
            cbind(seasonal_vect,nnLevels,nn_vect)
      }, error = function(e) {
            seasonal_vect=rep(0,length(ylog))   #stl() may fail, and then we would go on with the seasonality vector=0
            nnLevels=ylog
            nn_vect=ylog
            cbind(seasonal_vect,nnLevels,nn_vect)
        })

    seasonality = tryCatch({
    forecast = stlf(ts(stlAdj[,1] , frequency = 12), "period", h = maxForecastHorizon)
    seasonality_vector = as.numeric(forecast$mean)
    c(seasonality_vector)
    }, error = function(e) {
      seasonality_vector = rep(0, maxForecastHorizon)   #stl() may fail, and then we would go on with the seasonality vector=0
      c(seasonality_vector)
    })

    print(series)

    sav_df=data.frame(id=paste(idr,'|i',sep='')); #sav_df is the set of input values in the current window
    level=stlAdj[n,2]
    normalized_values = stlAdj[,3]-level
    sav_df=cbind(sav_df, t(normalized_values[1: n])) #inputs: past values normalized by the level
    print(sav_df)
    sav_df[,'nyb']='|#' #Not Your Business :-) Anything after '|#' is treated as a comment by CNTK's (unitil next bar)
                 #What follows is data that CNTK is not supposed to "see". We will use it in the validation R script.
    sav_df[,'level']=level
    sav_df = cbind(sav_df, t(seasonality))

    if (maxForecastHorizon==6) {
        write.table(sav_df, file=OUTPUT_P6, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)
    } else {
        write.table(sav_df, file=OUTPUT_P12, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)
    }

}#through all series from one file
