# Data preparation script
# Slawek Smyl, Feb-Sep 2016
# This script produces 4 files: training and validation files for 6 steps prediction horizon and, similarly, training and validation files for 12 steps prediction horizon. 

OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/CIF_2016/non_moving_window/"
INPUT_FILE="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/CIF_2016/cif-dataset.txt"

cif_df=read.csv(file=INPUT_FILE,sep=';',header = FALSE)

names(cif_df)[4:ncol(cif_df)]=paste('x',(1:(ncol(cif_df)-3)),sep='_')
names(cif_df)[1]="Series"
names(cif_df)[2]="maxPredHorizon"
str(cif_df); #summary(cif_df); 

OUTPUT_P6=paste(OUTPUT_DIR,"stl_6_seq2seq",sep='/')
OUTPUT_P12=paste(OUTPUT_DIR,"stl_12_seq2seq",sep='/')

#The validation file constains the training file, although only last record per each series in the validation file is used for calculating the metrics. 
#This is becasue we are using the recurrent networks with potentially long memory (LSTMs), so all the records are needed for "warm-up" or establishment of the state.  
for (validation in c(TRUE,FALSE)) {# 
	OUTPUT_PA6=OUTPUT_P6
	OUTPUT_PA12=OUTPUT_P12
	if (validation) {
		OUTPUT_PA6=paste(OUTPUT_PA6,'v',sep='')
		OUTPUT_PA12=paste(OUTPUT_PA12,'v',sep='')
	}
	OUTPUT_PATH6=paste(OUTPUT_PA6,'txt',sep='.')
	OUTPUT_PATH12=paste(OUTPUT_PA12,'txt',sep='.')
	
	unlink(OUTPUT_PATH6);unlink(OUTPUT_PATH12)

	for (idr in 1: nrow(cif_df)) {
	  oneLine_df=cif_df[idr,]
	  series=as.character(oneLine_df$Series) 
	
		maxForecastHorizon= oneLine_df$maxPredHorizon

		y=as.numeric(oneLine_df[4:(ncol(oneLine_df))])
		y=y[!is.na(y)]
		ylog=log(y)
		#str(y); plot(ylog)
		n=length(y)
		if (!validation) {
			n=n-maxForecastHorizon
			ylog=ylog[1:n]
		}
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
		#plot(ylog); lines(stlAdj[,3]); lines(stlAdj[,2],col='blue'); lines(stlAdj[,3]+stlAdj[,1],col=2)

		print(series)

		sav_df=data.frame(id=paste(idr,'|i',sep='')); #sav_df is the set of input values in the current window
		level=stlAdj[n-maxForecastHorizon,2]
		normalized_values = stlAdj[,3]-level
		sav_df=cbind(sav_df, t(normalized_values[1: (n-maxForecastHorizon)])) #inputs: past values normalized by the level
		print(sav_df)
		sav_df[,'o']='|o'
		sav_df=cbind(sav_df, t(normalized_values[(n - maxForecastHorizon + 1) :length(normalized_values)])) #outputs: future values normalized by the level.
		if(validation){
			sav_df[,'nyb']='|#' #Not Your Business :-) Anything after '|#' is treated as a comment by CNTK's (unitil next bar)
						 #What follows is data that CNTK is not supposed to "see". We will use it in the validation R script.
			sav_df[,'level']=level
			sav_df = cbind(sav_df, t(stlAdj[(n - maxForecastHorizon + 1) : n,1]))
		}

		if (maxForecastHorizon==6) {
			write.table(sav_df, file=OUTPUT_PATH6, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)
		} else {
			write.table(sav_df, file=OUTPUT_PATH12, row.names = F, col.names=F, sep=" ", quote=F, append = TRUE)
		}

	}#through all series from one file
}

