# Data preparation script
# Slawek Smyl, Feb-Sep 2016
# This script produces 4 files: training and validation files for 6 steps prediction horizon and, similarly, training and validation files for 12 steps prediction horizon. 

# args = commandArgs(trailingOnly=TRUE)
DATA_FILE = "/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/CIF_2016/cif-dataset.txt"
OUTPUT_DIR="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/CIF_2016/moving_window/without_stl_decomposition/"

cif_df=read.csv(file=DATA_FILE,sep=';',header = FALSE)

names(cif_df)[4:ncol(cif_df)]=paste('x',(1:(ncol(cif_df)-3)),sep='_')
names(cif_df)[1]="Series"
names(cif_df)[2]="maxPredHorizon"
str(cif_df); #summary(cif_df);

OUTPUT_P6=paste(OUTPUT_DIR,"cif_6",sep='/')
OUTPUT_P12=paste(OUTPUT_DIR,"cif_12",sep='/')

# if(length(args) == 0) {
	INPUT_SIZE_MULTIP=1.25  # using some reasoning and backesting, I decided to make input size a bit (here by 25%) larger than the maximum prediction horizon

	if (INPUT_SIZE_MULTIP!=1) {
		inputSize_6=as.integer(INPUT_SIZE_MULTIP*6)
		inputSize_12=as.integer(INPUT_SIZE_MULTIP*12)
	}
# }else {
# 	inputSize_6 = 1
# 	inputSize_12 = 1
# }

OUTPUT_P6=paste(OUTPUT_P6,'i',inputSize_6,sep='')
OUTPUT_P12=paste(OUTPUT_P12,'i',inputSize_12,sep='')

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
	
	firstTime=TRUE; save6_df=NULL; save12_df=NULL;
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
		
		# stlAdj= tryCatch({
		#   	sstl=stl(ts(ylog,frequency=12),"period")
		# 		seasonal_vect=as.numeric(sstl$time.series[,1])
		# 		nnLevels=as.numeric(sstl$time.series[,2])
		# 		nn_vect=as.numeric(sstl$time.series[,2]+sstl$time.series[,3]) # this is what we are going to work on: sum of the smooth trend and the random component (the seasonality removed)
		# 		cbind(seasonal_vect,nnLevels,nn_vect)
		#   }, error = function(e) {
		# 		seasonal_vect=rep(0,length(ylog))   #stl() may fail, and then we would go on with the seasonality vector=0
		# 		nnLevels=ylog
		# 		nn_vect=ylog
		# 		cbind(seasonal_vect,nnLevels,nn_vect)
		# 	})

		if(maxForecastHorizon == 6) {
			inputSize = inputSize_6
		}else {
			inputSize = inputSize_12
		}

	  	print(series)

		for (inn in inputSize:(n-maxForecastHorizon)) {

			# if(inputSize != 1) {
				level=ylog[inn] #last "trend" point in the input window is the "level" (the value used for the normalization)
			# }

			sav_df=data.frame(id=paste(idr,'|i',sep='')); #sav_df is the set of input values in the current window

			for (ii in 1:inputSize) {
				# if(inputSize != 1) {
					sav_df[,paste('r',ii,sep='')]=ylog[inn-inputSize+ii]-level  #inputs: past values normalized by the level
				# }else {
				# 	sav_df[,paste('r',ii,sep='')]=stlAdj[inn-inputSize+ii,3]
				# }
			}

			sav_df[,'o']='|o'
			for (ii in 1:maxForecastHorizon) {
				# if(inputSize != 1) {
					sav_df[,paste('o',ii,sep='')]=ylog[inn+ii]-level #outputs: future values normalized by the level.
				# }else {
				# 	sav_df[,paste('o',ii,sep='')]=stlAdj[inn+ii,3]
				# }
			}

			if(validation){
				sav_df[,'nyb']='|#' #Not Your Business :-) Anything after '|#' is treated as a comment by CNTK's (unitil next bar)
							 #What follows is data that CNTK is not supposed to "see". We will use it in the validation R script.
				# if(inputSize != 1)
				# {
					sav_df[,'level']=level
				# }
				# for (ii in 1:maxForecastHorizon) {
				# 	sav_df[,paste('s',ii,sep='')]=ylog[inn+ii]
				# }
			}

			if (maxForecastHorizon==6) {
				if (is.null(save6_df)) {
					save6_df=sav_df
				} else {
					save6_df=rbind(save6_df, sav_df)
				}
			} else {
				if (is.null(save12_df)) {
					save12_df=sav_df 
				} else {
					save12_df=rbind(save12_df, sav_df)
				}
			}
		} #steps
	}#through all series from one file
	
	#str(save6_df); 
	summary(save6_df); 
	#str(save12_df);
	summary(save12_df);
	
	write.table(save6_df, file=OUTPUT_PATH6, row.names = F, col.names=F, sep=" ", quote=F)
	write.table(save12_df, file=OUTPUT_PATH12, row.names = F, col.names=F, sep=" ", quote=F)
}

