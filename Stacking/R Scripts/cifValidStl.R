# validation script
# Slawek Smyl, Feb-Sep 2016
# This script post processes the CNTK output (forecast) file. for 12-long prediction horizon output.
# It calculates error (sMAPE) and graphs a sample series with the forecast

options(width=200)
validationFilePath="data/stl_12i15v.txt" 
validation_df=read.csv(validationFilePath, header = F, sep = " ")

cif_df=read.csv(file="cif-dataset.txt",sep=';',header = FALSE)
names(cif_df)[4:ncol(cif_df)]=paste('x',(1:(ncol(cif_df)-3)),sep='_')
names(cif_df)[1]="Series"
names(cif_df)[2]="maxPredHorizon"
str(cif_df); #head(cif_df)
#summary(cif_df); 

SIZE=12 #max prediction horizon, either 12 or 6
cif_df=cif_df[cif_df$maxPredHorizon==SIZE,]
INPUT_SIZE=15 #for prediction horizon=12, this would be 7 for prediction horizon=6

forecastFilePath="Output/stl_12i15_50/Out.z"
forecast_df=read.csv(forecastFilePath, header = F, sep = " ")

sumMPE=0; sumSMape=0; 
maxSMape=0; indexOfMaxSMape=0; its=0
listOfInterestingRows=list()
# validation_df[1:100,1:20]
prevSeries=validation_df[1,1]; ir=83
for (ir in 1:nrow(validation_df)) {
	if (validation_df[ir,1]!=prevSeries) {#previous row was the last one for a series
		its=its+1; listOfInterestingRows[[its]]=ir-1
		trueVales=as.numeric(validation_df[ir-1,(INPUT_SIZE+3):(INPUT_SIZE+2+SIZE)]) #seq label,INPUT_SIZE, outputLabel, (OUTPUT_)SIZE, nyb_label, level, (OUTPUT_)SIZE seasonality 
		level=as.numeric(validation_df[ir-1,INPUT_SIZE+4+SIZE])
		seasonality=as.numeric(validation_df[ir-1,(INPUT_SIZE+5+SIZE):(INPUT_SIZE+4+2*SIZE)])
		forecast=as.numeric(forecast_df[ir-1,])
		sMAPE=0;MPE=0; i=1; sav_df=NULL
		for (i in 1:SIZE) {
			forecast[i]=exp(forecast[i]+level+seasonality[i])
			trueVales[i]=exp(trueVales[i]+level+seasonality[i])
	
			pe=(forecast[i]-trueVales[i])/abs(trueVales[i])*100
			ape=abs(forecast[i]-trueVales[i])/(abs(forecast[i])+abs(trueVales[i]))*200
			MPE=MPE+pe
			sMAPE=sMAPE+ape
		}
		sMAPE=sMAPE/SIZE
		MPE=MPE/SIZE
		if (sMAPE>maxSMape) {
			maxSMape=sMAPE
			indexOfMaxSMape=its
		}
		sumSMape=sumSMape+sMAPE
		sumMPE=sumMPE+MPE
	}
	prevSeries=validation_df[ir,1]
}
its=its+1
listOfInterestingRows[[its]]=ir
trueVales=as.numeric(validation_df[ir,(INPUT_SIZE+3):(INPUT_SIZE+2+SIZE)]) #seq label,INPUT_SIZE, outputLabel, (OUTPUT_)SIZE, nyb_label, level, (OUTPUT_)SIZE seasonality 
level=as.numeric(validation_df[ir,INPUT_SIZE+4+SIZE])
seasonality=as.numeric(validation_df[ir,(INPUT_SIZE+5+SIZE):(INPUT_SIZE+4+2*SIZE)])
forecast=as.numeric(forecast_df[ir,])
sMAPE=0;MPE=0; i=1;sav_df=NULL
for (i in 1:SIZE) {
	forecast[i]=exp(forecast[i]+level+seasonality[i])
	trueVales[i]=exp(trueVales[i]+level+seasonality[i])
	
	pe=(forecast[i]-trueVales[i])/abs(trueVales[i])*100
	ape=abs(forecast[i]-trueVales[i])/(abs(forecast[i])+abs(trueVales[i]))*200
	MPE=MPE+pe
	sMAPE=sMAPE+ape
}
if (sMAPE>maxSMape) {
	maxSMape=sMAPE
	indexOfMaxSMape=its
}
sumSMape=sumSMape+sMAPE/SIZE
sumMPE=sumMPE+MPE/SIZE

print(paste("sMAPE=",signif(sumSMape/its,2)," bias=",signif(sumMPE/its,2),sep=''))


#############################################
# just a graph of a randomly select series
series=unique(cif_df$Series)
serId=sample(series,1)
serIdInReducedFile=which(cif_df$Series==serId) 
ir=listOfInterestingRows[[serIdInReducedFile]]
trueVales=as.numeric(validation_df[ir,(INPUT_SIZE+3):(INPUT_SIZE+2+SIZE)]) #seq label,INPUT_SIZE, outputLabel, (OUTPUT_)SIZE, nyb_label, level, (OUTPUT_)SIZE seasonality 
level=as.numeric(validation_df[ir,INPUT_SIZE+4+SIZE])
seasonality=as.numeric(validation_df[ir,(INPUT_SIZE+5+SIZE):(INPUT_SIZE+4+2*SIZE)])
forecast=as.numeric(forecast_df[ir,])
oneLine_df=cif_df[serIdInReducedFile,]
series=as.character(oneLine_df$Series) 
maxForecastHorizon= oneLine_df$maxPredHorizon
y=as.numeric(oneLine_df[4:(ncol(oneLine_df))])
reallyTrueValues=y[!is.na(y)] #from the original data set
 
for (i in 1:SIZE) {
	forecast[i]=exp(forecast[i]+level+seasonality[i])
	trueVales[i]=exp(trueVales[i]+level+seasonality[i]) #from the validation. 
}

ymax=max(trueVales,forecast, reallyTrueValues)
ymin=min(trueVales,forecast, reallyTrueValues)
plot(reallyTrueValues, type='b',ylim=c(ymin,ymax), main=serId, ylab='')
# trueValues should be equal to reallyTrueValues, so the blue color should cover the black at last SIZE points
lines((length(reallyTrueValues)-SIZE+1):length(reallyTrueValues),trueVales, col='blue') 
lines((length(reallyTrueValues)-SIZE+1):length(reallyTrueValues),forecast, col='green')
legend_str_vect=NULL; legend_cols_vect=NULL; legend_char_vect=NULL
legend_str_vect=c(legend_str_vect,"forecast")  
legend_cols_vect=c(legend_cols_vect, 'green')
legend_char_vect=c(legend_char_vect,16)

legend_str_vect=c(legend_str_vect,"true values") 
legend_cols_vect=c(legend_cols_vect, 'blue')
legend_char_vect=c(legend_char_vect,1)

legend("topright", legend_str_vect,
		pch=legend_char_vect, 
		col=legend_cols_vect, cex=1,bty='n')


print(paste("sMAPE=",signif(sumSMape/its,2)," bias=",signif(sumMPE/its,2),sep=''))
