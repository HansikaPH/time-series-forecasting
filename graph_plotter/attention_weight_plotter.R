library(ggplot2)
library(reshape2)


seasonalities <- read.csv(file="results/attention_weights/attention_weights_with_stl_decomposition.csv", header=TRUE, sep=",")
melted_data <- melt(seasonalities, id.var='Time_Step')
ggplot(melted_data, aes(x=Time_Step, y=value, col=variable)) + geom_line(color="red") + scale_x_continuous(breaks = round(seq(min(melted_data$Time_Step), max(melted_data$Time_Step), by = 20),1)) + ggtitle("Attention Weights Plot") +
   xlab("Time") + ylab("Weight")
