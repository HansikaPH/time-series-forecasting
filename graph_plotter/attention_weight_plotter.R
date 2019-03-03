library(ggplot2)
library(reshape2)


seasonalities <- read.csv(file="results/attention_weights/nn5_attention_weights.csv", header=TRUE, sep=",")
melted_data <- melt(seasonalities, id.var='Time_Step')
ggplot(melted_data, aes(x=Time_Step, y=value, col=variable)) + geom_line() + scale_x_continuous(breaks = round(seq(min(melted_data$Time_Step), max(melted_data$Time_Step), by = 40),1)) + scale_fill_continuous(guide = guide_legend()) +
  theme(legend.position="bottom") + ggtitle("Attention Weights Plot") +
   xlab("Time") + ylab("Weight")
