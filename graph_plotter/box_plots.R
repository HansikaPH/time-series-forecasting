library(ggplot2)
library(reshape2)


seasonalities <- read.csv(file="results/seasonality_strengths/all_seasonality.csv", header=TRUE, sep=",")
meltData <- melt(seasonalities)
p <- ggplot(meltData, aes(factor(variable), value))
p + geom_boxplot(outlier.shape=NA) + ggtitle("Box Plots of Seasonality Strengths") +
  xlab("Dataset Names") + ylab("Seasnality Strengths")
