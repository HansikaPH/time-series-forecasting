library(ggplot2)
library(reshape2)


seasonalities <- read.csv(file="results/seasonality_strengths/all_seasonality.csv", header=TRUE, sep=",")
meltData <- melt(seasonalities)
# p <- ggplot(meltData, aes(factor(variable), value))
# p + geom_boxplot(outlier.shape=NA) + ggtitle("Box Plots of Seasonality Strengths") +
#   xlab("Dataset Names") + ylab("Seasonality Strengths") + theme(text = element_text(size=20))

p <- ggplot(meltData, aes(factor(variable), value))
p + geom_violin() + geom_boxplot(width=0.1, aes(fill=variable), outlier.shape=NA) +
  # ggtitle("Violin Plots of Seasonality Strengths") +
    xlab("Dataset Names") + ylab("Seasonality Strengths")  + theme(text = element_text(size=20), legend.position = "none")
