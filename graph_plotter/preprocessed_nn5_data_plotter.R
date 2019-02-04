library(ggplot2)
library(reshape2)


data <- read.csv(file="preprocessed_nn5_data_1.csv", header=TRUE, sep=",")
# melted_data <- melt(data, id.var='Time_Step')
ggplot(data, aes(x=Time_Step)) + geom_line(aes(y=Validation), color="red") + geom_line(aes(y=Test), color="blue") +
   xlab("Time") + ylab("Data Value")