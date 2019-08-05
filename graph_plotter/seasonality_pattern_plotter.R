library(ggplot2)
library(reshape2)

file = read.csv(file="./datasets/text_data/M3/m3_other_seasonality.txt", sep=",", header = FALSE)
dataset <-as.data.frame(file)
dataset <- as.data.frame(t(dataset[1:400, 1:50]))
dataset["time"] = seq.int(nrow(dataset))

dataset <- melt(dataset, id.vars = "time")
dataset <- dataset[complete.cases(dataset), ]
colnames(dataset)[3] = "seasonal_values"

png("./graph_plotter/m3_other_seasonality_pattern.png", width = 2000, height = 1000)
ggplot(data=dataset,
       aes(x=time, y=seasonal_values, color=variable)) +
  geom_line() + theme(legend.position = "none", text = element_text(size=50))

dev.off()