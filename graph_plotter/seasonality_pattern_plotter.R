library(ggplot2)
library(reshape2)

file = read.csv(file="/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Codes/time-series-forecasting/datasets/text_data/kaggle_web_traffic/kaggle_seasonality.txt", sep=",", header = FALSE)
dataset <-as.data.frame(file)
dataset <- as.data.frame(t(dataset[1:400, 1:50]))
dataset["time"] = seq.int(nrow(dataset))

dataset <- melt(dataset, id.vars = "time")
dataset <- dataset[complete.cases(dataset), ]
colnames(dataset)[3] = "seasonal_values"

png("/media/hhew0002/f0df6edb-45fe-4416-8076-34757a0abceb/hhew0002/Academic/Monash University/Research Project/Written Papers/review_paper_content/review_paper_images/kaggle_seasonality_pattern.png", width = 2000, height = 1000)
ggplot(data=dataset,
       aes(x=time, y=seasonal_values, color=variable)) +
  geom_line() + theme(legend.position = "none", text = element_text(size=50))

dev.off()