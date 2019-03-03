library(ggplot2)
library(reshape2)

cif <- read.csv(file="graph_plotter/data/CIF_Mean_SMAPE_Ranks_Output.csv", header=TRUE, sep=",")
kaggle <- read.csv(file="graph_plotter/data/Kaggle_Mean_SMAPE_Ranks_Output.csv", header=TRUE, sep=",")
m3 <- read.csv(file="graph_plotter/data/M3_Mean_SMAPE_Ranks_Output.csv", header=TRUE, sep=",")
nn5 <- read.csv(file="graph_plotter/data/NN5_Mean_SMAPE_Ranks_Output.csv", header=TRUE, sep=",")

melted_cif_data <- melt(cif, id.var='Model_Name')
melted_kaggle_data <- melt(kaggle, id.var='Model_Name')
melted_m3_data <- melt(m3, id.var='Model_Name')
melted_nn5_data <- melt(nn5, id.var='Model_Name')

# ggplot() +
#   geom_text(data=melted_cif_data, aes(x=Model_Name, y=value, col=variable, label=substring(variable, 1, 1)), color='springgreen4') +
#   geom_text(data=melted_kaggle_data, aes(x=Model_Name, y=value, col=variable, label=substring(variable, 1, 1)), color='blue') +
#   geom_text(data=melted_m3_data, aes(x=Model_Name, y=value, col=variable, label=substring(variable, 1, 1)), color="red") +
#   geom_text(data=melted_nn5_data, aes(x=Model_Name, y=value, col=variable, label=substring(variable, 1, 1)), color="orange1") +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1)) + ggtitle("Mean SMAPE Ranks") +
#    xlab("Model") + ylab("Rank")
#
# ggplot() +
#   geom_text(data=melted_cif_data, aes(x=Model_Name, y=value, col=variable, label=variable), color='springgreen4') +
#   geom_text(data=melted_kaggle_data, aes(x=Model_Name, y=value, col=variable, label=variable), color='blue') +
#   geom_text(data=melted_m3_data, aes(x=Model_Name, y=value, col=variable, label=variable), color="red") +
#   geom_text(data=melted_nn5_data, aes(x=Model_Name, y=value, col=variable, label=variable), color="orange1") +
#   theme(axis.text.x = element_text(angle = 90, hjust = 1)) + ggtitle("Mean SMAPE Ranks") +
#    xlab("Model") + ylab("Rank")

# ggplot(data=melted_kaggle_data, aes(x=Model_Name, y=value, color=variable)) + geom_jitter(position=position_jitter(0)) + xlab("Wikipedia Web Traffic") + ylab("SMAPE Rank") +  theme(axis.text.x=element_blank(),
#                                                                                                                                                                      axis.ticks.x=element_blank())
ggplot() +
  # geom_text(data=melted_cif_data, aes(x=Model_Name, y=value, color=variable, label=x)) +
  # geom_text(data=melted_kaggle_data, aes(x=Model_Name, y=value, color=variable, label="k"))+
  geom_text(data=nn5, aes(x=Model_Name, y=Value, color=Output, label=x))+
  # geom_text(data=melted_m3_data, aes(x=Model_Name, y=value, color=variable, label="M3"))+
  geom_jitter(position=position_jitter(0.2))
