library(ggplot2)
library(reshape2)

# data <- read.csv(file="results/plotting_results/ensemble_data/All_MASE_Ranks_Output_Comparison.csv", header=TRUE, sep=",")
# data <- read.csv(file="results/plotting_results/ensemble_data/All_MASE_Ranks_Input_Comparison.csv", header=TRUE, sep=",")
# data <- read.csv(file="results/plotting_results/ensemble_data/All_SMAPE_Ranks_Cell_Comparison.csv", header=TRUE, sep=",")
# data <- read.csv(file="results/plotting_results/ensemble_data/All_SMAPE_Ranks_Optimizer_Comparison.csv", header=TRUE, sep=",")
data <- read.csv(file = "results/plotting_results/ensemble_data/error_comparison_m4_median_smape.csv", header = TRUE, sep = ",")
#data <- read.csv(file="results/plotting_results/ensemble_data/error_comparison_stacked_rank_smape.csv", header=TRUE, sep=",")
# data <- read.csv(file="results/plotting_results/ensemble_data/All_MASE_Ranks_Input_Size_Comparison_With_STL.csv", header=TRUE, sep=",")
# data <- read.csv(file="results/plotting_results/ensemble_data/stl_comparison_tourism_mean_mase.csv", header=TRUE, sep=",")

# ggplot(data=cif, aes(x=x, y=Mean_SMAPE_Rank, color=RNN_Unit)) + xlab("RNN Unit Comparison - CIF Dataset") + ylab("Mean SMAPE Ranks") +  theme(text = element_text(size=20), axis.text.x=element_blank(),
#                                                                                                                                                                     axis.ticks.x=element_blank())

# ggplot(data = kaggle, aes(x=x, y=Mean_SMAPE_Rank, color=Optimizer)) + xlab("Optimizer Comparison - Wikipedia Web Traffic") + ylab("Mean SMAPE Ranks") + geom_text(aes(label=Model_Name_2)) + theme(text = element_text(size=20), axis.text.x=element_blank(),
#                                                                                                                                 axis.ticks.x=element_blank())

#ggplot(data=cif, aes(x=x, y=Mean_SMAPE_Rank, color=RNN_Unit)) + xlab("RNN Unit Comparison") + ylab("Mean SMAPE Ranks") +  theme(text = element_text(size=20), axis.text.x=element_blank(),
#                                                                                                                               axis.ticks.x=element_blank())

# ggplot(data=cif, aes(x=x, y=Mean_SMAPE_Rank, color=RNN_Unit)) + xlab("RNN Unit Comparison") + ylab("Mean SMAPE Ranks") +  theme(text = element_text(size=20), axis.text.x=element_blank(),
#                                                                                                                                 axis.ticks.x=element_blank())

# input size comparison - plots with letters
#ggplot(data=data, aes(x=x, y=Mean_MASE_Rank, color=Input_Window_Size)) +  xlab("Input Window Size Comparison Without STL Decomposition") + ylab("Mean MASE Rank") +  geom_text(aes(label=substring(Dataset, 1, 1)), position=position_jitter(0.2), size=5) + theme(text = element_text(size=20), axis.text.x=element_blank(), axis.ticks.x=element_blank()) + scale_color_manual(values = c("red", "blue"))

# stacked performance - plots with letters
#ggplot(data=data, aes(x=x, y=SMAPE, color=Model_Name)) +  xlab("RNN Architecture Performance Comparison") + ylab("Median MASE Rank") +  geom_text(aes(label=Dataset), position=position_jitter(0.3), size=5) + theme(text = element_text(size=20), axis.text.x=element_blank(), axis.ticks.x=element_blank()) + scale_color_manual(values = c("red", "gold1", "blue3", "magenta"))
#c("red", "springgreen4", "blue3", "gold1", "coral4", "magenta")

#stl comparison 
#ggplot(data=data, aes(x=x, y=SMAPE, color=Model)) +  xlab("M4 Dataset") + ylab("Mean SMAPE") +  geom_point(size=4) + theme(text = element_text(size=20), axis.text.x=element_blank(), axis.ticks.x=element_blank()) + scale_color_manual(values = c("red", "blue3"))

#comparison with ETS
ggplot(data = data, aes(x = x, y = SMAPE, group = Model_Name)) +
    xlab("M4 Dataset") +
    ylab("Median SMAPE") +
    geom_point(aes(color = Model_Name, shape = Model_Name, size= Model_Name)) +
    theme(text = element_text(size = 20), axis.text.x = element_blank(), axis.ticks.x = element_blank()) +
    scale_color_manual(values = c("red", "gold1", "blue3", "darkorchid1", "forestgreen", "gray0")) +
    scale_shape_manual(values = c(15, 16, 17, 18, 8, 12)) +
  scale_size_manual(values=c(4,3,2.5, 4, 5, 5))
#geom_jitter(position=position_jitter(0.1))

#box plots
# p <- ggplot(data, aes(factor(RNN_Unit), Mean_SMAPE_Rank))
# p + geom_boxplot(outlier.shape=NA) +
#   xlab("RNN Unit") + ylab("Mean SMAPE Ranks") + theme(text = element_text(size=20))

# violin plots
# input size comparison
# p <- ggplot(data, aes(factor(Input_Window_Size), Mean_MASE_Rank))
# p + geom_violin() + geom_boxplot(width=0.1, aes(fill=Input_Window_Size)) +
#   xlab("Input Window Size With STL Decomposition") + ylab("Mean MASE Ranks") + theme(text = element_text(size=20), legend.position = "none")

# stacked performance
# p <- ggplot(data, aes(factor(Model_Name), SMAPE))
# p + geom_violin() + geom_boxplot(width=0.1, aes(fill=Model_Name)) +
#    xlab("RNN Architecture Performance Comparison") + ylab("Rank SMAPE") + theme(text = element_text(size=20), legend.position = "none")

#stl comparison
# p <- ggplot(data, aes(factor(Model), MASE))
# p + geom_violin() + geom_boxplot(width=0.1, aes(fill=Model)) +
#   xlab("Tourism Dataset") + ylab("Mean MASE") + theme(text = element_text(size=20), legend.position = "none")

#output component
# p <- ggplot(data, aes(factor(Output_Component), Mean_SMAPE_Rank))
# p + geom_violin() + geom_boxplot(width=0.1, aes(fill=Output_Component)) +
#   xlab("Output Component for Sequence to Sequence Architecture") + ylab("Mean MASE Ranks") + theme(text = element_text(size=20), legend.position = "none")

# cell comparison
# p <- ggplot(data, aes(factor(RNN_Unit), Mean_SMAPE_Rank))
# p + geom_violin() + geom_boxplot(width=0.1, aes(fill=RNN_Unit)) +
#  xlab("RNN Unit") + ylab("Mean SMAPE Ranks") + theme(text = element_text(size=20), legend.position = "none")

# optimizer comparison
# p <- ggplot(data, aes(factor(Optimizer), Mean_SMAPE_Rank))
# p + geom_violin() + geom_boxplot(width=0.1, aes(fill=Optimizer)) +
#  xlab("Optimizer") + ylab("Mean SMAPE Ranks") + theme(text = element_text(size=20), legend.position = "none")

# input format
# p <- ggplot(data, aes(factor(Input_Format), Mean_SMAPE_Rank))
# p + geom_violin() + geom_boxplot(width=0.1, aes(fill=Input_Format)) +
#  xlab("Input Format for S2SD Architecture") + ylab("Mean MASE Ranks") + theme(text = element_text(size=20), legend.position = "none")
