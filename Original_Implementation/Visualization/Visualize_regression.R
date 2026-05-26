################################################################################
#
# Script to calculate and visualize the correlation between predicted and
# observed Gene expression data
# Niels Paul 11.03.2024
#
################################################################################

#### Library import ############################################################
library(ggplot2)
library(yaml)
library(xgboost)
library(latex2exp)

#### Import data ###############################################################
cell_line <- "E003"

Paths <- yaml::read_yaml("../Path_config.yaml")

cell_line_files <- list.files(paste0(Paths$dataset_path,"/",cell_line))

Test_df_file <- cell_line_files[grep("\\d_test_regression_df.RData",cell_line_files)[1]]
XGBmodel_file <- cell_line_files[grep("^XGB_model_regression",cell_line_files)[1]]

xgb_model <- readRDS(file = paste0(Paths$dataset_path,"/",cell_line,"/",XGBmodel_file))
load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Test_df_file))

#### main ######################################################################

#### Calculate predicted_values ####
GE_predicted <- predict(xgb_model,xgb.DMatrix(data = as.matrix(test_df[,-1]), label = test_df[,1]))

#### calculate correlation of predicted and actual Geneexpression values ####
regression_df <- data.frame(Predicted = GE_predicted, Observed = test_df[,1])

correlation <- round(cor(regression_df$Observed, regression_df$Predicted,method = "spearman"),4)

if(Paths$Plot){
  png(paste0(Paths$Figures,"Visualize_regression","__1.png"),width = 500,height = 500) # figure __1
  # svg(paste0("/sybig/home/npa/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/",
  #           "Visualize_regression","_1.svg"),width = 8,height = 8)
  print({
    ggplot(regression_df, aes(x=Predicted, y = Observed)) +
      geom_point(alpha=0.5) +
      theme(legend.position = "None") +
      labs(x="Predicted gene expression", y="Observed gene expression") +
      annotate(geom = "text", label = TeX(paste0("$\\rho^{2}$ = ",correlation),output = "expression"),x=-3.4, y = 8)
  })
  dev.off()
}

