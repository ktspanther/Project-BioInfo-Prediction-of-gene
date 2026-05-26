#### Load libraries ####
library(xgboost)
library(caTools)
library(pROC)
library(pso)
library(dplyr)
library(data.table)
library(yaml)
library(ggplot2)

#### Load datasets ####
Paths <- yaml::read_yaml("../Path_config.yaml")

# import XGBoost explainer functions
source(Paths$Custom_functions)
cell_line <- "E003"

#### Load data and load model ####
cell_line_files <- list.files(paste0(Paths$dataset_path,"/",cell_line))
Train_df_file <- cell_line_files[grep("\\d_train_df.RData",cell_line_files)[1]]
Validation_df_file <- cell_line_files[grep("validation_df.RData",cell_line_files)[1]]
Test_df_file <- cell_line_files[grep("\\d_test_df.RData",cell_line_files)[1]]
XGBmodel_file <- cell_line_files[grep("XGB_model[^_]",cell_line_files)[1]]

xgb_model <- readRDS(file = paste0(Paths$dataset_path,"/",cell_line,"/",XGBmodel_file))
load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Train_df_file))
load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Validation_df_file))
load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Test_df_file))


Feature_df <- rbind(train_df,validation_df,test_df)
xgb_Feature <- xgb.DMatrix(data = as.matrix(Feature_df[,-1]), label = Feature_df[,1])


explainer = buildExplainer(xgb_model,xgb_Feature, type="binary", base_score = 0.5, trees = NULL)
detach(package:dplyr, unload = TRUE)

#### Oct4 high GE ####
gene_name <- "ENSG00000204531"
idx <- which(rownames(Feature_df) == gene_name)
if(Paths$Plot){
  png(paste0(Paths$Figures,"XGB_explainer","__1.png"),width = 500,height = 500) # figure __1
  # svg("~/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/Waterfall_OCT4_18022024.svg",
  #     width = 14,
  #     height = 8)
  print({
    showWaterfall(xgb_model, explainer, xgb_Feature, Feature_df,  idx, type = "binary")
  })
  dev.off()
}


#### TIE1 low GE ####
gene_name <- "ENSG00000066056"
idx <- which(rownames(Feature_df) == gene_name)
if(Paths$Plot){
  png(paste0(Paths$Figures,"XGB_explainer","__2.png"),width = 500,height = 500) # figure __2
  # svg("~/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/Waterfall_TIE1_18022024.svg",
  #     width = 14,
  #     height = 8)
  print({
    showWaterfall(xgb_model, explainer, xgb_Feature, Feature_df,  idx, type = "binary")
  })
  dev.off()
}

