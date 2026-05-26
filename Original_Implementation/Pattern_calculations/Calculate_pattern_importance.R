######################################################
# Author: Jonas Wolber
# Date: 08.09.22
# Description: The aim of this script is to calculate the pattern importance of all patterns
######################################################

#### Load libraries ####
library(xgboost)
library(parallel)
library(caTools)
library(pROC)
library(pso)
library(dplyr)
library(data.table)
library(yaml)

#### Paths ####
Paths <- yaml::read_yaml("../Path_config.yaml")

# import functions
source(Paths$Custom_functions)

setwd(Paths$PatternChrome_dir)

cell_lines <- list.dirs(full.names = FALSE, path = Paths$dataset_path)
cell_lines <- cell_lines[grep(pattern = "^E\\d{3}$",cell_lines)]

all_patterns <- data.frame()

for (cell_line in cell_lines){
  print(cell_line)
  #### Load data and load model ####
  cell_line_files <- list.files(paste0(Paths$dataset_path,"/",cell_line))
  Train_df_file <- cell_line_files[grep("\\d_train_df.RData",cell_line_files)[1]]
  # load seperated validation and test set
  Validation_df_file <- cell_line_files[grep("validation_df.RData",cell_line_files)[1]]
  Test_df_file <- cell_line_files[grep("\\d_test_df.RData",cell_line_files)[1]]
  Patterns_file <- cell_line_files[grep("_patterns.RData",cell_line_files)[1]]
  XGBmodel_file <- cell_line_files[grep("XGB_model[^_]",cell_line_files)[1]]
  
  xgb_model <- readRDS(file = paste0(Paths$dataset_path,"/",cell_line,"/",XGBmodel_file))
  load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Train_df_file))
  load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Validation_df_file))
  load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Test_df_file))
  load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Patterns_file))
  
  # create xgb matrices from dataframes
  xgb_train <- xgb.DMatrix(data = as.matrix(train_df[,-1]), label = train_df[,1])
  xgb_test <- xgb.DMatrix(data = as.matrix(test_df[,-1]), label = test_df[,1])
  xgb_validation <- xgb.DMatrix(data = as.matrix(validation_df[,-1]), label = validation_df[,1])
  
  #### Feature importance ####
  importance_matrix <- xgb.importance(model = xgb_model)
  patterns$Importance <- unlist(sapply(1:nrow(patterns), function(p){
    importance <- importance_matrix$Gain[importance_matrix$Feature == colnames(train_df)[1 + p]]
    if(length(importance)==0){0}
    else{importance}
  }))
  
  #### Feature correlation ####
  explainer = buildExplainer(xgb_model,xgb_train, type="binary", base_score = 0.5, trees = NULL)
  pred_breakdown = explainPredictions(xgb_model, explainer, xgb_test)
  patterns$Correlation <- unlist(sapply(match(colnames(train_df)[-1], colnames(pred_breakdown)[-length(pred_breakdown)]), 
    function(p){
      correlation <- cor(as.matrix(pred_breakdown)[, p], test_df[,(p + 1)])
      if(length(correlation)==0){0}
      else{correlation}
  }))
  
  #### Add patterns to common dataframe ####
  all_patterns <- rbind.data.frame(all_patterns, patterns)
  
  #### remove old files
  rm(list = c("xgb_model","patterns","test_df","train_df","validation_df",
              "xgb_test","xgb_train","xgb_validation","explainer"))
  gc()
  
}


all_patterns$Correlation <-  unlist(sapply(all_patterns$Correlation, function(p){
  if(is.na(p)){return(0)}
  else{return(p)}
}))


#### Correction ####
for (pattern in 1:nrow(all_patterns)) {
  if(all_patterns$Start[pattern] > all_patterns$End[pattern]){
    
    #### Switch values of start and end position ####
    start <- all_patterns$End[pattern]
    end <- all_patterns$Start[pattern]
    all_patterns$End[pattern] <- end
    all_patterns$Start[pattern] <- start
    
    #### Mirror the pattern ####
    width <- as.numeric(all_patterns$Width[pattern])
    index_Point_1 <- grep("Point_1",colnames(all_patterns))
    index_last_point <- grep(paste0("Point_",width),colnames(all_patterns))
    # unlist(all_patterns[pattern,grep("Point_1",colnames(all_patterns)):grep(paste0("Point_",width),colnames(all_patterns))])
    all_patterns[pattern, index_Point_1:index_last_point] <- all_patterns[pattern, index_last_point:index_Point_1]
  }
}

#### ensure datatype in all_pattern file
for(i  in 3:ncol(all_patterns)){
  all_patterns[,i] <- as.numeric(all_patterns[,i])
}


#### Save pattern data ####
save(all_patterns, file = paste0(Paths$analysis_data,"all_patterns.RData"))



