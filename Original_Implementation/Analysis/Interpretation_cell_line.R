######################################################
# Author: Jonas Wolber
# Date: 08.09.22
# Description: The aim of this script is to interpret how the XGBoost classifier
# comes to its prediction  at a cell line level
# Content: 
# 1. Feature contributions
# 2. Decision boundary
# 3. Contribution-frequency correlation
######################################################

#### Load libraries ####
library(xgboost)
library(ggpubr)
library(parallel)
library(caTools)
library(pROC)
library(pso)
library(dplyr)
library(data.table)
library(ggplot2)
library(yaml)

#### Load data ####
Paths <- yaml::read_yaml("../Path_config.yaml")

# import functions
source(Paths$Custom_functions)
#### import files ####
# load pattern information
load(paste0(Paths$analysis_data,"all_patterns.RData"))

cell_line <- "E003"

cell_line_files <- list.files(paste0(Paths$dataset_path,"/",cell_line))
Train_df_file <- cell_line_files[grep("\\d_train_df.RData",cell_line_files)[1]]
Validation_df_file <- cell_line_files[grep("validation_df.RData",cell_line_files)[1]]
Test_df_file <- cell_line_files[grep("\\d_test_df.RData",cell_line_files)[1]]
# Patterns_file <- cell_line_files[grep("_patterns.RData",cell_line_files)[1]]
XGBmodel_file <- cell_line_files[grep("XGB_model[^_]",cell_line_files)[1]]
Parameter_file <- cell_line_files[grep("^Parameters[^_]",cell_line_files)[1]]
  #/sybig/home/npa/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/datasets/E003/Parameters2024_02_05.rds

res <- readRDS(file = paste0(Paths$dataset_path,"/",cell_line,"/",Parameter_file))
xgb_model <- readRDS(file = paste0(Paths$dataset_path,"/",cell_line,"/",XGBmodel_file))
load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Train_df_file))
load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Validation_df_file))
load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Test_df_file))
# load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Patterns_file))



# setwd(paste(Paths$dataset_path,"/", cell_line, sep =""))
# load(paste(cell_line, date, "_train_df.RData", sep =""))
# load(paste(cell_line, date, "_validation_test_df.RData", sep =""))
# load(paste0(Paths$analysis_data,"all_patterns.RData"))
# #load(paste(cell_line, date, "_patterns.RData", sep =""))
# patterns <- all_patterns[all_patterns$Cell_line == cell_line,-sapply(c("Importance","Correlation"), grep,colnames(all_patterns))]

# #### Parameters ####
# eta <- 0.2
# nrounds <- 50
# set.seed(42)
# options(warn=-1)
xgb_early_stopping_rounds <- 10
# nrounds <- 50
# eta <- 0.2
# rounding_decimal <- 5
# hyperparameter_lower <- c(300, 0.005, 0, 1, 1, 1, 0, 0.1)
# hyperparameter_upper <- c(700, 0.2, 10, 10, 5, 10, 10, 0.7)
# hp_swarm_size <- 20
# hp_maxit_stagnate <- 3
# hp_maxit <- 20
# report <- T
# report_frequency <- 5
# exploitation_values <- c(0.8, 0.4)
# c_p <- 2.05
# c_g <- 2.05
# cell_line <- "E003"
# date <- "__18_09" # ToDo: Maybe Adjust Date to Sys.Date()


#### Backward elimination ####
# split <- sample.split(validation_test_df$GE, SplitRatio = 0.5)
# validation_df <- validation_test_df[split,]
# test_df <- validation_test_df[!split,]
# validation_accuracy <- round(auc(response = validation_df[,1], 
#   predictor = predict(xgb.train(data = xgb.DMatrix(data = as.matrix(train_df[,-1]), label = train_df[,1]),
#   nrounds = nrounds, eta = eta, objective = "binary:logistic", verbose = 0, eval.metric = "auc", 
#   tree_method = "hist"), xgb.DMatrix(data = as.matrix(validation_df[,-1]), label = validation_df[,1])), 
#   quiet = T),5)
# pattern <- ncol(train_df)
# while(pattern != 2){
#   be_train_df <- train_df[,-pattern]
#   be_validation_df <- validation_df[,-pattern]
#   be_test_df <- test_df[,-pattern]
#   be_accuracy <- round(auc(response = be_validation_df[,1], 
#     predictor = predict(xgb.train(data = xgb.DMatrix(data = as.matrix(be_train_df[,-1]), 
#     label = be_train_df[,1]), nrounds = nrounds, eta = eta, objective = "binary:logistic", verbose = 0, 
#     eval.metric = "auc",n_jobs = Paths$n_workers), xgb.DMatrix(data = as.matrix(be_validation_df[,-1]),
#     label = be_validation_df[,1])), quiet = T), 5)
#   if(be_accuracy > validation_accuracy){
#     train_df <- be_train_df
#     validation_df <- be_validation_df
#     test_df <- be_test_df
#     validation_accuracy <- be_accuracy
#     pattern <- ncol(train_df)
#     patterns <- patterns[-pattern,]
#   }
#   pattern <- pattern - 1
# }
# xgb_train <- xgb.DMatrix(data = as.matrix(train_df[,-1]), label = train_df[,1])
# xgb_test <- xgb.DMatrix(data = as.matrix(test_df[,-1]), label = test_df[,1])
# xgb_validation <- xgb.DMatrix(data = as.matrix(validation_df[,-1]), label = validation_df[,1])
# model <- xgb.train(data=xgb_train, nrounds = nrounds, eta = eta, 
#   watchlist=list(train = xgb_train, test = xgb_validation), 
#   objective = "binary:logistic", eval.metric = "auc", early_stopping_rounds = 5, verbose = 0,n_jobs = Paths$n_workers)
# 
# 
# #### Hyperparametertuning ####
# hyperparameter_tuning <- function(hyperparams){
#   round(auc(response = getinfo(xgb_validation, name = "label"), 
#     predictor = predict(xgb.train(data = xgb_train, verbose = 0,  nthread = Paths$n_workers, 
#     nrounds = hyperparams[1], eta = hyperparams[2], gamma = hyperparams[3], max_depth = round(hyperparams[4]), 
#     lambda = hyperparams[5], alpha = hyperparams[6], min_child_weight = hyperparams[7], 
#     subsample = hyperparams[8], objective = "binary:logistic", eval.metric = "auc",tree_method = "hist"), 
#     xgb_validation), quiet = T),rounding_decimal)
# }
# res <- psoptim(par = rep(NA, length(hyperparameter_lower)), fn = hyperparameter_tuning, 
#   lower = hyperparameter_lower, upper = hyperparameter_upper, 
#   control = list(fnscale = -1,vectorize = T, abstol = -1, 
#   s = hp_swarm_size, w = exploitation_values, maxit = hp_maxit, c.p = c_p, c.g = c_g, 
#   maxit.stagnate = hp_maxit_stagnate))
# 
# #### Create XGBoost training model ####
# xgb_model <- xgb.train(data = xgb_train,
#   nrounds = res$par[1], eta = res$par[2], gamma = res$par[3], max_depth = round(res$par[4]), 
#   lambda = res$par[5], alpha = res$par[6], min_child_weight = res$par[7], subsample = res$par[8], 
#   objective = "binary:logistic", tree_method = "exact",
#   watchlist=list(train = xgb_train, test = xgb_validation), 
#   early_stopping_rounds = xgb_early_stopping_rounds, verbose = 0, eval.metric = "auc",n_jobs = Paths$n_workers)

#### Feature importance ####
importance_matrix <- xgb.importance(model = xgb_model)
most_important_features <- importance_matrix$Feature[which(importance_matrix$Gain %in% 
  sort(importance_matrix$Gain, decreasing = T)[1:2])]
most_important_feature_values <- sort(importance_matrix$Gain, decreasing = T)[1:2]

#### add Histone modification info ####
pattern_info <- all_patterns[all_patterns$Cell_line == cell_line,]

importance_matrix$Histone_modification <- c("H3K4me3","H3K4me1","H3K36me3","H3K27me3","H3K9me3")[pattern_info$HM[sapply(paste0("^",importance_matrix$Feature,"$"),grep,pattern_info$Pattern_name)]]





#### Make feature contribution plot ####
P1 <- gsub('_', ' ', most_important_features[1])
P2 <- gsub('_', ' ', most_important_features[2])
importance_matrix$Feature <- gsub("_"," ", importance_matrix$Feature)
if(Paths$Plot){
  png(paste0(Paths$Figures,"Interpretation_cell_line","__1.png"),width = 500,height = 500) # figure __1
  # svg(paste0("/sybig/home/npa/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/","Interpretation_cell_line","__1.svg"),width = 8,height = 5)
  print({
    ggplot(data = importance_matrix, aes(x=reorder(Feature, -Gain),y=Gain, fill = Histone_modification)) +
      theme( axis.text.x = element_text(color = "black", angle=90)) +
      geom_col() +
      labs(x="Patterns", y = "Importance") 
  })
  dev.off()
}

#### Filter columns from datasets ####
twoD_train_df <-cbind.data.frame(train_df[,1],train_df[,c(most_important_features)])  
twoD_validation_df <- cbind.data.frame(validation_df[,1],validation_df[,c(most_important_features)])  
twoD_test_df <- cbind.data.frame(test_df[,1],test_df[,c(most_important_features)])  
colnames(twoD_test_df) <- c("GE", "p1", "p2")

#### Create XGBoost model with only two most important features ####
xgb_train <- xgb.DMatrix(data = as.matrix(twoD_train_df[,c(2,3)]), label = train_df[,1])
xgb_validation <- xgb.DMatrix(data = as.matrix(twoD_validation_df[,c(2,3)]), label = validation_df[,1])
twoD_model <- xgb.train(data = xgb_train,
  nrounds = res$par[1], eta = res$par[2], gamma = res$par[3], max_depth = round(res$par[4]), 
  lambda = res$par[5], alpha = res$par[6], min_child_weight = res$par[7], subsample = res$par[8], 
  objective = "binary:logistic", tree_method = "exact",
  watchlist=list(train = xgb_train, test = xgb_validation), 
  early_stopping_rounds = xgb_early_stopping_rounds, verbose = 0, eval.metric = "auc",n_jobs = Paths$n_workers)

#### Plot with observed test values ####
resolution <- 100
r <- sapply(twoD_train_df[,-1], range)
xs <- seq(r[1,1], r[2,1], length.out = resolution)  
ys <- seq(r[1,2], r[2,2], length.out = resolution)    
g <- cbind.data.frame(rep(xs, each=resolution), rep(ys, time = resolution))
colnames(g) <- colnames(r)
p <- predict(twoD_model, as.matrix(g),type = "prob")
q <- predict(twoD_model, as.matrix(g),type = "class")
p <- as.data.frame(p) %>% mutate(p=ifelse(p<=0.5,0,1))
p <- p %>% mutate(pred = q)
colnames(g) <- c("Pattern_1", "Pattern_2")

#### Visualize decision boundary ####
if(Paths$Plot){
  png(paste0(Paths$Figures,"Interpretation_cell_line","__2.png"),width = 500,height = 500) # figure __2
  # png(paste0("/sybig/home/npa/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/","Interpretation_cell_line","__2.png"),width = 750,height = 750)
  print({
    ggplot() +
      geom_raster(data= g, aes(x= Pattern_1, y=Pattern_2, fill=as.factor(p$p)), alpha = 0.2) +
      geom_point(data = twoD_test_df, aes(x=p1, y=p2, col = as.factor(GE))) +
      labs(x = P1, y = P2, fill = "Predicted gene expression", col = "Observations") + 
      theme(legend.position = "bottom")
  })
  dev.off()
}

xgb_train <- xgb.DMatrix(data = as.matrix(train_df[,-1]), label = train_df[,1])
xgb_test <- xgb.DMatrix(data = as.matrix(test_df[,-1]), label = test_df[,1])
xgb_validation <- xgb.DMatrix(data = as.matrix(validation_df[,-1]), label = validation_df[,1])
explainer = buildExplainer(xgb_model,xgb_train, type="binary", base_score = 0.5, trees = NULL)
pred_breakdown = explainPredictions(xgb_model, explainer, xgb_test)

pred_breakdown <- as.matrix(pred_breakdown)
importance_matrix <- xgb.importance(model = xgb_model)
pattern_correlations <- sapply(importance_matrix$Feature[importance_matrix$Feature %in% colnames(test_df)], function(p){
  cor(pred_breakdown[, p], test_df[,p])
})
pattern_correlations_df <- data.frame(Pattern = importance_matrix$Feature,
  Correlation = pattern_correlations, 
  Importance = importance_matrix$Gain)

# add histone modification information
pattern_correlations_df$Histone_modification <- c("H3K4me3","H3K4me1","H3K36me3","H3K27me3","H3K9me3")[pattern_info$HM[sapply(paste0("^",pattern_correlations_df$Pattern,"$"),grep,pattern_info$Pattern_name)]]

pattern_correlations_df$Pattern <- gsub("_"," ",pattern_correlations_df$Pattern)

if(Paths$Plot){
  png(paste0(Paths$Figures,"Interpretation_cell_line","__3.png"),width = 500,height = 500) # figure __3
  # svg(paste0("/sybig/home/npa/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/",
  #            "Interpretation_cell_line","__3.svg"),width = 8,height = 5)
  print({
    ggplot(data = pattern_correlations_df) +
      geom_col(aes(x = reorder(Pattern, -Importance), y = Correlation, fill = Histone_modification), width = 0.8) +
      theme() +
      labs(x = "", y = "Correlation") +
      theme(axis.text.x=element_text(color = "black", angle=90))
  })
  dev.off()
}

contributions <- as.matrix(pred_breakdown)

most_important_features <- importance_matrix$Feature[which(importance_matrix$Gain %in% 
                                                             sort(importance_matrix$Gain, decreasing = T)[1:10])]

#### Plot pattern frequency vs prediction contribution for two most important patterns ####
Pattern <- 1
pattern_df <- data.frame(Freq = test_df[,Pattern+1], Contribution = pred_breakdown[,Pattern], Col = cell_line)
pattern_1_plot <- ggplot(pattern_df, aes(x=Freq,y=Contribution, col = Col)) +
  geom_point() +
  geom_smooth(method = "gam") +
  theme(legend.position = "None") +
  labs(x="Pattern frequency", y = "Prediction contribution") +
  ggtitle("Pattern 1")

Pattern <- 7
pattern_df <- data.frame(Freq = test_df[,Pattern+1], Contribution = pred_breakdown[,Pattern], Col = cell_line)
pattern_7_plot <- ggplot(pattern_df, aes(x=Freq,y=Contribution, col = Col)) +
  geom_point() +
  geom_smooth(method = "gam") +
  labs(x="Pattern frequency", y = "Prediction contribution") +
  theme(legend.position = "None") +
  ggtitle("Pattern 7")


# Pattern <- 16
# pattern_df <- data.frame(Freq = test_df[,Pattern+1], Contribution = pred_breakdown[,Pattern], Col = cell_line)
# pattern_16_plot <- ggplot(pattern_df, aes(x=Freq,y=Contribution, col = Col)) +
#   geom_point() +
#   geom_smooth(method = "gam") +
#   theme(legend.position = "None") +
#   labs(x="Pattern frequency", y = "Prediction contribution") +
#   ggtitle("Pattern 16")
# 
# Pattern <- 5
# pattern_df <- data.frame(Freq = test_df[,Pattern+1], Contribution = pred_breakdown[,Pattern], Col = cell_line)
# pattern_5_plot <- ggplot(pattern_df, aes(x=Freq,y=Contribution, col = Col)) +
#   geom_point() +
#   geom_smooth(method = "gam") +
#   theme(legend.position = "None") +
#   labs(x="Pattern frequency", y = "Prediction contribution") +
#   ggtitle("Pattern 5")

if(Paths$Plot){
  png(paste0(Paths$Figures,"Interpretation_cell_line","__4.png"),width = 500,height = 500) # figure __4
  # svg(paste0("/sybig/home/npa/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/","Interpretation_cell_line","__4.svg"),width = 8,height = 11) #
  print({
    ggarrange(pattern_1_plot, pattern_7_plot, nrow = 2)
  })
  dev.off()
}


Pattern <- 1
pattern_df <- data.frame(Freq = test_df[,Pattern+1], Contribution = pred_breakdown[,Pattern], Col = cell_line)
if(Paths$Plot){
  png(paste0(Paths$Figures,"Interpretation_cell_line","__5.png"),width = 500,height = 500) # figure __5
  print({
    ggplot(pattern_df, aes(x=Freq,y=Contribution, col = Col)) +
      geom_point() +
      geom_smooth(method = "gam") +
      theme(legend.position = "None") +
      labs(x="Pattern frequency", y = "Prediction contribution") +
      ggtitle("Pattern 1")
  })
  dev.off()
}

#### Plot fitted line for 10 most important patterns ####
curves_df <- data.frame()
for (p in most_important_features) {
  freq_contribution_df <- data.frame(Freq = test_df[,p], contributions[,p])
  colnames(freq_contribution_df)[2] <- "Contribution"
  plot <- ggplot(data = freq_contribution_df, aes(x=Freq, y = Contribution)) +
    geom_point() +
    geom_smooth(method = "gam")
  ggp_data <- ggplot_build(plot)$data[[2]]
  x <- ggp_data$x
  y <- ggp_data$y
  pat <- as.factor(p)
  curves_df <- rbind.data.frame(curves_df, data.frame(x=x,y=y,pat = pat))
}
curves_df$Histone_modification <- curves_df$pat
curves_df$Histone_modification <- c("H3K4me3","H3K4me1","H3K36me3","H3K27me3","H3K9me3")[pattern_info$HM[sapply(paste0("^",curves_df$Histone_modification,"$"),grep,pattern_info$Pattern_name)]]
curves_df$pat <- gsub("Pattern_", "", curves_df$pat)
if(Paths$Plot){
  png(paste0(Paths$Figures,"Interpretation_cell_line","__6.png"),width = 500,height = 500) # figure __6
  # svg(paste0("/sybig/home/npa/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/","Interpretation_cell_line","__6.svg"),width = 8,height = 5)
  print({
    ggplot(data = curves_df, aes(x=x,y=y,col=pat)) + 
      geom_line(size = 0.8) +
      labs(x="Pattern frequency", "Prediction contribution", col = "Pattern")
})
  dev.off()
}

