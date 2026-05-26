#################################################################
# Author: Jonas Wolber
# Date: 12.08.22
# Description: This script aims to investigate the generalizability of the learned patterns
#################################################################

#### Load libraries ####
library(xgboost)
library(dplyr)
library(pROC)
library(parallel)
library(caTools)
library(pso)
library(yaml)
# library(ggplot2) # Niels: not needed

#### Load data ####
Paths <- yaml::read_yaml("../Path_config.yaml")

setwd(Paths$PatternChrome_dir)
load(Paths$RNAseq_path)
load(paste0(Paths$main_path,"E003/H3K9me3_50bp_bins.RData"))
load(paste0(Paths$analysis_data,"all_patterns.RData"))

RNA_seq <- RNA_seq[rownames(H3K9me3),]
rm(H3K9me3)
#load("analysis_data/generalizability_matrix.RData") # Niels: I think that is not needed here


#### Determine parameters ####
options(warn=-1)
nrounds <- 50
eta <- 0.2
hp_swarm_size <- 20
hp_maxit_stagnate <- 3
hp_maxit <- 20
exploitation_values <- c(0.8, 0.4)
c_p <- 2.05
c_g <- 2.05
hyperparameter_lower <- c(300, 0.005, 0, 1, 1, 1, 0, 0.1)
hyperparameter_upper <- c(700, 0.2, 10, 10, 5, 10, 10, 0.7)
xgb_early_stopping_rounds <- 10

#### Hyperparametertuning function ####
hyperparameter_tuning <- function(hyperparams){
  round(auc(response = getinfo(xgb_validation, name = "label"), 
    predictor = predict(xgb.train(data = xgb_train, verbose = 0,  nthread = Paths$n_workers, 
    nrounds = hyperparams[1], eta = hyperparams[2], gamma = hyperparams[3], max_depth = round(hyperparams[4]), 
    lambda = hyperparams[5], alpha = hyperparams[6], min_child_weight = hyperparams[7], 
    subsample = hyperparams[8], objective = "binary:logistic", eval.metric = "auc",tree_method = "hist"), 
    xgb_validation), quiet = T),5)
}

#### Create dataframe ####
cell_lines <- list.dirs(full.names = FALSE, path = Paths$dataset_path)
cell_lines <- cell_lines[grep(pattern = "^E\\d{3}$",cell_lines)]
set.seed(42)
date <- "__18_09" # ToDo: adopt with Sys.Date()
generalizability_matrix <- data.frame(Trained = 0, Tested = 0, auc_score = 0)
cl <- makePSOCKcluster(Paths$n_workers)
for (trained_cell_line in cell_lines[41:56]) {
  load(paste(Paths$dataset_path,"/",trained_cell_line,"/",trained_cell_line,date,"_train_df.RData",sep=""))
  #load(paste(Paths$dataset_path,"/",trained_cell_line,"/",trained_cell_line,date, "_patterns.RData",sep=""))
  patterns <- all_patterns[all_patterns$Cell_line == trained_cell_line,-sapply(c("Importance","Correlation"), grep,colnames(all_patterns))]
  
  original_train_df <- train_df
  patterns <- patterns[,-1] %>% mutate_if(is.character,as.numeric) 
  
  #### Train model on chosen cell line ####
  cell_line_model <- xgb.train(data = xgb.DMatrix(data = as.matrix(train_df[,-1]), label = train_df[,1]),  
    nrounds = nrounds, eta = eta, objective = "binary:logistic", verbose = 0, eval.metric = "auc", 
    tree_method = "hist")
  for (tested_cell_line in cell_lines) {
    #### Load data ####
    load(paste(Paths$main_path,tested_cell_line,"/H3K4me3_50bp_bins.RData",sep=""))
    load(paste(Paths$main_path,tested_cell_line,"/H3K4me3_50bp_bins.RData",sep=""))
    load(paste(Paths$main_path,tested_cell_line,"/H3K4me1_50bp_bins.RData",sep=""))
    load(paste(Paths$main_path,tested_cell_line,"/H3K36me3_50bp_bins.RData",sep=""))
    load(paste(Paths$main_path,tested_cell_line,"/H3K27me3_50bp_bins.RData",sep=""))
    load(paste(Paths$main_path,tested_cell_line,"/H3K9me3_50bp_bins.RData",sep=""))
    
    #### Binarize RNA ####
    RNA <- as.numeric(RNA_seq[,tested_cell_line])
    RNA <- (RNA > median(RNA)) + 0
    names(RNA) <- rownames(H3K27me3)
    
    #### Create validation_test_df ####
    if(tested_cell_line != trained_cell_line){
      validation_test_df <- data.frame(GE = RNA[!names(RNA) %in% rownames(train_df)])
      for (pattern in 1:nrow(patterns)) {
        switch (patterns$HM[pattern],
                hm_train <- H3K4me3[,patterns$Start[pattern]:patterns$End[pattern]],
                hm_train <- H3K4me1[,patterns$Start[pattern]:patterns$End[pattern]],
                hm_train <- H3K36me3[,patterns$Start[pattern]:patterns$End[pattern]],
                hm_train <- H3K27me3[,patterns$Start[pattern]:patterns$End[pattern]],
                hm_train <- H3K9me3[,patterns$Start[pattern]:patterns$End[pattern]]
        ) 
        mp <- unlist(patterns[pattern,6:(5+floor(patterns$Width[pattern]))])
        MP_threshold <- patterns$MP_threshold[pattern]
        checked_positions <- 1:(ncol(hm_train)-length(mp))
        genes <- rownames(validation_test_df)
        clusterExport(cl, c("mp", "MP_threshold", "checked_positions", "hm_train", "genes"))
        validation_test_df<-cbind.data.frame(validation_test_df,parSapply(cl, genes,function(g){
          prom <- hm_train[g,]
          sum(sapply(checked_positions, function(pos){
            cor(mp, prom[pos:(pos+length(mp)-1)])>MP_threshold}),na.rm = T)
        }))
      }
      colnames(validation_test_df) <- colnames(train_df)
    }
    else{
      load(paste(Paths$dataset_path,"/",trained_cell_line,"/",trained_cell_line,date, "_validation_test_df.RData",sep=""))
    } 
    
    #### Split into validation and test set ####
    split <- sample.split(validation_test_df$GE, SplitRatio = 0.5)
    validation_df <- validation_test_df[split,]
    test_df <- validation_test_df[!split,]  
    train_df <- original_train_df
    colnames(test_df) <- colnames(train_df)
    colnames(validation_df) <- colnames(train_df)
    
    #### Backward elimination ####
    validation_accuracy <- round(auc(response = validation_df[,1], 
      predictor = predict(xgb.train(data = xgb.DMatrix(data = as.matrix(train_df[,-1]), label = train_df[,1]),
      nrounds = nrounds, eta = eta, objective = "binary:logistic", verbose = 0, nthread = Paths$n_workers,
      eval.metric = "auc", tree_method = "hist"), 
      xgb.DMatrix(data = as.matrix(validation_df[,-1]), label = validation_df[,1])), quiet = T), 5)
    pattern <- ncol(train_df)  
    
    while(pattern != 2){
      be_train_df <- train_df[,-pattern]
      be_validation_df <- validation_df[,-pattern]
      be_test_df <- test_df[,-pattern]
      model <- xgb.train(data = xgb.DMatrix(data = as.matrix(be_train_df[,-1]), label = be_train_df[,1]), 
                         nthread = Paths$n_workers, nrounds = nrounds, eta = eta, objective = "binary:logistic", verbose = 0, eval.metric = "auc")
      xgb_validation <- xgb.DMatrix(data = as.matrix(be_validation_df[,-1]), label = be_validation_df[,1])
      be_accuracy <- round(auc(response = be_validation_df[,1], 
        predictor = predict(model, xgb_validation), quiet = T), 5)
      if(be_accuracy > validation_accuracy){
        train_df <- be_train_df
        validation_df <- be_validation_df
        test_df <- be_test_df
        validation_accuracy <- be_accuracy
        pattern <- ncol(train_df)
      }
      pattern <- pattern - 1
    }  
    xgb_train <- xgb.DMatrix(data = as.matrix(train_df[,-1]), label = train_df[,1])
    xgb_validation <- xgb.DMatrix(data = as.matrix(validation_df[,-1]), label = validation_df[,1])
    res <- psoptim(par = rep(NA, length(hyperparameter_lower)), fn = hyperparameter_tuning, 
      lower = hyperparameter_lower, upper = hyperparameter_upper, 
      control = list(fnscale = -1,vectorize = T, abstol = -1, s = hp_swarm_size, w = exploitation_values, 
      maxit = hp_maxit, c.p = c_p, c.g = c_g, maxit.stagnate = hp_maxit_stagnate))
    
    #### Final test accuracy of k-fold ####
    xgb_test <- xgb.DMatrix(data = as.matrix(test_df[,-1]), label = test_df[,1])
    model <- xgb.train(data = xgb_train, nthread = Paths$n_workers, 
     nrounds = res$par[1], eta = res$par[2], gamma = res$par[3], max_depth = round(res$par[4]), lambda = res$par[5], 
     alpha = res$par[6], min_child_weight = res$par[7], subsample = res$par[8], objective = "binary:logistic", 
     tree_method = "exact", watchlist=list(train = xgb_train, test = xgb_validation), 
     early_stopping_rounds = xgb_early_stopping_rounds, verbose = 0, eval.metric = "auc")
    auc_score <- round(auc(response = test_df[,1], 
      predictor = predict(model, xgb_test), quiet = T), 5)
    generalizability_matrix <-rbind.data.frame(generalizability_matrix, 
        c(trained_cell_line, tested_cell_line, auc_score))  
    print(paste(trained_cell_line, tested_cell_line, auc_score))
  }
}
stopCluster(cl)
generalizability_matrix <- generalizability_matrix[-1,]

save(generalizability_matrix, file = paste0(Paths$analysis_data,"generalizability_matrix.RData"))