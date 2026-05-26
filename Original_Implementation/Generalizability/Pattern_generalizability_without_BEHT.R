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
library(ggplot2)
library(yaml)

#### Load data ####
Paths <- yaml::read_yaml("../Path_config.yaml")

setwd(Paths$PatternChrome_dir)
load(Paths$RNAseq_path)
load(paste0(Paths$main_path,"E003/H3K9me3_50bp_bins.RData"))
RNA_seq <- RNA_seq[rownames(H3K9me3),]
rm(H3K9me3)

#load("analysis_data/generalizability_matrix.RData") # Niels: just loads precompiled data

#### Determine parameters ####
rounding_decimal = 5

# options(warn=-1)
# nrounds <- 50
# eta <- 0.2
# 
# hp_swarm_size <- 20
# hp_maxit_stagnate <- 3
# hp_maxit <- 20
# exploitation_values <- c(0.8, 0.4)
# c_p <- 2.05
# c_g <- 2.05
# hyperparameter_lower <- c(300, 0.005, 0, 1, 1, 1, 0, 0.1)
# hyperparameter_upper <- c(700, 0.2, 10, 10, 5, 10, 10, 0.7)
# xgb_early_stopping_rounds <- 10


#### Hyperparametertuning function ####
# hyperparameter_tuning <- function(hyperparams){
#   round(auc(response = getinfo(xgb_validation, name = "label"), 
#     predictor = predict(xgb.train(data = xgb_train, verbose = 0,  nthread = Paths$n_workers, 
#     nrounds = hyperparams[1], eta = hyperparams[2], gamma = hyperparams[3], max_depth = round(hyperparams[4]), 
#     lambda = hyperparams[5], alpha = hyperparams[6], min_child_weight = hyperparams[7], 
#     subsample = hyperparams[8], objective = "binary:logistic", eval.metric = "auc",tree_method = "hist"), 
#     xgb_validation), quiet = T),5)
# }

#### Create dataframe ####
load(paste0(Paths$analysis_data,"all_patterns.RData"))

cell_lines <- list.dirs(full.names = FALSE, path = Paths$dataset_path)
cell_lines <- cell_lines[grep(pattern = "^E\\d{3}$",cell_lines)]
cell_lines <- cell_lines[cell_lines %in% all_patterns$Cell_line]

set.seed(42)
# date <- "__18_09"
generalizability_matrix <- data.frame(Trained = 0, Tested = 0, auc_score = 0)
t1 <- Sys.time() # Niels
cl <- makePSOCKcluster(Paths$n_workers)
for (trained_cell_line in cell_lines) { # Niels [1:10]
  print(paste0("Model from ",trained_cell_line)) 
  #### Load data and load model ####
  cell_line_files <- list.files(paste0(Paths$dataset_path,"/",trained_cell_line))
  Train_df_file <- cell_line_files[grep("\\d_train_df.RData",cell_line_files)[1]]
  Validation_df_file <- cell_line_files[grep("validation_df.RData",cell_line_files)[1]]
  Test_df_file <- cell_line_files[grep("\\d_test_df.RData",cell_line_files)[1]]
  Patterns_file <- cell_line_files[grep("_patterns.RData",cell_line_files)[1]]
  XGBmodel_file <- cell_line_files[grep("XGB_model[^_]",cell_line_files)[1]]
  
  xgb_model <- readRDS(file = paste0(Paths$dataset_path,"/",trained_cell_line,"/",XGBmodel_file))
  load(file = paste0(Paths$dataset_path,"/",trained_cell_line,"/",Train_df_file))
  load(file = paste0(Paths$dataset_path,"/",trained_cell_line,"/",Validation_df_file))
  load(file = paste0(Paths$dataset_path,"/",trained_cell_line,"/",Test_df_file))
  load(file = paste0(Paths$dataset_path,"/",trained_cell_line,"/",Patterns_file))
  
  patterns <- all_patterns[all_patterns$Cell_line == trained_cell_line & all_patterns$Pattern_name %in% colnames(train_df),]
  
  
  # load(paste(Paths$dataset_path,"/",trained_cell_line,"/",trained_cell_line,date,"_train_df.RData",sep=""))
  # ##load(paste(Paths$dataset_path,"/",trained_cell_line,"/",trained_cell_line,date, "_patterns.RData",sep=""))
  # patterns <- all_patterns[all_patterns$Cell_line == trained_cell_line,-sapply(c("Importance","Correlation"), grep,colnames(all_patterns))]
  # 
  # original_train_df <- train_df
  # patterns <- patterns[,-1] %>% mutate_if(is.character,as.numeric) 
  # 
  # #### Train model on chosen cell line ####
  # print(paste0("Train model on ",trained_cell_line)) # Niels
  # cell_line_model <- xgb.train(data = xgb.DMatrix(data = as.matrix(train_df[,-1]), label = train_df[,1]),  
  #   nrounds = nrounds, eta = eta, objective = "binary:logistic", verbose = 0, eval.metric = "auc", 
  #   tree_method = "hist")
  
  
  
  for (tested_cell_line in cell_lines){
    print(paste0("Apply to ",tested_cell_line)) #  Niels
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
    
    #### Create apply_test_df ####
    if(tested_cell_line != trained_cell_line){
      print("Create Test DF") # Niels
      # validation_test_df <- data.frame(GE = RNA[!names(RNA) %in% rownames(train_df)])
      apply_test_df <- data.frame(GE = RNA[names(RNA) %in% rownames(test_df)]) # Niels
      
      for (pattern in 1:nrow(patterns)) {
        switch (patterns$HM[pattern],
                hm_train <- H3K4me3[,patterns$Start[pattern]:patterns$End[pattern]],
                hm_train <- H3K4me1[,patterns$Start[pattern]:patterns$End[pattern]],
                hm_train <- H3K36me3[,patterns$Start[pattern]:patterns$End[pattern]],
                hm_train <- H3K27me3[,patterns$Start[pattern]:patterns$End[pattern]],
                hm_train <- H3K9me3[,patterns$Start[pattern]:patterns$End[pattern]]
        ) 
        mp <- unlist(patterns[pattern,grep("Point_1",colnames(patterns)):grep(paste0("Point_",patterns$Width[pattern]),colnames(patterns))])
        MP_threshold <- patterns$MP_threshold[pattern]
        
        checked_positions <- 1:(ncol(hm_train)-length(mp)+1) # Niels +1
        genes <- rownames(apply_test_df)
        clusterExport(cl, c("mp", "MP_threshold", "checked_positions", "hm_train", "genes"))
        apply_test_df<-cbind.data.frame(apply_test_df,parSapply(cl, genes,function(g){
          prom <- hm_train[g,]
          sum(sapply(checked_positions, function(pos){
            cor(mp, prom[pos:(pos+length(mp)-1)])>MP_threshold}),na.rm = T)
        }))
      }
      colnames(apply_test_df)[-1] <- patterns$Pattern_name
    }
    else{
      print(paste0("Diagonal element ",trained_cell_line))
      apply_test_df <- test_df # Niels
      # load(paste(Paths$dataset_path,"/",trained_cell_line,"/",trained_cell_line,date, "_test_df.RData",sep=""))
    } 
    
    #### Split into validation and test set ####
    # split <- sample.split(apply_test_df$GE, SplitRatio = 0.5)
    # validation_df <- apply_test_df[split,]
    # test_df <- apply_test_df[!split,]  
    # train_df <- original_train_df
    # colnames(test_df) <- colnames(train_df)
    # colnames(validation_df) <- colnames(train_df)
    
    #### Backward elimination ####
    # print("Backward elimination") # Niels
    # validation_accuracy <- round(auc(response = validation_df[,1], 
    #   predictor = predict(xgb.train(data = xgb.DMatrix(data = as.matrix(train_df[,-1]), label = train_df[,1]),
    #   nrounds = nrounds, eta = eta, objective = "binary:logistic", verbose = 0, nthread = Paths$n_workers,
    #   eval.metric = "auc", tree_method = "hist"), 
    #   xgb.DMatrix(data = as.matrix(validation_df[,-1]), label = validation_df[,1])), quiet = T), 5)
    # pattern <- ncol(train_df)  
    # 
    # while(pattern != 2){
    #   be_train_df <- train_df[,-pattern]
    #   be_validation_df <- validation_df[,-pattern]
    #   be_test_df <- test_df[,-pattern]
    #   be_accuracy <- round(auc(response = be_validation_df[,1], 
    #     predictor = predict(xgb.train(data = xgb.DMatrix(data = as.matrix(be_train_df[,-1]), label = be_train_df[,1]), 
    #     nthread = Paths$n_workers, nrounds = nrounds, eta = eta, objective = "binary:logistic", verbose = 0, eval.metric = "auc"), 
    #     xgb.DMatrix(data = as.matrix(be_validation_df[,-1]), label = be_validation_df[,1])), quiet = T), 5)
    #   if(be_accuracy > validation_accuracy){
    #     train_df <- be_train_df
    #     validation_df <- be_validation_df
    #     test_df <- be_test_df
    #     validation_accuracy <- be_accuracy
    #     pattern <- ncol(train_df)
    #   }
    #   pattern <- pattern - 1
    # }  
    # xgb_train <- xgb.DMatrix(data = as.matrix(train_df[,-1]), label = train_df[,1])
    # xgb_validation <- xgb.DMatrix(data = as.matrix(validation_df[,-1]), label = validation_df[,1])
    # res <- psoptim(par = rep(NA, length(hyperparameter_lower)), fn = hyperparameter_tuning,
    #   lower = hyperparameter_lower, upper = hyperparameter_upper,
    #   control = list(fnscale = -1,vectorize = T, abstol = -1, s = hp_swarm_size, w = exploitation_values,
    #   maxit = hp_maxit, c.p = c_p, c.g = c_g, maxit.stagnate = hp_maxit_stagnate))
    
    #### Calculate AUC ####
    # print("Final test accuracy of k-fold") # Niels
    xgb_train <- xgb.DMatrix(data = as.matrix(train_df[,-1]), label = train_df[,1])
    
    # xgb_test <- xgb.DMatrix(data = as.matrix(apply_test_df[,-1]), label = train_df[,1])
    # xgb_validation <- xgb.DMatrix(data = as.matrix(validation_df[,-1]), label = validation_df[,1])
    
    # Todo: check whether this is sufficient
    auc_score <- round(auc(response = apply_test_df[,1],
                               predictor = predict(xgb_model,xgb.DMatrix(data = as.matrix(apply_test_df[,-1]))),
                               quiet = T), rounding_decimal)
    
    
    # auc_score <- round(auc(response = test_df[,1], 
    #   predictor = predict(xgb.train(data = xgb_train, nthread = Paths$n_workers, 
    #   nrounds = res$par[1], eta = res$par[2], gamma = res$par[3], max_depth = round(res$par[4]), lambda = res$par[5], 
    #   alpha = res$par[6], min_child_weight = res$par[7], subsample = res$par[8], objective = "binary:logistic", 
    #   tree_method = "exact",
    #   watchlist=list(train = xgb_train, test = xgb_validation), 
    #   early_stopping_rounds = xgb_early_stopping_rounds, verbose = 0, eval.metric = "auc"), 
    #   xgb.DMatrix(data = as.matrix(test_df[,-1]), label = test_df[,1])), quiet = T), 5)
    generalizability_matrix <-rbind.data.frame(generalizability_matrix, 
        c(trained_cell_line, tested_cell_line, auc_score))  
    print(paste(trained_cell_line, tested_cell_line, auc_score))
  }
  
  #### remove objects train sample
  rm(list = c("xgb_model","patterns","test_df","train_df","validation_df",
              "xgb_test","xgb_train","xgb_validation"))
  gc()
  
}
stopCluster(cl)

generalizability_matrix <- generalizability_matrix[-1,]
generalizability_matrix$auc_score <- as.numeric(generalizability_matrix$auc_score) 
saveRDS(generalizability_matrix,file = paste0(Paths$dataset_path,"/","generalizability_matrix.rds"))


t2 <- Sys.time() # Niels
print(t2 -t1) # Niels

#### Load dataframe ####
#load("~/Master_thesis/generalizability_matrix.RData") # Niels: just loads precompiled data

#### AUC score trained vs untrained ####
# trained_auc_scores <- c()
# untrained_auc_scores <- c()
# setwd("~/Master_thesis/Datasets") # Todo: can we remove that?
# cell_lines <- list.dirs(full.names = FALSE, path = Paths$dataset_path)
# cell_lines <- cell_lines[grep(pattern = "^E\\d{3}$",cell_lines)]
# for (trained_cell_line in cell_lines[1:3]) {
    trained_auc_score <- generalizability_matrix$auc_score[generalizability_matrix$Tested == generalizability_matrix$Trained]
    # trained_auc_scores <- c(trained_auc_scores, trained_auc_score)
    untrained_auc_score <- generalizability_matrix$auc_score[generalizability_matrix$Tested != generalizability_matrix$Trained]
    # untrained_auc_scores <- c(untrained_auc_scores, untrained_auc_score)
# }

mean_trained_auc_scores <- round(mean(as.numeric(trained_auc_score)),4)
sd_trained_auc_scores <- round(sd(as.numeric(trained_auc_score)),4)
mean_untrained_auc_scores <- round(mean(as.numeric(untrained_auc_score)),4)
sd_untrained_auc_scores <- round(sd(as.numeric(untrained_auc_score)),4)
setwd(paste0(Paths$dataset_path,"/"))
auc_score_comp_df <- data.frame(Group = c("Trained", "Untrained"), Mean_auc_score = c(mean_trained_auc_scores, mean_untrained_auc_scores), SD_auc_score = c(sd_trained_auc_scores, sd_untrained_auc_scores))
save(auc_score_comp_df, file = paste0(Paths$dataset_path,"/","pattern_generalizability.RData"))

