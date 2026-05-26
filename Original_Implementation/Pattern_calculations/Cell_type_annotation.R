################################################################################
# Author: Niels Paul
# Date: 17.02.2024
# Description: This script adds the sample information to stats file
################################################################################

##### library import ###########################################################
library(yaml)

##### import data ##############################################################
Paths <- yaml::read_yaml("../Path_config.yaml")

stats <- read.csv(file = paste0(Paths$dataset_path,"/stats.csv"))

##### add information ##########################################################
stats$sample_type <- c("PCU","ESCD","ESCD","ESCD","ESCD","ESCD","ESCD","ESCD","ESC","ESC","PC","PCU","PC","PC","PC","PC","PCU",
                          "PCU","PCU","PCU","PCU","PCU","PCU","PCU","PC","PT","PT","PT","PT","PT","PT","PT","PT","PT",
                          "PT","PT","PT","PT","PT","PT","PT","PT","PT","PT","PT","PT","CL","PCU","CL","CL","PCU",
                        "PCU","PCU","PCU","PCU","PCU")
# reformat colnames
stats$X <- NULL
colnames(stats) <- gsub("."," ",colnames(stats),fixed = TRUE)

##### save file ################################################################
save(stats,file = paste0(Paths$analysis_data,"Statistics/stats.RData"))

