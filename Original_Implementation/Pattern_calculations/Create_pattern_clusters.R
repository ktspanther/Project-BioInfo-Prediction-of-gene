#################################################################
# Author: Jonas Wolber
# Date: 12.08.22
# Description: This script aims to determine the clusters of patterns based on the similarity of their shapes using hierarchical clustering
# Steps:
# 1. Determine pattern clusters
#################################################################

######## Preparation ########
# Load libraries
library(cluster)
library(dplyr)
library(yaml)

# Load Directories
Paths <- yaml::read_yaml("../Path_config.yaml")
# Load data and create common dataframe
load(paste0(Paths$analysis_data,"/all_patterns.RData"))

######## 1. Determine pattern clusters ########
for (col in 7:13) {
  all_patterns[,col] <- all_patterns[,col] %>% as.numeric()
}
all_patterns_clustered <- data.frame()
for (hm in 1:5) {
  for (width in 3:7) {
    clustered_patterns <- all_patterns[all_patterns$Width == width & all_patterns$HM == hm,]
    #compute distance matrix
    d <- dist(clustered_patterns[,7:(6+width)], method = "euclidean")
    #gap_stat <- clusGap(all_patterns[,7:(6+width)], FUN = hcut, nstart = 25, K.max = 6, B = 50)
    cluster_number <- 4
    #perform hierarchical clustering using Ward's method
    groups <- cutree(hclust(d, method = "ward.D2" ), k=cluster_number)
    clustered_patterns$Cluster <- groups
    all_patterns_clustered <- rbind.data.frame(all_patterns_clustered, clustered_patterns)
  }
}

all_patterns_clustered <-  mutate(all_patterns_clustered, Combination = 
                      paste("HM", HM,"Width", Width, "Cluster", Cluster))

setwd(Paths$analysis_data)
save(all_patterns_clustered, file = "all_patterns_clustered.RData")


