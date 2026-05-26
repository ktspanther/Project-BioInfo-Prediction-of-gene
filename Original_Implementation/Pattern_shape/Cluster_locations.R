###########################################
# Description: Determine group size and aggregated importance per cluster and 
# find out the locations where the most important clusters are present
##########################################

#### Load libraries ####
library(ggplot2)
library(yaml)

#### Load data ####
Paths <- yaml::read_yaml("../Path_config.yaml")
load(paste0(Paths$analysis_data,"all_patterns_clustered.RData"))

#### Find out group size and aggregated importance ####
all_combis <- unique(all_patterns_clustered$Combination)
cluster_size <- sapply(all_combis, function(c){length(all_patterns_clustered$Importance[
  all_patterns_clustered$Combination==c])})
aggregated_importance <- sapply(all_combis, function(c){sum(all_patterns_clustered$Importance[
  all_patterns_clustered$Combination==c])/56})
sum(aggregated_importance)
cluster_df <- data.frame(Cluster=all_combis,Size=cluster_size,Importance=aggregated_importance)
sum(cluster_df$Importance)

#### Determine importance and location of five most important clusters ####
location_df <- data.frame(Position = rep(1:200, length(all_combis)), Importance = 0, 
                          Combination = c(sapply(all_combis, function(c){rep(c,200)})))
for (p in 1:nrow(all_patterns_clustered)) {
  positions <- all_patterns_clustered$Start[p]:all_patterns_clustered$End[p]
  importance <- as.numeric(all_patterns_clustered$Importance[p])
  if(is.na(importance)){importance <- 0}
  combination <- all_patterns_clustered$Combination[p]
  location_df$Importance[location_df$Position %in% positions & location_df$Combination == combination] <-
    location_df$Importance[location_df$Position %in% positions & location_df$Combination == combination] + importance
}
location_df$Importance <- location_df$Importance/sum(location_df$Importance)

#### Filter five most important combis ####
top_combis <- sort(aggregated_importance,decreasing = T)[1:5]
location_df <- location_df[location_df$Combination %in% names(top_combis),]

#### Plot figure ####
location_df$Position <- location_df$Position - 100
y_lab_position <- -0.00025

if(Paths$Plot){
  png(paste0(Paths$Figures,"Cluster_locations","__1.png"),width = 500,height = 500) # figure __1
  print({
    ggplot(data = location_df, aes(x = Position, y = Importance, col = Combination)) +
      geom_line(size = 1.1) +
      labs(x="Position on binned epigenome", y = "Relative importance") +
      annotate(geom = "text", label = "TSS", x = 0, y = y_lab_position) +
      annotate(geom = "text", label = "Upstream", x = -50, y = y_lab_position) +
      annotate(geom = "text", label = "Downstream", x = 50, y = y_lab_position)
  })
  dev.off()
}

#### Plot shape of five most important clusters ####
top_five_clusters <- names(top_combis)[1:5] # Niels top_five clusters was not defined so far

shape_df <- data.frame(Combi = top_five_clusters, Point_1_mean = 0, Point_2_mean = 0, Point_3_mean = 0, Point_4_mean = 0, Point_5_mean = 0, Point_6_mean = 0, Point_7_mean = 0)
for (combi in 1:length(top_five_clusters)) {
  selected_cluster <- all_patterns_clustered[all_patterns_clustered$Combination == top_five_clusters[combi],]
  width <- selected_cluster$Width[combi]
  means <- c()
  for (point in 1:width) {
    means <- c(means, mean(selected_cluster[,(6+point)]))
  }
  shape_df[combi,2:(1+width)] <- means
}
sd_df <- data.frame(Combi = top_five_clusters, Point_1_sd = 0, Point_2_sd = 0, Point_3_sd = 0, Point_4_sd = 0, Point_5_sd = 0, Point_6_sd = 0, Point_7_sd = 0)
for (combi in 1:length(top_five_clusters)) {
  selected_cluster <- all_patterns_clustered[all_patterns_clustered$Combination == top_five_clusters[combi],]
  width <- selected_cluster$Width[combi]
  sds <- c()
  for (point in 1:width) {
   sds <- c(sds, sd(selected_cluster[,(6+point)]))
  }
  shape_df[combi,2:(1+width)] <- sds
}


