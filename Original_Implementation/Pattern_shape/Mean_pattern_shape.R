#################################################################
# Author: Jonas Wolber
# Date: 12.08.22
# Description: This script aims to investigate the width and the shape of the trained patterns.
#################################################################

#### Load libraries ####
library(ggplot2)
library(ggpubr)
library(cluster)
library(dplyr)
library(factoextra)
library(yaml)

#### Load data and create common dataframe ####
Paths <- yaml::read_yaml("../Path_config.yaml")
load(paste0(Paths$analysis_data,"all_patterns.RData"))

#### Aesthetics ####
point_size <- 2.5
line_size <- 1.2
alpha <- 0.5
error_width <- 1
error_size <- 0.8

#### Mean shape for each width ####
set.seed(42)
hm <- 1
selected_patterns_df <- all_patterns[all_patterns$HM == hm,]
for (col in 2:ncol(selected_patterns_df)) {
  selected_patterns_df[,col] <- selected_patterns_df[,col] %>% as.numeric()
}
Point_means <- data.frame()
Point_sds <- data.frame()
for (width in 3:7) {
  width_point_means <- c()
  width_point_sds <- c()
  for (col in 7:13) {
    selected_points <- selected_patterns_df[selected_patterns_df$Width == width,col]
    mean_selected_points <- mean(selected_points)
    sd_selected_points <- sd(selected_points)
    width_point_means <- c(width_point_means, mean_selected_points)
    width_point_sds <- c(width_point_sds, sd_selected_points)
  }
  Point_means <- rbind.data.frame(Point_means, width_point_means)
  Point_sds <- rbind.data.frame(Point_sds, width_point_sds)
}
Point_means[1,4:7] <- NA
Point_means[2,5:7] <- NA
Point_means[3,6:7] <- NA
Point_means[4,7] <- NA

Point_sds[1,4:7] <- NA
Point_sds[2,5:7] <- NA
Point_sds[3,6:7] <- NA
Point_sds[4,7] <- NA

pattern_shape_df <- data.frame(Position = rep(1:7,5), 
  Point_means = unlist(sapply(1:5, function(w){Point_means[w,]})),
  Point_sds = unlist(sapply(1:5, function(w){Point_sds[w,]})),
  Width = c(sapply(as.character(3:7),function(w){rep(w,7)})))

if(Paths$Plot){
  png(paste0(Paths$Figures,"Mean_pattern_shape","__1.png"),width = 500,height = 500) # figure __1
  print({
    ggplot(data = pattern_shape_df, aes(x = Position, y = Point_means, fill = Width, col = Width)) +
      geom_line(size = 1.4) +
      geom_point(size = 4) +
      geom_errorbar(aes(x = Position, ymin = Point_means - Point_sds, ymax = Point_means + Point_sds), 
                    width = 0.3, size = 1.2) +
      labs(x="Position", y = "Average value") +
      scale_x_continuous(breaks = 1:7)
  })
  dev.off()
}
