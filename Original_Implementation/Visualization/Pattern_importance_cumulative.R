#################################
# Date: 20.10.
# Description: Describe the relationship between importance and pattern number
################################

#### Load libraries ####
library(ggplot2)
library(yaml)

#### Load data ####
Paths <- yaml::read_yaml("../Path_config.yaml")
load(paste0(Paths$analysis_data,"all_patterns.RData"))

cell_lines <- unique(all_patterns$Cell_line)

all_importance_pattern_number_df <- data.frame()
for (cell_line in cell_lines) {
  cell_line_patterns <- all_patterns[all_patterns$Cell_line == cell_line,]
  importance_values <- sort(cell_line_patterns$Importance,decreasing = T)
  importance_pattern_number_df <- data.frame(Number = 1:nrow(cell_line_patterns), Importance = importance_values)
  for (row in 1:nrow(importance_pattern_number_df)) {
    importance_pattern_number_df$Importance[row] <- sum(importance_values[1:row])
  }
  all_importance_pattern_number_df <- rbind.data.frame(all_importance_pattern_number_df, importance_pattern_number_df)
}

#### Get mean cumulative importance per pattern number ####
cumulative_importance_mean <- sapply(1:max(all_importance_pattern_number_df$Number), function(i){
  mean(all_importance_pattern_number_df$Importance[all_importance_pattern_number_df$Number==i])
})
cumulative_importance_sd <- sapply(1:max(all_importance_pattern_number_df$Number), function(i){
  sd <- sd(all_importance_pattern_number_df$Importance[all_importance_pattern_number_df$Number==i])
  if(length(sd)==0){0}
  else{sd}
})
group_size <- sapply(1:max(all_importance_pattern_number_df$Number), function(i){
  l <- length(all_importance_pattern_number_df$Importance[all_importance_pattern_number_df$Number==i])
  if(is.na(l)){0}
  else{l}
})

Lower_sd <- cumulative_importance_mean - cumulative_importance_sd / sqrt(group_size)
Higher_sd <- cumulative_importance_mean + cumulative_importance_sd / sqrt(group_size)
max_pattern <- max(all_importance_pattern_number_df$Number)
importance_pattern_number_df <- data.frame(Number = 1:max_pattern,3, 
  Cumulative_importance = cumulative_importance_mean,
  SD = cumulative_importance_sd, Col = "Col")

if(Paths$Plot){
  png(paste0(Paths$Figures,"Pattern_importance_cumulative","__1.png"),width = 500,height = 500) # figure __1
  print({
    ggplot(importance_pattern_number_df, aes(x=Number, y = Cumulative_importance, col = Col)) + 
      geom_line() +
      geom_point(size = 2.5) +
      labs(x="Pattern number", y = "Cumulative importance") +
      geom_errorbar(aes(x = 1:max_pattern, ymin = cumulative_importance_mean - cumulative_importance_sd, ymax = cumulative_importance_mean + cumulative_importance_sd), width = 0.5) +
      theme(legend.position = "None") +
      geom_text(label = group_size, x = 1:max_pattern, y = 1.025) +
      geom_text(label = "N = ", x = 0, y = 1.025)
  })
  dev.off()
}

