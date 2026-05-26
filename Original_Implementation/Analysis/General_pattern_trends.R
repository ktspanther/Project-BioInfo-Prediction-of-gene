######################################################
# Author: Jonas Wolber
# Date: 08.09.22
# Description: The aim of this script is to show general pattern trends
# Content: 
# 1. Width importance
# 2. Region importance
# 3. Histone modification importance
######################################################

#### Load libraries ####
library(dplyr)
library(ggplot2)
library(yaml)

#### Load pattern data ####
Paths <- yaml::read_yaml("../Path_config.yaml")

load(paste0(Paths$analysis_data,"all_patterns.RData"))

#### Pattern width ####
width_importance <- rep(0,5)
for (r in 1:nrow(all_patterns)) {
  width <- all_patterns$Width[r]
  importance <- all_patterns$Importance[r]
  width_importance[width - 2] <- width_importance[width - 2] + importance
}
pattern_width_df <- data.frame(Width = 3:7, 
    Importance = width_importance)

pattern_width_df$Importance <- pattern_width_df$Importance / sum(pattern_width_df$Importance) * 100
if(Paths$Plot){
  png(paste0(Paths$Figures,"General_pattern_trends","__1.png"),width = 500,height = 500) # figure __1
  print({
    ggplot(data = pattern_width_df, aes(x=Width, y = Importance, fill=as.factor(Width))) +
      geom_col(width = 0.7) +
      theme(legend.position = "None") +
      labs(x="Pattern width", y ="Relative importance (%)") +
      geom_text(label =  round(pattern_width_df$Importance,2), x = 3:7, y = pattern_width_df$Importance + 1.5, size = 3.2)
  })
  dev.off()
}
correlation <- cor(pattern_width_df$Width, pattern_width_df$Importance)

#### Pattern region ####
HMs <- c("H3K4me3", "H3K4me1", "H3K36me3", "H3K27me3", "H3K9me3", "Aggregated")
position_importance_df <- data.frame(Position = rep(1:200,6), Importance = 0, 
  HM = c(sapply(HMs, function(h){rep(h,200)})))
for (p in 1:nrow(all_patterns)) {
  positions <- all_patterns$Start[p]:all_patterns$End[p]
  importance <- unlist(all_patterns$Importance)[p]
  HM <- HMs[all_patterns$HM[p]]
  position_importance_df$Importance[position_importance_df$Position %in% 
    positions & position_importance_df$HM == HM] <- 
    position_importance_df$Importance[position_importance_df$Position%in% positions & 
    position_importance_df$HM == HM] + importance
  position_importance_df$Importance[position_importance_df$Position %in% 
    positions & position_importance_df$HM == "Aggregated"] <- 
    position_importance_df$Importance[position_importance_df$Position %in% 
    positions & position_importance_df$HM == "Aggregated"] + importance
}

position_importance_df$Importance <- position_importance_df$Importance / 
  sum(position_importance_df$Importance[position_importance_df$HM == "Aggregated"])
text_size <- 4

if(Paths$Plot){
  png(paste0(Paths$Figures,"General_pattern_trends","__2.png"),width = 500,height = 500) # figure __2
  # svg("~/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/resultsHM_loc_importance_18022024.svg",
  #     width = 10,
  #     height = 6)
  print({
    ggplot(data = position_importance_df, aes(x = Position - 100, y = Importance, col = HM)) + 
      geom_line(size = 1.1) +
      labs(x = "Position on binned epigenome", y = "Relative importance") +
      annotate(geom = "text", x = 0, y = - 0.0001, label = "TSS", size = text_size) +
      annotate(geom = "text", x = -50, y = - 0.0001, label = "Upstream", size = text_size) +
      annotate(geom = "text", x = 50, y = - 0.0001, label = "Downstream", size = text_size) +
      theme(text = element_text(size=Paths$font_size)) #,axis.text.x=element_text(angle = 45, hjust = 0))
  })
  dev.off()
}

# Relevant stats
# 47 - 91
sum(position_importance_df$Importance[position_importance_df$HM == "Aggregated" & 
  position_importance_df$Position > 46 & position_importance_df$Position < 92])
# upstream
sum(position_importance_df$Importance[position_importance_df$HM == "Aggregated" & 
  position_importance_df$Position < 101])
# downstream
sum(position_importance_df$Importance[position_importance_df$HM == "Aggregated" & 
  position_importance_df$Position > 100])

#### Histone modification importance ####
# upstream
hm_importance_df <- data.frame(HM = HMs[-6], 
  Importance = c(sapply(HMs[-6], function(h){
  sum(position_importance_df$Importance[position_importance_df$HM == h])
  })))

if(Paths$Plot){
  # svg("~/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/HMs_18022024.svg",
  #     width = 6,
  #     height = 6)
  png(paste0(Paths$Figures,"General_pattern_trends","__3.png"),width = 500,height = 500) # figure __3
  print({
    ggplot(data = hm_importance_df, aes(x="", y=Importance, fill=HM)) +
      geom_bar(stat="identity", width=1) +
      coord_polar("y", start=0) +
      geom_text(aes(label = round(Importance, 4)), position = position_stack(vjust=0.5), size = 3.5) +
      labs(x = NULL, y = NULL, fill = NULL)+
      theme(axis.line = element_blank(),
            axis.text = element_blank(),
            axis.ticks = element_blank())
  })
  dev.off()
}

#### Active / repressive tendencies of histone modifications ####
all_patterns$HM <- unlist(sapply(1:nrow(all_patterns), function(hm){HMs[all_patterns$HM[hm]]}))
all_patterns$Correlation_weighted <- all_patterns$Correlation * all_patterns$Importance

if(Paths$Plot){
  png(paste0(Paths$Figures,"General_pattern_trends","__4.png"),width = 500,height = 500) # figure __4
  # svg("~/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/HM_influence_expression_18022024.svg",
  #     width = 8,
  #     height = 6)
  print({
    ggplot(data = all_patterns, aes(x=reorder(factor(HM), -Correlation), y = Correlation, fill = HM)) +
      geom_violin(draw_quantiles = c(0.25, 0.5, 0.75)) +
      labs(x= "",  y = "Mean R squared correlation", fill = "Histone modification") +
      theme(legend.position = "None") +
      annotate(geom = "text", x = 1:5, y = 1.1, 
               label = paste("N =", sapply(HMs[-6],function(hm){length(all_patterns$HM[all_patterns$HM==hm])})))+
      theme(text = element_text(size=Paths$font_size)) #,axis.text.x=element_text(angle = 45, hjust = 0))
  })
  dev.off()
}

Mean_correlation <- unlist(sapply(HMs[-6], function(hm){mean(all_patterns$Correlation[all_patterns$HM==hm])}))
SD_correlation <- unlist(sapply(HMs[-6], function(hm){sd(all_patterns$Correlation[all_patterns$HM==hm])}))

#### Active / repressive regions ####
region_effect_df <- data.frame(Position = rep(1:200,6), Correlation = 0, 
                               HM = c(sapply(HMs, function(h){rep(h,200)})))
for (r in 1:nrow(all_patterns)) {
  Positions <- all_patterns$Start[r]:all_patterns$End[r]
  correlation <- all_patterns$Correlation[r]
  HM <- all_patterns$HM[r]
  region_effect_df$Correlation[region_effect_df$Position %in% Positions & region_effect_df$HM == HM] <-
    region_effect_df$Correlation[region_effect_df$Position %in% Positions & region_effect_df$HM == HM] + correlation
  region_effect_df$Correlation[region_effect_df$Position %in% Positions & region_effect_df$HM == HMs[6]] <-
    region_effect_df$Correlation[region_effect_df$Position %in% Positions & region_effect_df$HM == HMs[6]] + correlation
}
region_effect_df$Correlation <- region_effect_df$Correlation / max(region_effect_df$Correlation)
if(Paths$Plot){
  png(paste0(Paths$Figures,"General_pattern_trends","__5.png"),width = 500,height = 500) # figure __5
  print({
    ggplot(data = region_effect_df, aes(x=Position - 100, y = Correlation, col = HM)) +
      geom_line(size = 1) +
      labs(x="Position on binned epigenome", y = "Mean correlation")
  })
  dev.off()
}





