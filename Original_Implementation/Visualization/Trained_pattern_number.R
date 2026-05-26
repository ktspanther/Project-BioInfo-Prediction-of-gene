library(ggplot2)
library(yaml)

Paths <- yaml::read_yaml("../Path_config.yaml")
load(paste0(Paths$analysis_data,"Statistics/stats.RData"))

stats$all <- "color"
stats$pattern_number <- as.numeric(stats$`Number of trained patterns`)
stats$auc_score <- as.numeric(stats$`Mean AUC score`)
r_squared <- round(cor(stats$auc_score, stats$pattern_number),3)

if(Paths$Plot){
  # png(paste0(Paths$Figures,"Trained_pattern_number","__1.png"),width = 500,height = 500) # figure __1
  svg(paste0("/sybig/home/npa/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/","Trained_pattern_number","__1.svg"),width = 8,height = 5)
  print({
    ggplot(data = stats, aes(x=pattern_number, y = auc_score, color = all)) +
      geom_point() +
      theme(legend.position = "None") +
      geom_smooth(method=lm) +
      #annotate(geom = "text", label = paste("R squared:" ,r_squared), x = 45, y = 0.9) +
      labs(x="Number of patterns used in XGBoost classifier", y = "AUC score")
  })
  dev.off()
}
