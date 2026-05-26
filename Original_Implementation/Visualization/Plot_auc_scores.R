#######################################
# Author: Jonas Wolber
# Date: 26.08.22
# Description: This script is used to plot the AUC scores of the Pattern Chrome results against those of DeepChrome and ShallowChrome
#######################################
library(ggplot2)
library(forcats)
library(yaml)

#### Load data ####
Paths <- yaml::read_yaml("../Path_config.yaml")

cell_lines <- list.dirs(full.names = FALSE, path = Paths$dataset_path)
cell_lines <- cell_lines[grep(pattern = "^E\\d{3}$",cell_lines)]
load(paste0(Paths$analysis_data,"Statistics/stats.RData"))





#### AUC scores from DeepChrome and ShallowChrome ####
aucs_deepchrome <- c(0.77,0.81,0.8,0.82,0.77,0.76,0.8,0.79,0.79,0.77,0.8,0.81,0.81,0.83,0.83,0.8, 0.8,0.8,0.83,0.89,0.9,0.84,0.9,0.84, 0.8, 0.76, 0.8, 0.76,0.73,0.74,0.78,0.71,0.72,0.74,
                     0.73,0.82,0.72,0.76,0.74,0.9,0.79,0.78,0.76,0.74,0.69,0.73,0.83,0.91,0.92,0.85,0.83,0.84,0.83,0.92,0.83,0.83)
aucs_shallowchrome <- c(0.878,0.879,0.884,0.872,0.887,0.874,0.898,0.891,0.885,0.880,0.868,0.887,0.891,0.891,0.898,0.905, 0.893, 0.888,0.901,0.891,0.882,0.893, 0.876, 0.891, 0.883,0.804,0.850,0.85756,0.81386,0.84087,0.86323,0.85549,0.86363,0.83924,0.82661,
                        0.87473,0.84089,0.84009,0.84250,0.88465,0.84114,0.85575,0.84818,0.83271,0.83361,0.85408,0.89235,0.90511,0.91300,0.90319,0.89433,0.89197,0.88881,0.91957,0.89446,0.89072)
auc_sd_shallowchrome <-c(0.00300,0.00444, 0.00408,0.00383,0.00363,0.00275,0.00245,0.00367,0.00319,0.00384,0.00357,0.00294,0.00236,0.00253,0.00217,0.00257,0.00264,0.00155,0.00387,0.00330,0.00322,0.00285,0.00274,0.00373,0.00187, 0.00337, 0.00386, 0.00287,
                         0.00426, 0.00293, 0.00229, 0.00407, 0.00426, 0.00243, 0.00302, 0.00366, 0.00348, 0.00313, 0.00230, 0.00361, 0.00367, 0.00328, 0.00438, 0.00486, 0.00345, 0.00372, 0.00444, 0.00210, 0.00341, 0.00314, 0.00444, 0.00265, 0.00467, 0.00168, 
                         0.00443, 0.00450)


#### Make a DF ####
AUC_scores <- c(round(as.numeric(stats$`Mean AUC score`),3), aucs_deepchrome, aucs_shallowchrome)
# Errorbar <-  c(round(as.numeric(stats$`AUC score SD`),3), rep(NA,56), auc_sd_shallowchrome)
accuracy_df <- data.frame(
  AUC_scores = AUC_scores, 
  # Errorbar = Errorbar,
  Cell_line = rep(stats$`Cell line`,3),
  type = rep(stats$sample_type,3), # asign the biological context
  Algorithm = c(rep("PatternChrome",56),rep("DeepChrome",56), rep("ShallowChrome",56)))

# reorder ID by PatternChrome performance
accuracy_df$Cell_line <- reorder(accuracy_df$Cell_line,sapply(accuracy_df$Cell_line, function(x){
  return(accuracy_df[accuracy_df$Cell_line == x & accuracy_df$Algorithm == "PatternChrome","AUC_scores"])
}))

# reorder Id factor by the biological context
accuracy_df$Cell_line <- reorder(accuracy_df$Cell_line,sapply(paste0("^",accuracy_df$type,"$"), function(x){
  return(grep(x,c("PT","ESCD","ESC","PC","PCU","CL")))
}))

#### Plot AUC score for each cell line ####
if(Paths$Plot){
  # svg("~/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/AUC_plot_18022024.svg",
  #     width = 14,
  #     height = 7)
  png(paste0(Paths$Figures,"Plot_auc_scores","__1.png"),width = 500,height = 500) # figure __1
  auc_score_plot <- ggplot(accuracy_df, aes(x=Cell_line,y=AUC_scores, fill = Algorithm)) + 
    geom_col(position = "dodge2", width = 0.8) +
    labs(x="Cell line", y = "AUC score") +
    # geom_errorbar(position = "dodge2", width = 0.8, aes(x = Cell_line, ymin = AUC_scores - Errorbar, ymax = AUC_scores + Errorbar)) +
    coord_cartesian(ylim=c(0.6,1.0)) + # set y limits
    theme(legend.position = "bottom") +
    theme(axis.text.x = element_text(size = 10,angle = 90))+
    theme(text = element_text(size=Paths$font_size))
  auc_score_plot
  
  #### Save plot ####
  print({
    auc_score_plot
  })
  dev.off()
}

#### Plot mean AUC score ####
means <- round(c(mean(stats$`Mean AUC score`),mean(aucs_shallowchrome), mean(aucs_deepchrome)),4)
mean_sd_pc <- sd(stats$`Mean AUC score`)
mean_sd_dc <- sd(aucs_deepchrome)
mean_sd_sc <- sd(aucs_shallowchrome)
mean_aucs <- data.frame(Mean = means, SD =  c(mean_sd_pc, mean_sd_sc, mean_sd_dc), Algorithm = c("PatternChrome", "ShallowChrome", "DeepChrome"))

if(Paths$Plot){
  png(paste0(Paths$Figures,"Plot_auc_scores","__2.png"),width = 500,height = 500) # figure __2
  print({
    ggplot(mean_aucs, aes(x=fct_rev(fct_reorder(Algorithm,Mean)),y=Mean,fill=Algorithm)) +
      geom_col(width = 0.5) +
      geom_errorbar(aes(x = Algorithm, ymin = Mean - SD, ymax = Mean + SD), width = 0.4) +
      geom_text(aes(label = means, y = Mean + SD + 0.02), position = position_dodge(0.8), vjust = 0, size = 3.5) +
      theme(legend.position = "none") +
      labs(x="Algorithm", y="Mean AUC score")
  })
  dev.off()
}

#### Statistics ####
stats_pc <- summary(stats$`Mean AUC score`)
auc_comparison_df <- data.frame(PC = stats$`Mean AUC score`, DC = aucs_deepchrome, SC = aucs_shallowchrome)
max_auc_df <- data.frame(PC = 0, DC = 0, SC = 0)
for (row in 1:nrow(auc_comparison_df)) {
  max_auc_df[,which.max(auc_comparison_df[row,])] <- max_auc_df[,which.max(auc_comparison_df[row,])] + 1
}

#### Histogram performance distribution ####
auc_hist_df <- data.frame(AUC_score = stats$`Mean AUC score`, Algorithm = rep("PatternChrome",56))

if(Paths$Plot){
  png(paste0(Paths$Figures,"Plot_auc_scores","__3.png"),width = 500,height = 500) # figure __3
  print({
    ggplot(data = auc_hist_df, aes(x=AUC_score, fill = Algorithm)) +
      geom_histogram(position = "dodge", binwidth = 0.025)
  })
  dev.off()
}


if(Paths$Plot){
  png(paste0(Paths$Figures,"Plot_auc_scores","__4.png"),width = 500,height = 500) # figure __4
  print({
    ggplot(accuracy_df, aes(x=Algorithm, y=AUC_scores)) + 
      geom_violin()
  })
  dev.off()
}

