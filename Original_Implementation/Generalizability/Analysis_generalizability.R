###################################
# Description: This script aims to investigate and visualize the generalizability matrix data.
##################################

#### Load libraries ####
library(ggplot2)
library(yaml)

#### Load data ####
Paths <- yaml::read_yaml("../Path_config.yaml")

load(file = paste0(Paths$dataset_path,"/","pattern_generalizability.RData"))
generalizability_matrix <- readRDS(file = paste0(Paths$dataset_path,"/","generalizability_matrix.rds"))
load(paste0(Paths$analysis_data,"Statistics/stats.RData"))


auc_scores <- sapply(unique(generalizability_matrix$Trained), function(cl){
  mean(generalizability_matrix$auc_score[generalizability_matrix$Trained==cl])
})

#### Sort cell lines by mean AUC score ####
if(Paths$Plot){
  png(paste0(Paths$Figures,"Analysis_generalizability","__1.png"),width = 500,height = 500) # figure __1
  print({
    ggplot() +
      geom_col(aes(x=reorder(names(auc_scores),-auc_scores), y =auc_scores, fill=names(auc_scores))) +
      theme(legend.position = "None") +
      labs(x="Trained cell line", y = "Average AUC score on tested cell line") +
      theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))
  })
  dev.off()
}

# #### AUC score trained vs untrained ####
trained_auc_score <- generalizability_matrix$auc_score[generalizability_matrix$Tested == generalizability_matrix$Trained]
untrained_auc_score <- generalizability_matrix$auc_score[generalizability_matrix$Tested != generalizability_matrix$Trained]

if(Paths$Plot){
  png(paste0(Paths$Figures,"Analysis_generalizability","__2.png"),width = 500,height = 500) # figure __2
  auc_comp_plot <- ggplot(data = auc_score_comp_df, aes(x=Group, y = Mean_auc_score, fill = Group)) +
    geom_col(width = 0.6) +
    geom_errorbar(position = "dodge2", width = 0.5, aes(x = Group, ymin = Mean_auc_score - SD_auc_score, ymax = Mean_auc_score + SD_auc_score)) +
    theme(legend.position = "None") + 
    geom_text(aes(label = Mean_auc_score, y = Mean_auc_score + 0.05), position = position_dodge(0.8), vjust = 0, size = 5) +
    geom_text(aes(label = paste("N = ", c(length(trained_auc_score),length(untrained_auc_score))), y = Mean_auc_score + 0.1), position = position_dodge(0.8), vjust = 0, size = 5) +
    labs(x = "", y = "Mean AUC score")
  print({
    auc_comp_plot
  })
  dev.off()
}



# merge the sample type:
generalizability_matrix <- merge(generalizability_matrix,stats[,c("Cell line","sample_type")],by.x = 1,by.y = 1, all.x = TRUE)
colnames(generalizability_matrix)[grep("^sample_type$",colnames(generalizability_matrix))] <- "sample_type_trained"

generalizability_matrix <- merge(generalizability_matrix,stats[,c("Cell line","sample_type")],by.x = 2,by.y = 1, all.x = TRUE)
colnames(generalizability_matrix)[grep("^sample_type$",colnames(generalizability_matrix))] <- "sample_type_tested"

# reorder Ids of the Trained samples by biological context
generalizability_matrix$Trained <- reorder(generalizability_matrix$Trained,sapply(paste0("^",generalizability_matrix$sample_type_trained,"$"), function(x){
  return(grep(x,c("PT","ESCD","ESC","PC","PCU","CL")))
}))

# reorder Ids of the Tested samples by biological context
generalizability_matrix$Tested <- reorder(generalizability_matrix$Tested,sapply(paste0("^",generalizability_matrix$sample_type_tested,"$"), function(x){
  return(grep(x,c("PT","ESCD","ESC","PC","PCU","CL")))
}))

#### Create heatmap ####
if(Paths$Plot){
  png(paste0(Paths$Figures,"Analysis_generalizability","__3.png"),width = 900,height = 700) # figure __3
  # svg("~/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/Generalizability_Heatmap.svg",
  #     width = 21,
  #     height = 18)
  pg_heatmap <- ggplot(generalizability_matrix, aes(Trained, Tested)) +
    geom_tile(aes(fill = auc_score))+ #, colour = "white") +
    scale_fill_viridis_c()+
    theme(text = element_text(size=24),axis.text.x=element_text(angle = 90)) #,)
  
  print({
    pg_heatmap
  })
  dev.off()
}
