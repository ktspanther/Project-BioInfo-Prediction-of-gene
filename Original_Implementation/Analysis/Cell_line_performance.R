#########################################################
# Date: 16.10.
# Description: Investigate the best and worst performing algorithms of the PatternChrome algorithm
#########################################################

#### Load libraries ####
library(ggplot2)
library(yaml)

#### Load data ####
Paths <- yaml::read_yaml("../Path_config.yaml")

# load functions
source(Paths$Custom_functions)

setwd(Paths$PatternChrome_dir)
load(Paths$RNAseq_path)
load(paste0(Paths$analysis_data,"Statistics/stats.RData"))
load(paste0(Paths$main_path,"E003/H3K4me3_50bp_bins.RData"))

#### Modify data ####
cell_lines <- stats$`Cell line`
genes <- rownames(H3K4me3)[rownames(H3K4me3) %in% rownames(RNA_seq)]
RNA_seq <- RNA_seq[genes,]
performance <- as.numeric(stats$`Mean AUC score`)

#### Sort cell lines by performance ####
worst_performing_cl <- stats$`Cell line`[which.min(stats$`Mean AUC score`)]
best_performing_cl <- stats$`Cell line`[which.max(stats$`Mean AUC score`)]

if(Paths$Plot){
  png(paste0(Paths$Figures,"Cell_line_performance","__1.png"),width = 500,height = 500) # figure __1
  print({
    visualize_RNA_dist(worst_performing_cl,RNAseq_mat = RNA_seq)
  })
  dev.off()
}


#### Correlation between performance and RNA seq data ####
rna_diff_df <- data.frame()
for (cl in cell_lines) {
  RNA <- as.numeric(RNA_seq[,cl])
  RNA_log <- log(RNA + 0.01)
  median_RNA <- median(RNA_log)
  RNA_binarized <- (RNA > median_RNA)+0
  RNA_df <- data.frame(RNA_log = RNA_log, RNA_binarized = RNA_binarized)
  rownames(RNA_df) <- genes
  sum_RNA <- sum(RNA_df$RNA_log)
  RNA_zeros <- length(RNA[RNA==0])
  sd_RNA <- sd(RNA_log)
  high_rna_genes <- genes[RNA > median(RNA)]
  low_rna_genes <- genes[RNA <= median(RNA)]
  sd_RNA_groups <- sd(RNA_df[high_rna_genes,1])-sd(RNA_df[low_rna_genes,1])
  diff_RNA_groups <- sum(RNA_df[high_rna_genes,1])-sum(RNA_df[low_rna_genes,1])
  rna_diff_df <- rbind.data.frame(rna_diff_df, c(sum_RNA, sd_RNA, RNA_zeros,sd_RNA_groups,diff_RNA_groups))
}
colnames(rna_diff_df) <- c("RNA_sum","RNA_sd", "RNA_zeros","RNA_group_sd_diff","RNA_group_sum_diff")

#### Correlation between difference in Chip Seq counts and performance ####
chip_seq_df <- data.frame()
for (cl in cell_lines) {
  load(paste(Paths$main_path,cl,"/H3K4me1_50bp_bins.RData", sep = ""))
  load(paste(Paths$main_path,cl,"/H3K4me3_50bp_bins.RData", sep = ""))
  load(paste(Paths$main_path,cl,"/H3K9me3_50bp_bins.RData", sep = ""))
  load(paste(Paths$main_path,cl,"/H3K27me3_50bp_bins.RData", sep = ""))
  load(paste(Paths$main_path,cl,"/H3K36me3_50bp_bins.RData", sep = ""))
  
  high_rna_genes <- genes[as.numeric(RNA_seq[,cl]) > median(as.numeric(RNA_seq[,cl]))]
  low_rna_genes <- genes[as.numeric(RNA_seq[,cl]) <= median(as.numeric(RNA_seq[,cl]))]
  H3K4me3_sum_diff <- sum(H3K4me3[high_rna_genes,]) - sum(H3K4me3[low_rna_genes,])
  H3K4me1_sum_diff <- sum(H3K4me1[high_rna_genes,]) - sum(H3K4me1[low_rna_genes,])
  H3K36me3_sum_diff <- sum(H3K36me3[high_rna_genes,]) - sum(H3K36me3[low_rna_genes,])
  H3K27me3_sum_diff <- sum(H3K27me3[high_rna_genes,]) - sum(H3K27me3[low_rna_genes,])
  H3K9me3_sum_diff <- sum(H3K9me3[high_rna_genes,]) - sum(H3K9me3[low_rna_genes,])
  H3K4me3_sd_diff <- sd(H3K4me3[high_rna_genes,]) - sd(H3K4me3[low_rna_genes,])
  H3K4me1_sd_diff <- sd(H3K4me1[high_rna_genes,]) - sd(H3K4me1[low_rna_genes,])
  H3K36me3_sd_diff <- sd(H3K36me3[high_rna_genes,]) - sd(H3K36me3[low_rna_genes,])
  H3K27me3_sd_diff <- sd(H3K27me3[high_rna_genes,]) - sd(H3K27me3[low_rna_genes,])
  H3K9me3_sd_diff <- sd(H3K9me3[high_rna_genes,]) - sd(H3K9me3[low_rna_genes,])
  H3K4me3_sum <- sum(H3K4me3)
  H3K4me1_sum <- sum(H3K4me1)
  H3K36me3_sum <- sum(H3K36me3)
  H3K27me3_sum <- sum(H3K27me3)
  H3K9me3_sum <- sum(H3K9me3)
  H3K4me3_sd <- sd(H3K4me3)
  H3K4me1_sd <- sd(H3K4me1)
  H3K36me3_sd <- sd(H3K36me3)
  H3K27me3_sd <- sd(H3K27me3)
  H3K9me3_sd <- sd(H3K9me3)
  chip_seq_df <- rbind.data.frame(chip_seq_df, c( 
    H3K4me3_sum_diff,H3K4me1_sum_diff,H3K36me3_sum_diff,H3K27me3_sum_diff,H3K9me3_sum_diff,
    H3K4me3_sd_diff,H3K4me1_sd_diff,H3K36me3_sum_diff,H3K27me3_sd_diff,H3K9me3_sd_diff,
    H3K4me3_sum,H3K4me1_sum,H3K36me3_sum,H3K27me3_sum,H3K9me3_sum,
    H3K4me3_sd,H3K4me1_sd,H3K36me3_sd,H3K27me3_sd,H3K9me3_sd))
}
colnames(chip_seq_df) <- c("H3K4me3_sum_diff","H3K4me1_sum_diff","H3K36me3_sum_diff",
                           "H3K27me3_sum_diff","H3K9me3_sum_diff","H3K4me3_sd_diff","H3K4me1_sd_diff",
                           "H3K36me3_sd_diff","H3K27me3_sd_diff","H3K9me3_sd_diff",
                           "H3K4me3_sum","H3K4me1_sum","H3K36me3_sum","H3K27me3_sum","H3K9me3_sum",
                           "H3K4me3_sd","H3K4me1_sd","H3K36me3_sd","H3K27me3_sd","H3K9me3_sd")
cor_df <- cbind(performance,rna_diff_df,chip_seq_df)
colnames(cor_df)[1] <- "AUCs"
# colnames(cor_df) <- c("AUCs",colnames(rna_diff_df), colnames(chip_seq_df))

model <- lm(AUCs ~ ., data = cor_df)
model_summary <-  summary(model)
coefficients <-  model_summary$coefficients


# cor_df <- cor_df[,c("AUCs",rownames(coefficients)[-1])]


#### Backward elimination ####
adj_r_squared <- model_summary$adj.r.squared
improving <- TRUE
while(improving){
  least_significant_var <- which(colnames(cor_df)==rownames(coefficients)[which.max(coefficients[,4])])
  be_cor_df <- cor_df[,-least_significant_var]
  be_model <- lm(AUCs ~ ., data = be_cor_df)
  be_model_summary <-  summary(be_model)
  be_adj_r_squared <- be_model_summary$adj.r.squared
  if(be_adj_r_squared < adj_r_squared){break}
  print(paste(rownames(coefficients)[which.max(coefficients[,4])], be_adj_r_squared))
  adj_r_squared <- be_adj_r_squared
  cor_df <- be_cor_df
  coefficients <-  be_model_summary$coefficients
}

new_df <- cbind.data.frame(rna_diff_df,chip_seq_df)
be_model_summary

auc_preds <-  predict(be_model,newdata =  new_df)

pred_df <- data.frame(Observations = performance, Predictions = auc_preds, col = "Col")

if(Paths$Plot){
  # png(paste0(Paths$Figures,"Cell_line_performance","__2.png"),width = 500,height = 500) # figure __2
  svg(paste0("/sybig/home/npa/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/Cell_line_performance","__2.svg"),width = 8,height = 5)
  print({
    ggplot(data = pred_df, aes(x=Predictions,y=Observations, col = col)) + 
      geom_point() +
      labs(x="Predicted AUC score", y = "Observed AUC score") +
      theme(legend.position = "None")
  })
  dev.off()
}

cor(auc_preds, performance)
sort(coefficients[,4])

#### Correlation between pattern number and performance ####
stats$color <- "color"
pattern_number_df <- data.frame(Number = stats$`Number of trained patterns`, AUC = performance, col = "Col")

if(Paths$Plot){
  png(paste0(Paths$Figures,"Cell_line_performance","__3.png"),width = 500,height = 500) # figure __3
  print({
    ggplot(data = pattern_number_df, aes(x=Number, y = AUC, col = col)) +
      geom_point() +
      theme(legend.position = "None") +
      labs(x="Number of patterns used in XGBoost classifier", y = "Mean AUC score") +
      geom_smooth(method = "lm")
  })
  dev.off()
}


#### Tissue type ####
# types <- c("PCU", "ESCD","ESCD","ESCD","ESCD","ESCD","ESCD","ESCD","ESC","ESC","PC",
#            "PCU","PC","PC","PC","PC","PCU", "PCU","PCU", "PCU","PCU","PCU","PCU","PCU",
#            "PC","PT","PT","PT","PT","PT","PT","PT","PT","PT","PT","PT","PT","PT","PT",
#            "PT","PT","PT","PT","PT","PT","PT","CL","PCU","CL","CL","PCU","PCU","PCU",
#            "PCU","PCU", "PCU")



# stats$sample_type <- sample_sample_type
sample_type_df <- data.frame(Performance = as.numeric(stats$`Mean AUC score`), Type = stats$sample_type)
group_size <- sapply(unique(stats$sample_type), function(t){length(stats$sample_type[stats$sample_type==t])})
group_size <- c(3,2,7,6,17,21)
group_means <- round(sapply(unique(stats$sample_type), function(t){mean(as.numeric(stats$`Mean AUC score`)[stats$sample_type==t])}),4)

if(Paths$Plot){
  png(paste0(Paths$Figures,"Cell_line_performance","__4.png"),width = 500,height = 500) # figure __4
  # svg("~/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/Performance_sampletype_18022024.svg",
  #     width = 9,
  #     height = 6)
  print({
    ggplot(data = sample_type_df,aes(x=Type,y=Performance, fill=Type)) +
      geom_violin() +
      labs(x="Tissue type", y="Mean AUC score ", fill = "Tissue type") +
      annotate(geom = "text", label = paste("N =", group_size), x= 1:6, y = 0.9525, size = 3.5)
  })
  dev.off()  
}

performance_tissue_df <- cbind.data.frame(performance, rna_diff_df, chip_seq_df)
profile_tissue_df <- data.frame()
for (col in 1:ncol(performance_tissue_df)) {
  param_cor <- cor(performance_tissue_df[,1],performance_tissue_df[,col])
  profile_tissue_df <- rbind.data.frame(profile_tissue_df,c(colnames(new_df)[col],param_cor))
}
colnames(profile_tissue_df) <- c("Parameter", "Correlation")
profile_tissue_df$Correlation <- as.numeric(profile_tissue_df$Correlation)
profile_tissue_df$Absolute_correlation <- abs(profile_tissue_df$Correlation)

profile_tissue_df <- profile_tissue_df[profile_tissue_df$Absolute_correlation %in% 
          sort(profile_tissue_df$Absolute_correlation,decreasing = T)[1:8],]

profile_tissue_df$Parameter <- c("RNA Sum","RNA Sum Diff","RNA 0", "RNA Sd",
  "RNA Sd Diff","H3K4me1 SD", "H3K4me1 Sum", "H3K9me3 Sum Diff")

if(Paths$Plot){
  png(paste0(Paths$Figures,"Cell_line_performance","__5.png"),width = 500,height = 500) # figure __5
  print({
    ggplot(data = profile_tissue_df, aes(x=reorder(Parameter,Absolute_correlation),
                                         y=Correlation,fill = Parameter)) +
      geom_col(width = 0.6) +
      labs(x="Parameter",y="Pearson R squared correlation") +
      theme(axis.text.x = element_text(size=8, angle=30))
  })
  dev.off()
}
