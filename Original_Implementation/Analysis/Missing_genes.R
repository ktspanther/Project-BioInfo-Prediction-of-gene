library(ggplot2)
library(car)
library(ggpubr)
library(yaml)

Paths <- yaml::read_yaml("../Path_config.yaml")
setwd(Paths$PatternChrome_dir)
load(Paths$RNAseq_path)
load(paste0(Paths$main_path,"E003/H3K9me3_50bp_bins.RData"))


included_genes <- rownames(RNA_seq)[rownames(RNA_seq) %in% rownames(H3K9me3)]
excluded_genes <- rownames(RNA_seq)[!rownames(RNA_seq) %in% rownames(H3K9me3)]

stats_df <- data.frame()
hist_df <- data.frame()
cell_lines <- colnames(RNA_seq)
for (cl in cell_lines) {
  RNA_included <- as.numeric(RNA_seq[included_genes,cl])
  RNA_included <- log(RNA_included + 0.1)
  RNA_excluded <- as.numeric(RNA_seq[excluded_genes,cl])
  RNA_excluded <- log(RNA_excluded + 0.1)
  mean_included <- mean(RNA_included)
  mean_excluded <- mean(RNA_excluded)
  mean_diff <- mean_included - mean_excluded
  sd_included <- sd(RNA_excluded)
  sd_excluded <- sd(RNA_included)
  sd_diff <- sd_included - sd_excluded
  t_test_stats <- t.test(RNA_excluded, RNA_included)
  p <- t_test_stats$p.value
  stats_df <- rbind.data.frame(stats_df, c(cl, mean_included, mean_excluded, mean_diff, sd_included, 
    sd_excluded,sd_diff, p))
  hist_df <- rbind.data.frame(hist_df, data.frame(c(RNA_included, RNA_excluded),
    c(rep("Included", length(included_genes)),rep("Excluded", length(excluded_genes)))))
}


cell_line <- "E003"

RNA <- as.numeric(RNA_seq[,cell_line])
RNA_log <- log(RNA + 0.1)
median <- median(RNA_log)


colnames(hist_df) <- c("Value", "Group")
group_size <- table(hist_df$Group)
if(Paths$Plot){
  png(paste0(Paths$Figures,"Missing_genes","__1.png"),width = 500,height = 500) # figure __1
  print({
    ggplot(hist_df, aes(x=Group, y=Value, fill = Group)) +
      geom_violin(draw_quantiles = c(0.25, 0.5, 0.75)) +
      annotate(geom = "text", label = paste("N =", group_size), x = 1:2, y = 13) +
      labs(x="",y="Log-transformed gene expression reads") +
      theme(legend.position = "None")
  })
  dev.off()
}

t.test(hist_df$Value[hist_df$Group== "Included"], hist_df$Value[hist_df$Group== "Excluded"])

if(Paths$Plot){
  png(paste0(Paths$Figures,"Missing_genes","__2.png"),width = 500,height = 500) # figure _2
  # svg(paste0("/sybig/home/npa/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/",
  #            "Missing_genes","_2.svg"),width = 8,height = 3.5)
  print({
    ggplot(hist_df, aes(x=Value, fill = Group)) +
      geom_histogram(bins = 100, position = "dodge") +
      labs(x="Log-transformed gene expression reads", y = "Count")
  })
  dev.off()
}

mean(hist_df$Value[hist_df$Group=="Excluded"])


#### 0 Read proportion ####
nrow(hist_df[hist_df$Value == min(hist_df[,1]) &hist_df$Group == "Excluded"  ,])/
  nrow(hist_df[hist_df$Group == "Excluded"  ,])
nrow(hist_df[hist_df$Value == min(hist_df[,1]) &hist_df$Group == "Included"  ,]) /
  nrow(hist_df[hist_df$Group == "Included"  ,])

#### Proportion below and above median ####

nrow(hist_df[hist_df$Value < median(hist_df$Value) & hist_df$Group == "Excluded"  ,])/
  nrow(hist_df[hist_df$Group == "Excluded"  ,])
nrow(hist_df[hist_df$Value < median(hist_df$Value) & hist_df$Group == "Included"  ,]) /
  nrow(hist_df[hist_df$Group == "Included"  ,])

     