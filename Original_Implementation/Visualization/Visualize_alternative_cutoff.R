#####################################################
# Author: Jonas Wolber
# Date: 14.09.22
# Description: Aim of this script is to make a cutoff point for binary classification of the RNA Seq data.
#####################################################

#### Load libraries ####
library(ggplot2)
library(dplyr)
library(yaml)

#### Load RNA Seq file ####
Paths <- yaml::read_yaml("../Path_config.yaml")
load(Paths$RNAseq_path)

cell_lines <- list.dirs(full.names = FALSE, path = Paths$dataset_path)
cell_lines <- cell_lines[grep(pattern = "^E\\d{3}$",cell_lines)]
cell_line <- cell_lines[1]
load(paste(Paths$main_path,cell_line,"/H3K4me3_50bp_bins.RData", sep =""))
RNA <- as.numeric(RNA_seq[rownames(H3K4me3),cell_line])
RNA_log <- log(RNA + 0.01)
RNA_df <- data.frame(GE_log = RNA_log)

#### Calculate alternative cutoff by finding the valley in the distribution between q2 and the median ####
median_ge <- median(RNA_log)
q2_ge <- quantile(RNA_log)[2]
q2 <- as.vector(quantile(RNA_log))[2]
q3 <- as.vector(quantile(RNA_log))[3]
RNA_values_around_ac <- RNA_log[RNA_log>quantile(RNA_log)[2] & RNA_log<quantile(RNA_log)[3]]

bins <- seq(q2,q3,length.out = round(length(RNA_values_around_ac)/100))
bin_freq <- rep(0,length(bins))
for (b in 1:(length(bins)-1)) {
  bin_freq[b] <- sum(between(RNA_values_around_ac, bins[b], bins[b+1]))
}
bin_freq <- bin_freq[-length(bin_freq)]
alternative_cutoff <- bins[which.min(bin_freq)]

#### Aesthetics ####
text_size <- 3
bins <- 100
segment_size <- 0.8

if(Paths$Plot){
  png(paste0(Paths$Figures,"Visualize_alternative_cutoff","__1.png"),width = 500,height = 500) # figure __1
  # svg(paste0("/sybig/home/npa/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/","Visualize_alternative_cutoff","__1.svg"),width = 8,height = 5)
  print({
    ggplot(data = RNA_df, aes(x=GE_log)) +
      geom_histogram(bins =  bins) +
      annotate(geom = "segment", x = median_ge, xend = median_ge, y = 0, yend = 500, color = "red", size = segment_size) +
      annotate(geom = "text", label = "Median", x = median_ge, y = - 30, color = "red", size = text_size) +
      annotate(geom = "segment", x = q2_ge, xend = q2_ge, y = 0, yend = 500, color = "red", size = segment_size) +
      annotate(geom = "text", label = "Second quantile", x = q2_ge, y = - 30, color = "red", size = text_size) +
      annotate(geom = "segment", x = alternative_cutoff, xend = alternative_cutoff, y = 0, yend = 500, color = "blue", size = segment_size) +
      annotate(geom = "text", label = "Alternative_cutoff", x = alternative_cutoff, y = - 100, color = "blue", size = text_size) +
      labs(x="Log transformed gene expression reads", y = "Count")
  })
  dev.off()
}
