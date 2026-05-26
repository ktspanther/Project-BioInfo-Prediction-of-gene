######################################################
# Author: Jonas Wolber
# Date: 08.09.22
# Description: The aim of this script is to visualize the input data for my classifier.
# For this the binned epigenome regions of the five histone modifications for a gene are plotted.
# Content: 
# 1. Plot showing input data
######################################################

#### Load libraries ####
library(ggplot2)
library(ggpubr)
library(yaml)

#### Load pattern data ####
Paths <- yaml::read_yaml("../Path_config.yaml")
setwd(paste0(Paths$main_path,"E003/"))
load("H3K4me1_50bp_bins.RData")
load("H3K4me3_50bp_bins.RData")
load("H3K9me3_50bp_bins.RData")
load("H3K27me3_50bp_bins.RData")
load("H3K36me3_50bp_bins.RData")

#### Parameters ####
gene <- 100
y_lab_size <- 8
line_size <- 1.0
Positions <- -99:100
ylab <- "Mean histone read"
xlab <- "Position on binned epigenome"
y_text_height <- -1
text_size <- 3.5
#"#F8766D" "#A3A500" "#00BF7D" "#00B0F6" "#E76BF3"
#### Plot figure ####
if(Paths$Plot){
  png(paste0(Paths$Figures,"Visualize_input_data","__1.png"),width = 500,height = 500) # figure __1
  
  H3K4me3_plot <- ggplot() + 
    geom_line(aes(x=Positions, y = H3K4me3[gene,]), color = "#F8766D", size = line_size) +
    labs(x = xlab, y = ylab) +
    theme(axis.title.y = element_text(size=y_lab_size))
  H3K4me1_plot <- ggplot() + 
    geom_line(aes(x=Positions, y = H3K4me1[gene,]), color = "#A3A500", size = line_size) +
    labs(x = xlab, y = ylab) +
    theme(axis.title.y = element_text(size=y_lab_size))
  H3K36me3_plot <- ggplot() + 
    geom_line(aes(x=Positions, y = H3K36me3[gene,]), color = "#00BF7D", size = line_size) +
    labs(x = xlab, y = ylab) +
    theme(axis.title.y = element_text(size=y_lab_size))
  H3K27me3_plot <- ggplot() + 
    geom_line(aes(x=Positions, y = H3K27me3[gene,]), color = "#00B0F6", size = line_size) +
    labs(x = xlab, y = ylab) +
    theme(axis.title.y = element_text(size=y_lab_size))
  H3K9me3_plot <- ggplot() + 
    geom_line(aes(x=Positions, y = H3K9me3[gene,]), color = "#E76BF3", size = line_size) +
    labs(x = xlab, y = ylab) +
    theme(axis.title.y = element_text(size=y_lab_size)) +
    annotate(geom = "text", label = "TSS", x = 0, y = y_text_height, size = text_size) +
    annotate(geom = "text", label = "Upstream", x = -50, y = y_text_height, size = text_size) +
    annotate(geom = "text", label = "Downstream", x = 50, y = y_text_height, size = text_size) +
    annotate(geom = "text", label = " ", x = 0, y = y_text_height - 1, size = text_size) +
    scale_y_continuous(breaks = y_text_height:3, labels = c(" ", 0:3))
  print({
    ggarrange(H3K4me3_plot, H3K4me1_plot, H3K36me3_plot, H3K27me3_plot, H3K9me3_plot, nrow = 5,
              labels = c("H3K4me3", "H3K4me1", "H3K36me3", "H3K27me3", "H3K9me3"),
              hjust = -0.7, vjust = 1, font.label = list(size = 11))
    
  })
  dev.off()
}
