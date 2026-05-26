####################################################
# Author: Jonas Wolber
# Date: 21.09.22.
# Description: This script aims to visualize where in the histone signal the pattern is found
####################################################

#### Load libraries ####
library(ggplot2)
library(ggpubr)
library(yaml)

#### Parameters ####
set.seed(42)

#### Load data ####
Paths <- yaml::read_yaml("../Path_config.yaml")

cell_line <- "E003"

cell_line_files <- list.files(paste0(Paths$dataset_path,"/",cell_line))
Train_df_file <- cell_line_files[grep("\\d_train_df.RData",cell_line_files)[1]]
Validation_df_file <- cell_line_files[grep("validation_df.RData",cell_line_files)[1]]
Test_df_file <- cell_line_files[grep("\\d_test_df.RData",cell_line_files)[1]]

load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Train_df_file))
load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Validation_df_file))
load(file = paste0(Paths$dataset_path,"/",cell_line,"/",Test_df_file))


load(paste0(Paths$analysis_data,"all_patterns.RData"))
patterns <- all_patterns[all_patterns$Cell_line == cell_line,-sapply(c("Importance","Correlation"), grep,colnames(all_patterns))]


#### Aesthetics ####
point_size <- 5
line_size <- 1.2

#### Pattern plot ####
pattern <- 1
mp <- as.numeric(patterns[pattern,8:(7+patterns$Width[pattern])])
pattern_df <- data.frame(Position = 1:patterns$Width[pattern], 
  Height = as.numeric(mp), 
  pattern = "H3K4me3")

if(Paths$Plot){
  png(paste0(Paths$Figures,"Pattern_location_visualization","__1.png"),width = 500,height = 500) # figure __1
  # svg(paste0("/sybig/home/npa/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/","Pattern_location_visualization","__1.svg"),width = 8,height = 5)
  print({
    pattern_plot <- ggplot(data = pattern_df, aes(x=Position, y=Height, col = pattern)) + 
      geom_line(size = line_size) +
      geom_point(size = point_size) +
      labs(x="Position", y = "Relative height") +
      theme(legend.position = "None")
    pattern_plot
  })
  dev.off()
}

#### Find matches in histone signal data ####
load(paste0(Paths$main_path,cell_line,"/H3K4me3_50bp_bins.RData"))

# pattern
genes <- rownames(validation_df)[rownames(validation_df) %in% rownames(H3K4me3)]
start <- patterns$Start[pattern]
end <- patterns$End[pattern]
checked_positions <- start:end
MP_threshold <- as.numeric(patterns$MP_threshold[pattern])

gene <- "ENSG00000204531"

visualize_matches <- function(gene,gene_symbol){
  hm <- H3K4me3[gene, start:end]
  checked_positions <- 1:(length(hm)-length(mp)+1) # Niels: +1
  matches <- sapply(checked_positions, function(pos){cor(mp, hm[pos:(pos+length(mp)-1)])>MP_threshold})
  matches[is.na(matches)] <- FALSE 
  matches <-  checked_positions[matches]
  line_df <- data.frame(Position = checked_positions, Signal = H3K4me3[gene,checked_positions], HM = "H3K4me3")
  
  cors <- sapply(checked_positions, function(pos){cor(mp, hm[pos:(pos+length(mp)-1)])})
  cors[is.na(cors)] <- 0 
  cors[cors < 0] <- 0
  cor_df <- data.frame(Position = checked_positions, Signal = cors, HM = "H3K4me3")
  
  match_plot <- ggplot() + 
    geom_line(data = line_df, aes(x=Position,y=Signal, col = HM), size = line_size) +
    geom_point(aes(x=matches,y=H3K4me3[gene,matches]), size = 2) +
    labs(x="Position on binned epigenome",y="Read frequency") +
    theme(legend.position = "None") +
    scale_x_continuous(breaks = seq(range(line_df$Position)[1],range(cor_df$Position)[2],length.out=5),labels = seq(patterns$Start[pattern],patterns$End[pattern],length.out=5)-100) +
    ggtitle(paste("Gene ", gene_symbol))
  
    cor_plot <- ggplot() +
    geom_line(data = cor_df, aes(x=Position,y=Signal, col = HM), size = 1) +
    geom_hline(yintercept=MP_threshold, linetype='dashed', col = 'blue', size = 1) +
    theme(legend.position = "None") +
    labs(x="Position on binned epigenome", y = "r") +
    scale_x_continuous(breaks = seq(range(cor_df$Position)[1],range(cor_df$Position)[2],length.out=5),labels = seq(patterns$Start[pattern],patterns$End[pattern]-length(mp)+1,length.out=5)-100) +
    ggtitle(paste("Pattern frequency: ", length(matches)))
  return(ggarrange(match_plot, cor_plot, nrow = 2, ncol = 1))
}

visualize_gene <- function(gene_high, gene_low){
  high <- visualize_matches(gene_high,gene_symbol = "OCT4")
  low <- visualize_matches(gene_low,gene_symbol = "TIE1")
  return(ggarrange(high, low, nrow = 1, ncol = 2, 
    vjust = 1.0, hjust = -0.25,font.label = list(size = 12))
    )
}
# 1. Oct4, 2. TIE1
if(Paths$Plot){
  # png(paste0(Paths$Figures,"Pattern_location_visualization","__2.png"),width = 500,height = 500) # figure __2
  svg(paste0("/sybig/home/npa/Dokumente/Promotion_Niels/Paper_PatternChrome/patternchrome/Manuscript/Figures_paper/","Pattern_location_visualization","_2.svg"),width = 18,height = 5)
  print({
    visualize_gene("ENSG00000204531","ENSG00000066056")
  })
  dev.off()
}

# validation_df["ENSG00000204531",2]
# train_df["ENSG00000066056",2]
