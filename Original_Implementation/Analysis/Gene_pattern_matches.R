#########################################
# Description: This script aims to compare the locations of 
# trained patterns from cell line E105 in all 56 cell lines
########################################

#### Load libraries ####
library(ggplot2)
library(ggpubr)
library(yaml)

#### Load data ####
Paths <- yaml::read_yaml("../Path_config.yaml")
setwd(Paths$PatternChrome_dir)
load(paste0(Paths$analysis_data,"all_patterns.RData"))

cell_lines <- list.dirs(full.names = FALSE, path = Paths$dataset_path)
cell_lines <- cell_lines[grep(pattern = "^E\\d{3}$",cell_lines)]

#### Filter patterns ####
trained_cell_line <- "E105"
patterns <- all_patterns[all_patterns$Cell_line==trained_cell_line,]
rm(all_patterns)

#### Gene ID ####
id <- "ENSG00000163602" #RYBP #"ENSG00000204531" # OCT4

#### Get patterns with highest importance ####
top_patterns <- which(patterns$Importance %in% sort(patterns$Importance,decreasing = T)[1:5])
#### Initialize dataframe ####
match_freq <- data.frame(Positions=rep(1:200,length(top_patterns)),
                         Freq=rep(0,length(top_patterns)), 
                         Pattern = c(sapply(top_patterns, function(p){rep(p,200)})))

#### Get matches ####
for (p in top_patterns) {
  print(p)
  #### Pattern characteristics ####
  width <- patterns[p,6]
  mp <- as.numeric(unlist(patterns[p,7:(6+width)]))
  mp_threshold <- as.numeric(patterns[p,5])
  start <- 1#min(patterns[p,2:3])
  end <- 200#max(patterns[p,2:3])
  hm <- patterns[p,4]
  
  #### Get match positions in all cell lines ####
  for(cell_line in cell_lines){
    switch (hm,
            load(paste(Paths$main_path,cell_line,"/H3K4me3_50bp_bins.RData",sep="")),
            load(paste(Paths$main_path,cell_line,"/H3K4me1_50bp_bins.RData",sep="")),
            load(paste(Paths$main_path,cell_line,"/H3K36me3_50bp_bins.RData",sep="")),
            load(paste(Paths$main_path,cell_line,"/H3K27me3_50bp_bins.RData",sep="")),
            load(paste(Paths$main_path,cell_line,"/H3K9me3_50bp_bins.RData",sep=""))
    )
    switch (hm,
            search_area <- H3K4me3[id,start:end],
            search_area <- H3K4me1[id,start:end],
            search_area <- H3K36me3[id,start:end],
            search_area <- H3K27me3[id,start:end],
            search_area <- H3K9me3[id,start:end]
    )
    checked_positions <- 1:(length(search_area)-width)
    matches <- which(sapply(checked_positions, function(pos){
      frame <- search_area[pos:(pos+width-1)]
      if(sum(frame)==0){return(F)} # skip if there is no chIP-seq read (gives error on cor())
      else{cor(mp, frame)>mp_threshold} # return()
    }))
    match_positions <- start + matches - 1
    match_freq[match_freq$Positions %in% match_positions & match_freq$Pattern==p,2] <- 
      match_freq[match_freq$Positions %in% match_positions & match_freq$Pattern==p,2] + 1
  }
}

match_freq$Pattern <- c(sapply(match_freq[,3], function(h){
  switch (patterns[h,4],
    paste(h, "H3K4me3"),
    paste(h, "H3K4me1"),
    paste(h, "H3K36me3"),
    paste(h, "H3K27me3"),
    paste(h, "H3K9me3")
  ) 
}))

#### Visualize match positions ####
if(Paths$Plot){
  png(paste0(Paths$Figures,"Gene_pattern_matches","__1.png"),width = 500,height = 500) # figure __1
  print({
    ggplot(match_freq,aes(x=Positions,y=Freq, col=as.factor(Pattern))) +
      geom_line() +
      labs(x="Positions",y="Match frequency", col="Pattern")
  })
  dev.off()
}
  
#### Visualize pattern ####
if(Paths$Plot){
  png(paste0(Paths$Figures,"Gene_pattern_matches","__2.png"),width = 500,height = 500) # figure __2
  print({
    ggplot() +
      geom_line(aes(x=1:length(mp),y=mp))
  })
  dev.off()
}


#### Visualize matches ####
if(Paths$Plot){
  png(paste0(Paths$Figures,"Gene_pattern_matches","__3.png"),width = 500,height = 500) # figure __3
  print({ggplot() +
      geom_line(aes(x=1:length(search_area),y=search_area)) +
      geom_point(aes(x=matches,y=search_area[matches]))
  })
  dev.off()
}
