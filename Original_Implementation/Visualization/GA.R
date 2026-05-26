####

# This is code to get the figures for the General Abstract

#### Load libraries ####
library(ggplot2)
library(caTools)
library(dplyr)
library(data.table)
library(xgboost)

#### Load data ####
load("~/patternchrome/datasets/Binned_sequencing_data/E003/H3K4me3_50bp_bins.RData")
load("~/patternchrome/datasets/RNA_seq.RData")
load("~/patternchrome/datasets/E003/E003__18_09_patterns.RData")
load("~/patternchrome/datasets/E003/E003__18_09_train_df.RData")

#### Specify gene, cell_line and pattern ####
gene <- rownames(train_df)[1]
cell_line <- "E003"
pattern <- 1
mp <- as.numeric(patterns[pattern,7:(6+patterns$Width[pattern])])
pattern_df <- data.frame(Position = 1:patterns$Width[pattern], 
                         Height = as.numeric(mp), 
                         pattern = "H3K4me3")

#### Aesthetics ####
hm_color = "blue"
pattern_color = "orange"
segment_color = "red"
rna_color = ""
segment_size <- 3
hm_line_size <- 2
hm_point_size <- 8
pattern_line_size <- 5
point_size <- 14
Positions <- -99:100
MP_threshold <- 0.5

#### Histone modifcation profile visualization ####
ggplot() + 
  geom_line(aes(x=Positions, y = H3K4me3[gene,]), color = hm_color, size = hm_line_size) +
  labs(x = "", y = "", size=2) +
  theme_void()

#### Plot a single pattern ####
ggplot(data = pattern_df, aes(x=Position, y=Height)) + 
  geom_line(linewidth = pattern_line_size, col = pattern_color) +
  geom_point(size = point_size, col = pattern_color) +
  theme_void()

#### Gene expression classification visualization ####
RNA <- as.numeric(RNA_seq[rownames(RNA_seq),cell_line])
RNA_log <- log(RNA + 0.01)
RNA_df <- data.frame(GE_log = RNA_log)
RNA_df$col <- "col"
bins <- 100
median <- median(RNA_df$GE_log)
ggplot(data = RNA_df, aes(x=GE_log)) +
  geom_histogram(bins =  bins) +
  annotate(geom = "segment", x = median, xend = median, y = 0, yend = 1000, color = "red", size = segment_size) +
  labs(x="", y = "") +
  theme_void()

#### Pattern location visualization ####
MP_threshold <- as.numeric(patterns$MP_threshold[pattern])
hm <- H3K4me3[gene, patterns$Start[pattern]:patterns$End[pattern]]
checked_positions <- 1:(length(hm)-length(mp))
matches <- sapply(checked_positions, function(pos){cor(mp, hm[pos:(pos+length(mp)-1)])>MP_threshold})
matches[is.na(matches)] <- FALSE 
matches <-  checked_positions[matches]
line_df <- data.frame(Position = checked_positions, Signal = H3K4me3[gene,checked_positions], HM = "H3K4me3")
ggplot() + 
  geom_line(data = line_df, aes(x=Position,y=Signal), size = hm_line_size, col=hm_color) +
  geom_point(aes(x=matches,y=H3K4me3[gene,matches]), size = hm_point_size) +
  theme_void()

#### Feature importance visualization ####
buildExplainer = function(xgb.model, trainingData, type = "binary", base_score = 0.5, trees_idx = NULL){
  
  #col_names = attr(trainingData, ".Dimnames")[[2]]
  col_names = colnames(trainingData)
  trees = xgb.model.dt.tree(col_names, model = xgb.model, trees = trees_idx)
  nodes.train = predict(xgb.model,trainingData,predleaf =TRUE)
  tree_list = getStatsForTrees(trees, nodes.train, type = type, base_score = base_score)
  explainer = buildExplainerFromTreeList(tree_list,col_names)
  return (explainer)
}
buildExplainerFromTreeList = function(tree_list,col_names){
  
  ####accepts a list of trees and column names
  ####outputs a data table, of the impact of each variable + intercept, for each leaf
  
  tree_list_breakdown <- vector("list", length(col_names)  + 3)
  names(tree_list_breakdown) = c(col_names,'intercept', 'leaf','tree')
  num_trees = length(tree_list)
  
  for (x in 1:num_trees){
    tree = tree_list[[x]]
    tree_breakdown = getTreeBreakdown(tree, col_names)
    tree_breakdown$tree = x - 1
    tree_list_breakdown = rbindlist(append(list(tree_list_breakdown),list(tree_breakdown)))
  }
  
  return (tree_list_breakdown)
  
}
explainPredictions = function(xgb.model, explainer ,data){
  
  #Accepts data table of the breakdown for each leaf of each tree and the node matrix
  #Returns the breakdown for each prediction as a data table
  
  nodes = predict(xgb.model,data,predleaf =TRUE)
  
  colnames = names(explainer)[1:(ncol(explainer)-2)]
  
  preds_breakdown = data.table(matrix(0,nrow = nrow(nodes), ncol = length(colnames)))
  setnames(preds_breakdown, colnames)
  num_trees = ncol(nodes)
  for (x in 1:num_trees){
    nodes_for_tree = nodes[,x]
    tree_breakdown = explainer[tree==x-1]
    
    preds_breakdown_for_tree = tree_breakdown[match(nodes_for_tree, tree_breakdown$leaf),]
    preds_breakdown = preds_breakdown + preds_breakdown_for_tree[,colnames,with=FALSE]
  }
  
  return (preds_breakdown)
  
}
findLeaves = function(tree, currentnode){
  
  if (tree[currentnode,'Feature']=='Leaf'){
    leaves = currentnode
  }else{
    leftnode = tree[currentnode,Yes]
    rightnode = tree[currentnode,No]
    leaves = c(findLeaves(tree,'leftnode',with=FALSE),findLeaves(tree,'rightnode',with=FALSE))
  }
  
  return (sort(leaves))
  
  
}
findPath = function(tree, currentnode, path = c()){
  
  #accepts a tree data table, and the node to reach
  #path is used in the recursive function - do not set this
  
  while(currentnode>0){
    path = c(path,currentnode)
    currentlabel = tree[Node==currentnode,ID]
    currentnode = c(tree[Yes==currentlabel,Node],tree[No==currentlabel,Node])
  }
  return (sort(c(path,0)))
  
}
getLeafBreakdown = function(tree,leaf,col_names){
  
  ####accepts a tree, the leaf id to breakdown and column names
  ####outputs a list of the impact of each variable + intercept
  
  impacts = as.list(rep(0,length(col_names)))
  names(impacts) = col_names
  
  path = findPath(tree,leaf)
  reduced_tree = tree[Node %in% path,.(Feature,uplift_weight)]
  
  impacts$intercept=reduced_tree[1,uplift_weight]
  reduced_tree[,uplift_weight:=shift(uplift_weight,type='lead')]
  
  tmp = reduced_tree[,.(sum=sum(uplift_weight)),by=Feature]
  tmp = tmp[-nrow(tmp)]
  impacts[tmp[,Feature]]=tmp[,sum]
  
  return (impacts)
}
getStatsForTrees = function(trees, nodes.train, type = "binary", base_score = 0.5){
  #Accepts data table of tree (the output of xgb.model.dt.tree)
  #Returns a list of tree, with the stats filled in
  
  tree_list = copy(trees)
  tree_list[,leaf := Feature == 'Leaf']
  tree_list[,H:=Cover]
  
  non.leaves = which(tree_list[,leaf]==F)
  
  
  # The default cover (H) seems to lose precision so this loop recalculates it for each node of each tree
  j = 0
  for (i in rev(non.leaves)){
    left = tree_list[i,Yes]
    right = tree_list[i,No]
    tree_list[i,H:=tree_list[ID==left,H] + tree_list[ID==right,H]]
    j=j+1
  }
  
  
  if (type == 'regression'){
    base_weight = base_score
  } else{
    base_weight = log(base_score / (1-base_score))
  }
  
  tree_list[leaf==T,weight:=base_weight + Quality]
  
  tree_list[,previous_weight:=base_weight]
  tree_list[1,previous_weight:=0]
  
  tree_list[leaf==T,G:=-weight*H]
  
  tree_list = split(tree_list,as.factor(tree_list$Tree))
  num_tree_list = length(tree_list)
  treenums =  as.character(0:(num_tree_list-1))
  t = 0
  for (tree in tree_list){
    t=t+1
    num_nodes = nrow(tree)
    non_leaf_rows = rev(which(tree[,leaf]==F))
    for (r in non_leaf_rows){
      left = tree[r,Yes]
      right = tree[r,No]
      leftG = tree[ID==left,G]
      rightG = tree[ID==right,G]
      
      tree[r,G:=leftG+rightG]
      w=tree[r,-G/H]
      
      tree[r,weight:=w]
      tree[ID==left,previous_weight:=w]
      tree[ID==right,previous_weight:=w]
    }
    
    tree[,uplift_weight:=weight-previous_weight]
  }
  
  return (tree_list)
}
getTreeBreakdown = function(tree, col_names){
  
  ####accepts a tree (data table), and column names
  ####outputs a data table, of the impact of each variable + intercept, for each leaf
  
  tree_breakdown <- vector("list", length(col_names)  + 2)
  names(tree_breakdown) = c(col_names,'intercept','leaf')
  
  leaves = tree[leaf==T, Node]
  
  for (leaf in leaves){
    
    leaf_breakdown = getLeafBreakdown(tree,leaf,col_names)
    leaf_breakdown$leaf = leaf
    tree_breakdown = rbindlist(append(list(tree_breakdown),list(leaf_breakdown)))
  }
  
  return (tree_breakdown)
}
showWaterfall = function(xgb.model, explainer, DMatrix, data.matrix, idx, threshold, type = "binary"){
  breakdown = explainPredictions(xgb.model, explainer, slice(DMatrix,as.integer(idx)))
  weight = rowSums(breakdown)
  pred = 1/(1+exp(-weight))
  breakdown_summary = as.matrix(breakdown)[1,]
  i = order(abs(breakdown_summary),decreasing=TRUE)
  breakdown_summary = breakdown_summary[i]
  intercept = breakdown_summary[names(breakdown_summary)=='intercept']
  breakdown_summary = breakdown_summary[names(breakdown_summary)!='intercept']
  data_for_label = data.matrix[idx,names(breakdown_summary)]  # Niels
  data_for_label = data_for_label[names(breakdown_summary)!='intercept'] # Niels
  i_other =which(abs(breakdown_summary)<threshold)
  other_impact = 0
  if (length(i_other > 0)){
    other_impact = sum(breakdown_summary[i_other])
    names(other_impact) = 'other'
    breakdown_summary = breakdown_summary[-i_other]
    data_for_label = data_for_label[-i_other]
  }
  if (abs(other_impact) > 0){
    breakdown_summary = c(intercept, breakdown_summary, other_impact)
    data_for_label = c("", data_for_label,"")
    labels = paste0(names(breakdown_summary)," = ", data_for_label)
    labels[1] = 'intercept'
    labels[length(labels)] = 'other'
  }else{
    breakdown_summary = c(intercept, breakdown_summary)
    data_for_label = c("", data_for_label)
    labels = paste0(names(breakdown_summary)," = ", data_for_label)
    labels[1] = 'intercept'
  }
  if (!is.null(getinfo(DMatrix,"label"))){
  }
  inverse_logit_trans <- scales::trans_new("inverse logit",
                                           transform = plogis,
                                           inverse = qlogis)
  
  inverse_logit_labels = function(x){return (1/(1+exp(-x)))}
  logit = function(x){return(log(x/(1-x)))}
  ybreaks<-logit(seq(2,98,2)/100)
  waterfalls::waterfall(values = breakdown_summary,
                        rect_text_labels = rep(" ", length(breakdown_summary)),
                        labels = labels,
                        total_rect_text = " ",
                        calc_total = TRUE)  +
    theme_void()
}
xgb_train <- xgb.DMatrix(data = as.matrix(train_df[,-1]), label = train_df[,1])
xgb_model <- xgb.train(data = xgb_train,
                       nrounds = 50, 
                       objective = "binary:logistic", tree_method = "exact", eval.metric = "auc")
explainer = buildExplainer(xgb_model,xgb_train, type="binary", base_score = 0.5, trees = NULL)
detach(package:dplyr, unload = TRUE)
idx <- which(rownames(train_df) == gene)
showWaterfall(xgb_model, explainer, xgb_train, train_df,  idx, type = "binary", threshold = 0.09)
