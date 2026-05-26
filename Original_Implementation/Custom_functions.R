# 
#   Copyright (C) 2024  Niels Benjamin Paul
# 
#   This program is free software: you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation, either version 3 of the License, or
#   (at your option) any later version.
# 
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
# 
#   You should have received a copy of the GNU General Public License
#   along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# The functions buildExplainer, explainPredictions, getStatsForTrees,
# buildExplainerFromTreeList, getTreeBreakdown, getLeafBreakdown, findPath,
# findLeaves and showWaterfall are copied from
# https://github.com/AppliedDataSciencePartners/xgboostExplainer/tree/master,
# where they are published under GNU General Public License as published by
# the Free Software Foundation, version 3 by 

################################################################################
#
# Script that comprises all function definitions and is imported via source() by
# other scripts
#
################################################################################

# Analysis: Cell_line_performance.R
visualize_RNA_dist <- function(cl,RNAseq_mat){
  RNA <- as.numeric(RNAseq_mat[,cl])
  RNA_log <- log(RNA + 0.01)
  median_RNA <- median(RNA_log)
  RNA_binarized <- (RNA > median_RNA)+0
  RNA_df <- data.frame(RNA_log = RNA_log, RNA_binarized = RNA_binarized)
  ggplot(RNA_df, aes(x=RNA_log)) +
    geom_histogram(bins = 50) +
    annotate(geom = "segment", x = median_RNA, xend = median_RNA, y = 0, yend = 500,
             color = "red", size = 1)
}

# Visualization/XGB_explainer.R
buildExplainer = function(xgb.model, trainingData, type = "binary", base_score = 0.5, trees_idx = NULL){
  
  #col_names = attr(trainingData, ".Dimnames")[[2]]
  col_names = colnames(trainingData) # introduced by Niels. the above does not work for this version of xgboost
  cat('\nCreating the trees of the xgboost model...')
  trees = xgb.model.dt.tree(col_names, model = xgb.model, trees = trees_idx)
  cat('\nGetting the leaf nodes for the training set observations...')
  nodes.train = predict(xgb.model,trainingData,predleaf =TRUE)
  
  cat('\nBuilding the Explainer...')
  cat('\nSTEP 1 of 2')
  tree_list = getStatsForTrees(trees, nodes.train, type = type, base_score = base_score)
  cat('\n\nSTEP 2 of 2')
  explainer = buildExplainerFromTreeList(tree_list,col_names)
  
  cat('\n\nDONE!\n\n')
  
  return (explainer)
}

# Analysis/Interpretation_cell_line.R
explainPredictions = function(xgb.model, explainer ,data){
  #Accepts data table of the breakdown for each leaf of each tree and the node matrix
  #Returns the breakdown for each prediction as a data table
  
  nodes = predict(xgb.model,data,predleaf =TRUE)
  
  colnames = names(explainer)[1:(ncol(explainer)-2)]
  
  preds_breakdown = data.table(matrix(0,nrow = nrow(nodes), ncol = length(colnames)))
  setnames(preds_breakdown, colnames)
  
  num_trees = ncol(nodes)
  
  cat('\n\nExtracting the breakdown of each prediction...\n')
  pb <- txtProgressBar(style=3)
  for (x in 1:num_trees){
    nodes_for_tree = nodes[,x]
    tree_breakdown = explainer[tree==x-1]
    
    preds_breakdown_for_tree = tree_breakdown[match(nodes_for_tree, tree_breakdown$leaf),]
    preds_breakdown = preds_breakdown + preds_breakdown_for_tree[,colnames,with=FALSE]
    
    setTxtProgressBar(pb, x / num_trees)
  }
  
  cat('\n\nDONE!\n')
  
  return (preds_breakdown)
  
}

# Anlysis/Interpretation_cell_line.R
getStatsForTrees = function(trees, nodes.train, type = "binary", base_score = 0.5){
  #Accepts data table of tree (the output of xgb.model.dt.tree)
  #Returns a list of tree, with the stats filled in
  
  tree_list = copy(trees)
  tree_list[,leaf := Feature == 'Leaf']
  tree_list[,H:=Cover]
  
  non.leaves = which(tree_list[,leaf]==F)
  
  
  # The default cover (H) seems to lose precision so this loop recalculates it for each node of each tree
  cat('\n\nRecalculating the cover for each non-leaf... \n')
  pb <- txtProgressBar(style=3)
  j = 0
  for (i in rev(non.leaves)){
    left = tree_list[i,Yes]
    right = tree_list[i,No]
    tree_list[i,H:=tree_list[ID==left,H] + tree_list[ID==right,H]]
    j=j+1
    setTxtProgressBar(pb, j / length(non.leaves))
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
  cat('\n\nFinding the stats for the xgboost trees...\n')
  pb <- txtProgressBar(style=3)
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
    setTxtProgressBar(pb, t / num_tree_list)
  }
  
  return (tree_list)
}

# Analysis/Interpretation_cell_line.R
buildExplainerFromTreeList = function(tree_list,col_names){
  ####accepts a list of trees and column names
  ####outputs a data table, of the impact of each variable + intercept, for each leaf
  
  tree_list_breakdown <- vector("list", length(col_names)  + 3)
  names(tree_list_breakdown) = c(col_names,'intercept', 'leaf','tree')
  
  num_trees = length(tree_list)
  
  cat('\n\nGetting breakdown for each leaf of each tree...\n')
  pb <- txtProgressBar(style=3)
  
  for (x in 1:num_trees){
    tree = tree_list[[x]]
    tree_breakdown = getTreeBreakdown(tree, col_names)
    tree_breakdown$tree = x - 1
    tree_list_breakdown = rbindlist(append(list(tree_list_breakdown),list(tree_breakdown)))
    setTxtProgressBar(pb, x / num_trees)
  }
  
  return (tree_list_breakdown)
}

# Analysis/Interpretation_cell_line.R
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

# Analysis/Interpretation_cell-line.R
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

#Analysis/Interpretation_cell_line.R
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

# Predictions/PatternChrome_alternative_cutoff.R
get_alternative_cutoff <- function(RNA){
  RNA_log <- log(RNA + 0.01)
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
  RNA <- (RNA_log > alternative_cutoff) + 0
  names(RNA) <- rownames(H3K4me3)
  return(RNA)
}

# Visualization/XGB_explainer.R
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

# Visualization/XGB_explainer.R
showWaterfall = function(xgb.model, explainer, DMatrix, data.matrix, idx, type = "binary", threshold = 0.0001, limits = c(NA, NA)){
  
  
  breakdown = explainPredictions(xgb.model, explainer, xgboost::slice(DMatrix,as.integer(idx)))
  
  weight = rowSums(breakdown)
  if (type == 'regression'){
    pred = weight
  }else{
    pred = 1/(1+exp(-weight))
  }
  
  
  breakdown_summary = as.matrix(breakdown)[1,]
  
  #data_for_label = data.matrix[idx,]
  
  i = order(abs(breakdown_summary),decreasing=TRUE)
  
  breakdown_summary = breakdown_summary[i]
  #data_for_label = data_for_label[i] 
  
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
    cat("\nActual: ", getinfo(slice(DMatrix,as.integer(idx)),"label"))
  }
  cat("\nPrediction: ", pred)
  cat("\nWeight: ", weight)
  cat("\nBreakdown")
  cat('\n')
  print(breakdown_summary)
  
  if (type == 'regression'){
    
    waterfalls::waterfall(values = breakdown_summary,
                          rect_text_labels = round(breakdown_summary, 2),
                          labels = labels,
                          total_rect_text = round(weight, 2),
                          calc_total = TRUE,
                          total_axis_text = "Prediction") +
      theme(axis.text.x = element_text(angle = 45, hjust = 1))
  }else{
    
    inverse_logit_trans <- scales::trans_new("inverse logit",
                                             transform = plogis,
                                             inverse = qlogis)
    
    inverse_logit_labels = function(x){return (1/(1+exp(-x)))}
    logit = function(x){return(log(x/(1-x)))}
    
    ybreaks<-logit(seq(2,98,2)/100)
    
    waterfalls::waterfall(values = breakdown_summary,
                          rect_text_labels = round(breakdown_summary, 2),
                          labels = labels,
                          total_rect_text = round(weight, 2),
                          calc_total = TRUE,
                          total_axis_text = "Prediction")  +
      scale_y_continuous(labels = inverse_logit_labels,
                         breaks = ybreaks, limits = limits) +
      theme(axis.text.x = element_text(angle = 90, hjust = 1, size=Paths$font_size),
            text = element_text(size=Paths$font_size))
    
  }
}



