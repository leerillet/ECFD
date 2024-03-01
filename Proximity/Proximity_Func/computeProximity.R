#'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#'#calculate the proximity
# list1 = drug_target
# list2 = disease_target
#'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

computeProximity <- function(list1,list2,graph_info,metric='min',Dim=1,Perc_thr=5){
  
  graph <- graph_info$graph
  node <- graph_info$node
  
  from <- which(node %in% list1)
  to <- which(node %in% list2)
  
  distance_matrix <- distances(graph, v = V(graph)[from], to = V(graph)[to])
  
  if(metric == "min"){
    minimum <- computeMinimum(distance_matrix,dim = Dim,perc_thr = Perc_thr)
    proximity<- mean(minimum, na.rm = T)
    
  }else if (metric == "median"){
    median_value <- computeMedian(distance_matrix,dim = Dim,perc_thr = Perc_thr)
    proximity<- mean(median_value, na.rm = T)
    
  }else if(metric == "mean"){
    mean_value <- computeMean(distance_matrix,dim = Dim,perc_thr = Perc_thr)
    proximity <- mean(mean_value, na.rm = T)
    
  }else if(metric == "max"){
    maximum <- computeMax(distance_matrix,dim = Dim,perc_thr = Perc_thr)
    proximity <- mean(maximum, na.rm = T)
    
  }else if(metric == "mode"){
    mode_value <- computeMode(distance_matrix,dim = Dim,perc_thr = Perc_thr)
    proximity <- mean(mode_value, na.rm = T)
    
  }
  
  return(proximity)
  
}

