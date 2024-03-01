#'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~·
#'Function compute DegreeDistribution
#'@usage 
#'nodelist=disease_target
# graph_info, igraph object for interactome
#'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

computeDegreeDistribution <- function(nodelist,graph_info){
  
  graph <- graph_info$graph
  node <- graph_info$node
  
  from <- which(node %in% nodelist)
  d <- degree(graph, v = V(graph)[from])
  
  t <- table(d)
  
  degree_sorted <- as.numeric(names(t))
  
  freq <- as.numeric(t)
  
  df <- data.frame(degree=degree_sorted,frequency=freq)

  return(df)

}