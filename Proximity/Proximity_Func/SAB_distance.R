########################################################################################################
#
# Function_Distances.R
#
# Function used to calculate the distance between a set of disease genes and a set of drug targets
# 
# INPUT: Network: network to consider as background interaction network
#        DiseaseGenes: vector with the gene Symbols identified as disease genes (filtered to the network)
#        DrugTargets: vector of the targets of a selected drug (filtered to the network)
# OUTPUT: Distance between the two sets of disease genes and drug targets considering different measurements       
#
# 
#
########################################################################################################

### libraries 
require(igraph)

### date
date <- gsub("-","",Sys.Date())

#===================================================================================================================

### Minimal distance function, overall minimum of the distances between each pair of disease gene and drug target
minimal_distance <- function(Network, DiseaseGenes, DrugTargets){
  
  # Get distance matrix with all distances between each pait for disease gene and drug target
  distance_matrix <- igraph::distances(Network, v = DiseaseGenes, to = DrugTargets)
  # If there are Inf values (should not occur with our networks) substitute them to NAs to be able to use sum and min in the correct way
  distance_matrix[distance_matrix == Inf] <- NA
  
  # if all distances are Na values and one can't connect disease genes and drug targets, return NA as minimal distance
  if(sum(is.na(distance_matrix)) == length(DiseaseGenes)*length(DrugTargets)){
    min_dist <- NA
  }else{
    # else calculate the minimal distance
    min_dist <- min(distance_matrix, na.rm = T)
  }
  return(min_dist)
}

#===================================================================================================================

### closest distance: average shortest path length between drug targets and the nearest disease protein
closest_distance <- function(Network, DiseaseGenes, DrugTargets){
  
  # Get distance matrix with all distances between each pait for disease gene and drug target
  distance_matrix <- igraph::distances(Network, v = DrugTargets, to = DiseaseGenes, mode = "out")
  # If there are Inf values (should not occur with our networks) substitute them to NAs to be able to use sum and min in the correct way
  distance_matrix[distance_matrix == Inf] <- NA
  
  # if all distances are Na values and one can't connect disease genes and drug targets, return NA as distance
  if(sum(is.na(distance_matrix)) == length(DiseaseGenes)*length(DrugTargets)){
    closest_dist <- NA
  }else{
   # else: delete the disease genes from the distance matrix that can't be reached from any drug targe
    distance_matrix <- as.matrix(distance_matrix[,colSums(is.na(distance_matrix))!=length(DrugTargets)])
    distance_matrix <- as.matrix(distance_matrix[rowSums(is.na(distance_matrix))!=length(colnames(distance_matrix)),])
    # select for each drug target (for each column) the minimal distance to a disease gene
    closest_dist <- apply(distance_matrix, 1, min, na.rm = T)
    # take the average of these distances
    closest_dist <- sum(closest_dist, na.rm = T)/length(rownames(distance_matrix))
  }
  return(closest_dist)
}

#===================================================================================================================

### weighted closest distance: average shortest path length between drug targets and the nearest disease protein

weighted_closest_distance <- function(Network, DiseaseGenes, DrugTargets){
  
  # Get distance matrix with all distances between each pait for disease gene and drug target
  distance_matrix <- igraph::distances(Network, v = DiseaseGenes, to = DrugTargets)
  # If there are Inf values (should not occur with our networks) substitute them to NAs to be able to use sum and min in the correct way
  distance_matrix[distance_matrix == Inf] <- NA
  
  # if all distances are Na values and one can't connect disease genes and drug targets, return NA as distance
  if(sum(is.na(distance_matrix)) == length(DiseaseGenes)*length(DrugTargets)){
    closest_dist <- NA
  }else{
    # else: delete the drug targets from the distance matrix thsat can't be reached from any disease gene
    distance_matrix <- as.matrix(distance_matrix[,colSums(is.na(distance_matrix))!=length(DiseaseGenes)])
    # select for each drug target (for each column) the minimal distance to a disease gene
    closest_dist <- apply(distance_matrix, 2, min, na.rm = T)
    # set weights for the targets
    weights <- vector("numeric", length = length(DrugTargets))
    weights <- ifelse(DrugTargets %in% DiseaseGenes, 0.6, 0.4)
    # take the weighted average of the minimal distances
    closest_dist <- sum(closest_dist * weights, na.rm = T)/(length(DrugTargets)*sum(weights))
  }
  return(closest_dist)
}

#===================================================================================================================

### shortest distance: average shortest path length between all targets of a drug and the disease proteins
shortest_distance <- function(Network, DiseaseGenes, DrugTargets){

  # Get distance matrix with all distances between each pait for disease gene and drug target
  distance_matrix <- igraph::distances(Network, v = DiseaseGenes, to = DrugTargets)
  # If there are Inf values (should not occur with our networks) substitute them to NAs to be able to use sum and min in the correct way
  distance_matrix[distance_matrix == Inf] <- NA
  
  # if all distances are Na values and one can't connect disease genes and drug targets, return NA as distance
  if(sum(is.na(distance_matrix)) == length(DiseaseGenes)*length(DrugTargets)){
    shortest_dist <- NA
  }else{
    # else: delete the drug targets from the distance matrix thsat can't be reached from any disease gene
    distance_matrix <- as.matrix(distance_matrix[,colSums(is.na(distance_matrix))!=length(DiseaseGenes)])
    # the shortest distance is then defined as the average value of all distances   
    shortest_dist<- sum(distance_matrix)/(length(DrugTargets)*length(DiseaseGenes))
  }
  return(shortest_dist)
}

#===================================================================================================================

### kernel distance
kernel_distance <- function(Network, DiseaseGenes, DrugTargets){
  
  # Get distance matrix with all distances between each pait for disease gene and drug target
  distance_matrix <- igraph::distances(Network, v = DiseaseGenes, to = DrugTargets)
  # If there are Inf values (should not occur with our networks) substitute them to NAs to be able to use sum and min in the correct way
  distance_matrix[distance_matrix == Inf] <- NA
  
  # if all distances are Na values and one can't connect disease genes and drug targets, return NA as distance
  if(sum(is.na(distance_matrix)) == length(DiseaseGenes)*length(DrugTargets)){
    kernel_dist <- NA
  }else{
    # else: delete the drug targets from the distance matrix thsat can't be reached from any disease gene
    distance_matrix <- as.matrix(distance_matrix[,colSums(is.na(distance_matrix))!=length(DiseaseGenes)])
    # kernel distance uses a transformation of the distances to give more weight to close distances 
    kernel_dist <- exp(-(distance_matrix +1))/length(DiseaseGenes)
    kernel_dist <- log(apply(kernel_dist, 2, sum, na.rm = T))
    kernel_dist <- -sum(kernel_dist)/length(DrugTargets)
  }
  return(kernel_dist)
}

#===================================================================================================================

### centre distance
centre_distance <- function(Network, DiseaseGenes, DrugTargets){
  
  # Get distance matrix with all distances between each pait for disease gene and drug target
  distance_matrix <- igraph::distances(Network, v = DiseaseGenes, to = DrugTargets)
  # If there are Inf values (should not occur with our networks) substitute them to NAs to be able to use sum and min in the correct way
  distance_matrix[distance_matrix == Inf] <- NA
  # if all distances are Na values and one can't connect disease genes and drug targets, return NA as distance
  if(sum(is.na(distance_matrix)) == length(DiseaseGenes)*length(DrugTargets)){
    centre_dist <- NA
  }else{
    # else: calculate the central point in the set of disease genes: the gene with the minimal distance to all others
    centre_disease <- igraph::distances(Network, v = DiseaseGenes, to = DiseaseGenes)
    centre_disease <- apply(centre_disease, 1, sum, na.rm = T)
    argmin <- which(centre_disease  == min(centre_disease) )
    
    # I have to distinguish between one or more minimal values for the central point
    if(length(argmin) == 1){
      # if we have only one, calculate the average distance between this gene and the drug targets
      centre_dist <- sum(distance_matrix[names(argmin),], na.rm = T)/length(DrugTargets)
    }else{
      # if we have more calculate the average distance between each of these with the drug targets 
      centre_dist <- NULL
      for(i in 1:length(argmin)){
        centre_dist[i] <- sum(distance_matrix[names(argmin)[i],], na.rm = T)/length(DrugTargets)
      }
      # and select the average of these values as final distance
      centre_dist <- mean(centre_dist)
    }
  }
  return(centre_dist)
  
}

#===================================================================================================================

### helpfunction - closest within distance between one set of genes
closest_within_distance <- function(Network, Genes){
  
  # Get distance matrix with all distances between each pait for disease gene and drug target
  distance_matrix <- igraph::distances(Network, v = Genes, to = Genes)
  # If there are Inf values (should not occur with our networks) substitute them to NAs to be able to use sum and min in the correct way
  distance_matrix[distance_matrix == Inf] <- NA
  # All diagonal entries are set to NA (Distance betwwen one gene and itself should be set to Inf)
  for(i in 1:dim(distance_matrix)[1]){
    distance_matrix[i,i] <- NA
  }
  
  # else: delete the rows and cols with only Na values, as the regarding genes can't be reached from the others
  distance_matrix <- as.matrix(distance_matrix[,colSums(is.na(distance_matrix))!=length(Genes)])
  distance_matrix <- as.matrix(distance_matrix[rowSums(is.na(distance_matrix))!=length(Genes),])
  
  # if all distances are Na values and one can't connect disease genes and drug targets, return NA as distance
  if(sum(is.na(distance_matrix)) == length(Genes)*length(Genes)){
    closest_within_dist <- NA
  }else{
    # else: consider the average minimal distance between all genes as final distance
    closest_within_dist <- apply(distance_matrix, 2, min, na.rm = T)
    closest_within_dist <- sum(closest_within_dist, na.rm = T)/length(Genes)
  }
  return(closest_within_dist)
}

#===================================================================================================================

### Separation
separation_distance <- function(Network, DiseaseGenes, DrugTargets){
  
  # If one of the gene sets has only one gene the measurement doesn't work (the within closeness is Inf), return NA
  if(length(DiseaseGenes) == 1 | length(DrugTargets) == 1){
    separation_dist = NA
  }else{
    # calculate the within distance for the two sets
    closest_disease <- closest_within_distance(Network, DiseaseGenes)
    closest_drugtarget <- closest_within_distance(Network, DrugTargets)
    # calculate the closest distance between Disease genes and Drug targets as well as between Drug Targets and Disease genes
    closest_disease_drugtarget <- closest_distance(Network, DiseaseGenes, DrugTargets)
    closest_drugtarget_disease <- closest_distance(Network, DrugTargets, DiseaseGenes)
    # calculate dispersion and the final distance
    dispersion <- (length(DrugTargets)*closest_disease_drugtarget + length(DiseaseGenes)*closest_drugtarget_disease)/(length(DrugTargets) + length(DiseaseGenes))
    separation_dist <-  dispersion - ((closest_disease + closest_drugtarget)/2) 
  }
  return(separation_dist)
}

#===================================================================================================================

### Final function used to get all distances for a network, disease genes and drug targets

# calculate_distances <- function(Network, DiseaseGenes, DrugTargets){
#   
#   minimal <- minimal_distance(Network, DiseaseGenes, DrugTargets)
#   shortest <- shortest_distance(Network, DiseaseGenes, DrugTargets)
#   closest <- closest_distance(Network, DiseaseGenes, DrugTargets)
#   weighted_closest <- weighted_closest_distance(Network, DiseaseGenes, DrugTargets)
#   kernel <- kernel_distance(Network, DiseaseGenes, DrugTargets)
#   centre <- centre_distance(Network, DiseaseGenes, DrugTargets)
#   separation <- separation_distance(Network, DiseaseGenes, DrugTargets)
#   
#   result <- cbind("minimal" = minimal,
#                  "shortest" = shortest,
#                  "closest" = closest,
#                  "weighted_closest" = weighted_closest,
#                  "kernel" = kernel,
#                  "centre" = centre,
#                  "separation" = separation)
#   return(result)
# }

#===================================================================================================================

### Final function used to get all distances for a network, disease genes and drug targets - faster version, 
### because it uses the distance calculation just once


calculate_distances <- function(Network, DiseaseGenes, DrugTargets, weight){
  
  # Get distance matrix with all distances between each pair for disease gene and drug target
  distance_matrix <- igraph::distances(Network, v = DiseaseGenes, to = DrugTargets)
  # If there are Inf values (should not occur with our networks) substitute them to NAs to be able to use sum and min in the correct way
  distance_matrix[distance_matrix == Inf] <- NA
  
  # if all distances are Na values and one can't connect disease genes and drug targets, return NA as minimal distance
  if(sum(is.na(distance_matrix)) == length(DiseaseGenes)*length(DrugTargets)){
    min_dist <- NA
    closest_dist <- NA
    weighted_closest_dist <- NA
    shortest_dist <- NA
    kernel_dist <- NA
    centre_dist <- NA
  }else{
    
    ### minimal distance: minimum of all distances
    min_dist <- min(distance_matrix, na.rm = T)
    
    ### for the other distances: delete the drug targets from the distance matrix that can't be reached from any disease gene
    distance_matrix <- as.matrix(distance_matrix[,colSums(is.na(distance_matrix))!=length(DiseaseGenes)])
    
    ### closest and weighted closest distance
    # select for each drug target (for each column) the minimal distance to a disease gene
    closest_dist <- apply(distance_matrix, 2, min, na.rm = T)
    # set weights for the drug targets: if they are also in the pathway weight, otherwise 1-weight
    weights <- vector("numeric", length = length(DrugTargets))
    weights <- ifelse(DrugTargets %in% DiseaseGenes, weight, (1-weight))
    # take the weighted average of the minimal distances
    weighted_closest_dist <- sum(closest_dist * weights, na.rm = T)/(length(DrugTargets)*sum(weights))
    # the closest distance is the average of the minimal distances (not weighted)
    closest_dist <- sum(closest_dist, na.rm = T)/length(DrugTargets)
    
    ### shortest distance: average value of all distances 
    shortest_dist<- sum(distance_matrix)/(length(DrugTargets)*length(DiseaseGenes))
    
    ### kernel distance: uses a transformation of the distances to give more weight to close distances 
    kernel_dist <- exp(-(distance_matrix +1))/length(DiseaseGenes)
    kernel_dist <- log(apply(kernel_dist, 2, sum, na.rm = T))
    kernel_dist <- -sum(kernel_dist)/length(DrugTargets)
    
    ### centre distance: calculate the central point in the set of disease genes (the gene with the minimal distance to all others)
    centre_disease <- igraph::distances(Network, v = DiseaseGenes, to = DiseaseGenes)
    centre_disease <- apply(centre_disease, 1, sum, na.rm = T)
    argmin <- which(centre_disease  == min(centre_disease) )
    # I have to distinguish between one or more minimal values for the central point
    if(length(argmin) == 1){
      # if we have only one, calculate the average distance between this gene and the drug targets
      centre_dist <- sum(distance_matrix[names(argmin),], na.rm = T)/length(DrugTargets)
    }else{
      # if we have more calculate the average distance between each of these with the drug targets 
      centre_dist <- NULL
      for(i in 1:length(argmin)){
        centre_dist[i] <- sum(distance_matrix[names(argmin)[i],], na.rm = T)/length(DrugTargets)
      }
      # and select the average of these values as final distance
      centre_dist <- mean(centre_dist)
    }
  }
  
  ### separation distance: external function because it uses different distance matrices
  separation_dist <- separation_distance(Network, DiseaseGenes, DrugTargets)
  
  ### get the vector of final results
  result <- cbind("minimal" = min_dist,
                  "shortest" = shortest_dist,
                  "closest" = closest_dist,
                  "weighted_closest" = weighted_closest_dist,
                  "kernel" = kernel_dist,
                  "centre" = centre_dist,
                  "separation" = separation_dist)
  return(result)
  
  
}



