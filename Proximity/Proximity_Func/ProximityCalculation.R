#'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#'@description different proximity calculation methods 
#'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#========
#max mode
#========
computeMax <- function(distance_matrix,dim=1,perc_thr=5){
  
  # dim = 1 -> row mean (ie, from DrugTarget to DiseaseGene), 
  # dim = 2 -> col mean (ie, from DiseaseGene to DrugTarget)
  maximum <- apply(distance_matrix,dim,function(x){
    
    ind <- which(is.infinite(x))
    
    if(length(ind) != length(x) ) x <- x[-ind]
    
    maximum <- max(x,na.rm = T)
  })
  
  count <- length(maximum[is.infinite(maximum)])
  
  perc <- ( count / length(maximum) ) *100
  
  if( perc > perc_thr & !is.na(perc) ){
    
    maximum <- maximum
    
  }else{
    
    maximum <- maximum[is.finite(maximum)]
    
  }
  
}

#========
#Mean mode
#========
computeMean <- function(distance_matrix,dim=1,perc_thr=5){
  
  # dim = 1 -> row mean (ie, from DrugTarget to DiseaseGene), 
  # dim = 2 -> col mean (ie, from DiseaseGene to DrugTarget)
  
  mean_value <- apply(distance_matrix,dim,function(x){
    
    ind <- which(is.infinite(x))
    
    if(length(ind) != length(x) ) x <- x[-ind]
    
    mean_value <- mean(x,na.rm = T)
  })
  
  count <- length(mean_value[is.infinite(mean_value)])
  
  perc <- ( count / length(mean_value) ) *100
  
  if( perc > perc_thr & !is.na(perc) ){
    
    mean_value <- mean_value
    
  }else{
    
    mean_value <- mean_value[is.finite(mean_value)]
    
  }
  
}

#========
#Median mode
#========
computeMedian <- function(distance_matrix,dim=1,perc_thr=5){
  
  # dim = 1 -> row mean (ie, from DrugTarget to DiseaseGene), 
  # dim = 2 -> col mean (ie, from DiseaseGene to DrugTarget)
  
  median_value <- apply(distance_matrix,dim,median,na.rm = T)
  
  count <- length(median_value[is.infinite(median_value)])
  
  perc <- ( count / length(median_value) ) *100

  if( perc > perc_thr & !is.na(perc)  ){
    
    median_value <- median_value
    
  }else{
    
    median_value <- median_value[is.finite(median_value)]
    
  }
  
}

#=============
#minimum mode
#=============
computeMinimum <- function(distance_matrix,dim=1,perc_thr=5){
  
  # dim = 1 -> row mean (ie, from DrugTarget to DiseaseGene), 
  # dim = 2 -> col mean (ie, from DiseaseGene to DrugTarget)
  
  minimum <- apply(distance_matrix,dim,min,na.rm = T)
  
  count <- length(minimum[is.infinite(minimum)])
  
  perc <- ( count / length(minimum) ) *100
  
  if( perc > perc_thr & !is.na(perc)  ){
    
    minimum <- minimum
    
  }else{
    
    minimum <- minimum[is.finite(minimum)]
    
  }
  
}

#=============
#mode mode
#=============
computeMode <- function(distance_matrix,dim=1,perc_thr=5){
  
  # dim = 1 -> row mean (ie, from DrugTarget to DiseaseGene)
  # dim = 2 -> col mean (ie, from DiseaseGene to DrugTarget)
  
  mode_value <- apply(distance_matrix,dim,function(x){
    
    mode <- Mode(x,na.rm = T)
    mode <- min(mode)
    
  })
  
  count <- length(mode_value[is.infinite(mode_value)])
  
  perc <- ( count / length(mode_value) ) *100
  
  if( perc > perc_thr & !is.na(perc) ){
    
    mode_value <- mode_value
    
  }else{
    
    mode_value <- mode_value[is.finite(mode_value)]
    
  }
  
}