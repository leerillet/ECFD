computeSimilarity <- function(drug_disease_net) {
  
  max_proximity <- max(drug_disease_net$Proximity, na.rm = T)
  
  similarity <-  round(( max_proximity - drug_disease_net$Proximity ) / max_proximity,5)
  
  drug_disease_net <- data.frame(drug_disease_net, Similarity = similarity)
  
  return(drug_disease_net)
  
}