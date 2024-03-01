#'==============================================================================
#'@function: SAB Separation model for drug combination prediction
#'@param targetlist1 the one target list
#'@param targetlist2 the other target list
#'@param interactome the interaction network
#'@param drugtargetdt the drug-target dataframe
#' @param singlelist=T: the drug list is single, i.e. targetlist1=targetlist2
#'@export sabres the drug combination prediction result
#'==============================================================================

source("./Model/Proximity/Proximity_config.R")

#'==============================
#'load the ppi,disease,herb/drug data
#'==============================
inputfile = inputFiles()

######################################
# input files
Interactome = inputfile$interactome
diseasetargetDT = inputfile$disease_gene
drugtargetDT = inputfile$drug_target

#get the drug list
druglist = unique(drugtargetDT$Drug)
#get the combination dataframe
drugcbdt = as.data.frame(t(combn(druglist,2)),stringsAsFactors = F)
colnames(drugcbdt) = c("Drug1","Drug2")
drugcbdt = drugcbdt[!duplicated(drugcbdt),]
druglist1 = drugcbdt$Drug1
druglist2 = drugcbdt$Drug2 

# targetlist1 = druglist
# targetlist2 = druglist
# interactome = Interactome
# drugtargetdt = drugtargetDT
# singlelist=T

SAB = function(targetlist1,targetlist2,interactome,drugtargetdt,singlelist=T){
  #build the igraph object
  ppinetwork <- graph_from_data_frame(interactome, directed = FALSE)
  ppinetwork <- decompose.graph(ppinetwork, mode = "weak")[[1]]
  if(isTRUE(singlelist)){
    #for single list, e.g. single drug list
    uniquelist = unique(c(targetlist1,targetlist2))
    sabres = as.data.frame(t(combn(uniquelist,2)),stringsAsFactors = F)
    colnames(sabres) = c("Item1","Item2")
  }else{
    sabres = expand.grid("Item1"=targetlist1,"Item2"=targetlist2,stringsAsFactors = F)
  }
  sabres$Daa = rep("",nrow(sabres))
  sabres$Dbb = rep("",nrow(sabres))
  sabres$Dab = rep("",nrow(sabres))
  sabres$Sab = rep("",nrow(sabres))
  
  for(i in 1:nrow(sabres)){
    print(i)
    drug1 = sabres[i,1]
    drug2 = sabres[i,2]
    drugtargetlist1 = drugtargetdt[which(drugtargetdt$Drug==drug1),"Gene"]
    drugtargetlist2 = drugtargetdt[which(drugtargetdt$Drug==drug2),"Gene"]
    if(length(drugtargetlist1)<=1|length(drugtargetlist2)<=1){
      sabres[i,3] = NA
      sabres[i,4] = NA
      sabres[i,5] = NA
    }else{
      
      #Daa
      drugtargetdt1 = as.data.frame(t(combn(drugtargetlist1,2)),stringsAsFactors = F)
      colnames(drugtargetdt1) = c("From","To")
      shortdist1 = c()
      for(j in 1:nrow(drugtargetdt1)){
        startnode = drugtargetdt1[j,1]
        endnode = drugtargetdt1[j,2]
        shortdist1[j] = distances(v = startnode,to = endnode,graph = ppinetwork)
      }
      sabres[i,3] = mean(shortdist1,na.rm = T)

      #Dbb
      drugtargetdt2 = as.data.frame(t(combn(drugtargetlist2,2)),stringsAsFactors = F)
      colnames(drugtargetdt2) = c("From","To")
      shortdist2 = c()
      for(k in 1:nrow(drugtargetdt2)){
        startnode = drugtargetdt2[k,1]
        endnode = drugtargetdt2[k,2]
        shortdist2[k] = distances(v = startnode,to = endnode,graph = ppinetwork)
      }
      sabres[i,4] = mean(shortdist2,na.rm = T)
      
      #Dab
      drugtargetdt3 = expand.grid("From"=drugtargetlist1,"To"=drugtargetlist2,stringsAsFactors = F)
      # colnames(drugtargetdt3) = c("From","To")
      shortdist3 = c()
      for(m in 1:nrow(drugtargetdt3)){
        startnode = drugtargetdt3[m,1]
        endnode = drugtargetdt3[m,2]
        shortdist3[m] = distances(v = startnode,to = endnode,graph = ppinetwork)
      }
      sabres[i,5] = mean(shortdist3,na.rm = T)
    }
  }
  sabres = na.omit(sabres)
  sabres$Daa = round(as.numeric(sabres$Daa),5)
  sabres$Dbb = round(as.numeric(sabres$Dbb),5)
  sabres$Dab = round(as.numeric(sabres$Dab),5)
  sabres$Sab = round(sabres$Dab-0.5*(sabres$Daa+sabres$Dbb),5)
  return(sabres)
}
# distance_matrix <- igraph::distances(ppinetwork, v = drugtargetlist1, to = drugtargetlist1)
# # If there are Inf values (should not occur with our networks) substitute them to NAs to be able to use sum and min in the correct way
# distance_matrix[distance_matrix == Inf] <- NA
# # All diagonal entries are set to NA (Distance betwwen one gene and itself should be set to Inf)
# for(i in 1:dim(distance_matrix)[1]){
#   distance_matrix[i,i] <- NA
# }
# # else: delete the rows and cols with only Na values, as the regarding genes can't be reached from the others
# distance_matrix <- as.matrix(distance_matrix[,colSums(is.na(distance_matrix))!=length(drugtargetlist1)])
# distance_matrix <- as.matrix(distance_matrix[rowSums(is.na(distance_matrix))!=length(drugtargetlist1),])
# closest_within_dist <- apply(distance_matrix, 2, min, na.rm = T)
# closest_within_dist <- sum(closest_within_dist, na.rm = T)/length(drugtargetlist1)