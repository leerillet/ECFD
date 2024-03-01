#'==============================================================================
#'@function: SAB Separation model for drug combination prediction
#'@param targetlist1 the one target list
#'@param targetlist2 the other target list
#'@param interactome the interaction network
#'@param drugtargetdt the drug-target dataframe
#' @param singlelist=T: the drug list is single, i.e. targetlist1=targetlist2
#'@export sabres the drug combination prediction result
#'@references some codes refer the NAPDE_Misselbeck_2019
#'==============================================================================

source("./Model/Proximity/Proximity_config.R")
source("./Model/Proximity/Proximity_Func/SAB_distance.R")
#'==============================
#'load the ppi,disease,herb/drug data
#'==============================
# inputfile = inputFiles()

######################################
# input files
# Interactome = inputfile$interactome
# diseasetargetDT = inputfile$disease_gene
# drugtargetDT = inputfile$drug_target
# targetlist1 = druglist
# targetlist2 = druglist
# interactome = Interactome
# drugtargetdt = drugtargetDT
# singlelist=T
# mode="min"
# dim=1
# perc_thr=0.05
######################################

SAB = function(targetlist1,targetlist2,interactome,drugtargetdt,
               singlelist=T,mode="min",dim=1,perc_thr=0.05){
  #build the igraph object
  ppinetwork = graph_from_data_frame(interactome, directed = FALSE)
  ppinetwork = decompose.graph(ppinetwork, mode = "weak")[[1]]
  graph_info = getGraph(interactome)
  
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
    try({
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
      Daa = closest_within_distance(ppinetwork,drugtargetlist1)
      sabres[i,3] = Daa

      #Dbb
      Dbb = closest_within_distance(ppinetwork,drugtargetlist2)
      sabres[i,4] = Dbb
      
      #Dab
      # calculate the closest distance between two list
      closest_atob = computeProximity(drugtargetlist1,
                                       drugtargetlist2,
                                       graph_info,
                                       metric = mode,
                                       Dim = dim,
                                       Perc_thr = perc_thr)          
      closest_btoa = computeProximity(drugtargetlist2,
                                      drugtargetlist1,
                                      graph_info,
                                      metric = mode,
                                      Dim = dim,
                                      Perc_thr = perc_thr)   
      # calculate dispersion and the final distance
      dispersion = (length(drugtargetlist2)*closest_atob + length(drugtargetlist1)*closest_btoa)/(length(drugtargetlist1) + length(drugtargetlist2))
      separation_dist = dispersion - ((Daa + Dbb)/2) 
      sabres[i,5] = separation_dist
      
    }
    })
  }
  sabres = na.omit(sabres)
  sabres$Daa = round(as.numeric(sabres$Daa),5)
  sabres$Dbb = round(as.numeric(sabres$Dbb),5)
  sabres$Dab = round(as.numeric(sabres$Dab),5)
  sabres$Sab = round(sabres$Dab-0.5*(sabres$Daa+sabres$Dbb),5)
  return(sabres)
}