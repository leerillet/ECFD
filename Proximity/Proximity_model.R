#'==============================================================================
#'@function: ProximityModel: THE PROXIMITY MODEL
#'@input 1)interactome:ppi data:SYMBOL1, SYMBOL2
#'@input 2)diseasedt:disease data: Disease, Gene
#'@input 3)drugdt: drug data:Drug, Gene
#'@output druginfo:Drug with proximity value
#'
#'@param1: mode:max, mean, median, min, mode
#'@param2: dim, defautl = 1 
#'dim = 1 -> row mean (ie, from DrugTarget to DiseaseGene), 
#' dim = 2 -> col mean (ie, from DiseaseGene to DrugTarget)
#'@param3: iter: the random times, default=100
#'@param4:Perc_thr, threshold for proximity calculation, default=5
#'@test:
# interactome = Interactome
# diseasedt = diseasetargetDT
# drugdt = drugtargetDT
# mode = "Median"
# dim = 1
# iter =5
# perc_thr = 5
#'==============================================================================

ProximityModel = function(interactome,diseasedt,drugdt,mode,dim,iter,perc_thr){
  #'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #' build the interactome graph
  #'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  graph_info = getGraph(interactome)
  
  #'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #'Build the drug info
  #'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  drug_info <- aggregate(.~Drug,data = drugdt,length)
  colnames(drug_info) <- c("Drug","TargetNum")
  
  #'==============================
  #'compute disease target degree distribution
  #'==============================
  disease_target = unique(diseasedt$Gene)
  DG_degree_distribution = computeDegreeDistribution(disease_target,graph_info)
  
  #'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  #'#calculate the proximity:
  #'from disease nodes to drug(dim = 1)
  #'from drug target to disease target(dim=2)
  #'FOR each drug
  #'drug_info = head(drug_info)
  #'@note column name of the drugdt: Drug, Gene
  #'~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  if(nrow(DG_degree_distribution)> 0){
    
    if(nrow(drug_info)>0){
      #GET the drug proximity
      #'@example x=drug_info[2,1] 
      drug_info$Proximity <- sapply(X = drug_info$Drug,FUN = function(x){
        print(x)
        #get the drug target for each drug
        drug_target <- drugdt[which(drugdt$Drug==x),"Gene"]
        proximity <- computeProximity(drug_target,
                                      disease_target,
                                      graph_info,
                                      metric = mode,
                                      Dim = dim,
                                      Perc_thr = perc_thr)
        return(proximity)
      },simplify = T) 
      
      random_list = list()
      #GET the random proximity
      #x=drug_info$Drug[1]
      #x=drug_info$Drug[1:10]
      random_list <- sapply(X = drug_info$Drug,FUN = function(x){
        print(x)
        drug_target <- drugdt[which(drugdt$Drug==x),"Gene"]
        
        #compute drug target degree distribution
        DR_degree_distribution <- computeDegreeDistribution(drug_target,graph_info)
        
        if(nrow(DR_degree_distribution)>0){
          randomProximity <- computeRandomProximity(DR_degree_distribution,
                                                    DG_degree_distribution,
                                                    graph_info,
                                                    metric = mode,
                                                    iter = iter)
          return(randomProximity)
        }else{
          return(NULL)
        }
      },simplify = T)
      
      #transform the random list to dataframe
      # binding columns together
      random_list = as.data.frame(random_list,stringsAsFactors = F)
      randomdf = do.call(rbind, random_list)
      
      # converting to a dataframe
      randomdf = as.data.frame(randomdf,stringsAsFactors = F)    
      
      #random mean value
      randomdf$mean <- apply(randomdf,MARGIN = 1,FUN = function(x){
        mean_x=mean(x,na.rm = T)
        return(mean_x)},simplify = T)
      
      #random sd value
      randomdf$sd <- apply(randomdf,MARGIN = 1,FUN = function(x){
        sd_x=sd(x,na.rm = T)
        return(sd_x)},simplify = T)
      randomdf$Drug = rownames(randomdf)
      randomdf = randomdf[,c("Drug","mean","sd")]
      
      drug_info = dplyr::left_join(drug_info,randomdf,by=c("Drug"="Drug"))
      
      # #remove na data
      drug_info = na.omit(drug_info)
      
      #get the ZScore and PValue
      drug_info$ZScore <- (drug_info$Proximity-drug_info$mean)/drug_info$sd 
      drug_info$PValue <- pnorm(drug_info$ZScore,lower.tail = T)
      return(drug_info)
    }
    
  }else{
    
    return(NULL)
    
  }    
  
}