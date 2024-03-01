#'==============================================================================
#'@description define the parameters and the input files of the proximity model
#'==============================================================================


#'==============================================================================
#'@description trace the functions
#'==============================================================================
source("./Model/Proximity/Proximity_Func/computeDegreeDistribution.R")
source("./Model/Proximity/Proximity_Func/getGraph.R")
source("./Model/Proximity/Proximity_Func/computeDegreeDistribution.R")
source("./Model/Proximity/Proximity_Func/computeProximity.R")
source("./Model/Proximity/Proximity_Func/ProximityCalculation.R")
source("./Model/Proximity/Proximity_Func/computeRandomProximity.R")
source("./Model/Proximity/Proximity_Func/selectRandomNodes.R")

#'==============================================================================
#'@function: set the proximity model paremetres:
#'@param1: mode:max, mean, median, min, mode
#'@param2: thr_pval, default=0.05
#'@param3: dim, defautl = 1 
#'dim = 1 -> row mean (ie, from DrugTarget to DiseaseGene), 
#' dim = 2 -> col mean (ie, from DiseaseGene to DrugTarget)
#'@param4: iter: the random times, default=100
#'@param5:Perc_thr, threshold for proximity calculation, default=5
#'==============================================================================
config = function(){
  #the proximity mode
  mode = "min"
  #the p-value threshold
  thr_pval = 0.05
  #dim
  dim = 1
  #iter
  iter = 100
  #Perc_thr
  perc_thr = 5
  # edge-weight = similarity or proximity
  interactions = "similarity"  
  # adjust similarity or not
  adjust_link = T          
  # add new drug-disease association or not (without compute pval)
  new_link = F              
  
  if(interactions == "proximity"){
    distance = "proximity"
  }
  if( (interactions == "similarity") & (adjust_link == F) ){
    distance = "similarity"
  } 
  if( (interactions == "similarity") & (adjust_link == T) ){
    distance = "adjusted_similarity"
  }
  # parameters for computing subnetwork
  # sel_drug = "tocilizumab"
  # sel_disease = "Severe Acute Respiratory Syndrome"
  sel_drug = NULL
  sel_disease = NULL
  
  configsettings = list(mode = mode,
                        thr_pval = thr_pval,
                        dim = dim,
                        iter = iter,
                        perc_thr = perc_thr,
                        interactions = interactions,
                        adjust_link = adjust_link,
                        new_link = new_link,
                        distance = distance,
                        sel_drug = sel_drug,
                        sel_disease = sel_disease
  )
  return(configsettings)
}

#' #'==============================================================================
#' #'@function: define the input file
#' #'==============================================================================
inputFiles = function(){
#'   ########################################
#'   # ppi interactome
#'   # target-target
ppiinteractome = read.table("./Data/ppidata.csv", header = T, sep = '\t', check.names = F)
ppiinteractome = ppiinteractome[,c(1,2)]
colnames(ppiinteractome) = c("From","To")
ppiinteractome = ppiinteractome[!duplicated(ppiinteractome),]

allppigene = unique(c(ppiinteractome$From,ppiinteractome$To))
#drug-target
drug_target = read.table("./Data/Drug_target/DTI_approved_combine.csv", header = T, sep = '\t', check.names = F)
colnames(drug_target) = c("Drug","Gene")
drug_target = drug_target[drug_target$Gene%in%allppigene,]

#'disease-target   
disease_target = read.delim("./result/diseasesensitivity/senrestarget.csv",header = T, sep = '\t', check.names = F)
colnames(disease_target) = c("Disease","Gene")
disease_target = disease_target[disease_target$Gene%in%allppigene,]
#'   
#'   ########################################
#'   
  #combine the drug-target, disease-target and target-target(ppi) data
  input_file = list(interactome = ppiinteractome,
                    disease_gene = disease_target,
                    drug_target = drug_target)
                    # known_association =  known_association)

  return(input_file)
}




