#'==============================================================================
#'@function: ASAB model for drug effect analysis
#'@param targetlist1 the one target list
#'@param targetlist2 the other target list
#'@param interactome the interaction network
#'@param drugtargetdt the drug-target dataframe
#'@export sabres the drug combination prediction result
#'==============================================================================
source("./Model/Proximity/Proximity_main.R") 
source("./Model/Proximity/Similarity/computeSimilarity.R")
source("./Model/Proximity/Similarity/createAdjMatrix.R")

#'==============================
#'load the default settings
#'==============================
configSettings = config()

#'==============================
#'load the ppi,disease,herb/drug data
#'==============================
inputfile = inputFiles()
#######################################

######################################
# input parameters
Mode = configSettings$mode
Thr_pval = configSettings$thr_pval
Perc_thr = configSettings$perc_thr
interactions <- configSettings$interactions
adjust_link <- configSettings$adjust_link
new_link <- configSettings$new_link

# input files
Interactome = inputfile$interactome
diseasetargetDT = inputfile$disease_gene
drugtargetDT = inputfile$drug_target


druginfo = ProximityModel(Interactome,diseasetargetDT,drugtargetDT,Mode,Dim,Iter=1,Perc_thr)
druginfo = na.omit(druginfo)
druginfo = druginfo[which(!is.infinite(druginfo$ZScore)),]
druginfo = druginfo[which(druginfo$PValue<=Thr_pval),]
druginfo$disease = rep("Disease",nrow(druginfo))
colnames(druginfo)[1]=c("drug")
druginfo$interactions = rep("similarity",nrow(druginfo))

drug_disease_net = druginfo
if(interactions == "similarity"){
  drug_disease_net = computeSimilarity(drug_disease_net)
} 
