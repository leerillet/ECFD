library(igraph)
library(data.table)
library(R.utils)
library(stringr)
library(e1071)
library(DescTools)
options(stringsAsFactors = F)
#source the proximity config
######################################
source("./Model/Proximity/Proximity_config.R")
source("./Model/Proximity/Proximity_model.R")
######################################

#'==============================
#'load the default settings
#'==============================
configSettings = config()

#'==============================
#'load the ppi,disease,herb/drug data
#'==============================
inputfile = inputFiles()

######################################
# input parameters
Mode = configSettings$mode
Thr_pval = configSettings$thr_pval
Dim = configSettings$dim
Iter = configSettings$iter
Perc_thr = configSettings$perc_thr

# input files
Interactome = inputfile$interactome
diseasetargetDT = inputfile$disease_gene
drugtargetDT = inputfile$drug_target
# colnames(drugtargetDT) = c("Drug","Gene")

#get the drug proximity score by the model
druginfo = ProximityModel(Interactome,diseasetargetDT,drugtargetDT,Mode,Dim,Iter,Perc_thr)
#remove the na data and select significant data
druginfo = na.omit(druginfo)
druginfo = druginfo[which(druginfo$PValue<=Thr_pval),]