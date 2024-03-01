import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest,f_regression
from sklearn.decomposition import PCA

def get_columns(data):
    duplicated = data.T.duplicated()
    col = [i for i in range(len(duplicated)) if duplicated[i] == False]
    data = data.iloc[:,col]
    col = []
    for i in range(len(data.columns)):
        if len(data.iloc[:,i].unique())!=1:
            col.append(i)
    data = data.iloc[:,col]
    return list(data.columns)

def get_drugdata(Label,drug_info_feature):
    n_sample = len(Label)
    drugdata = np.zeros((n_sample, drug_info_feature.shape[1]-2))
    for i in range(n_sample):
        drug_id = Label.at[i,"pubchem"]
        drug_feature = get_one_drug(drug_info_feature,drug_id)
        drugdata[i] = drug_feature
    print(drugdata.shape)
    return drugdata

def get_omicdata(Label,omic_info_feature):
    n_sample = len(Label)
    omicdata = np.zeros((n_sample, omic_info_feature.shape[1]-2))
    for i in range(n_sample):
        cellline = Label.at[i,"clname"]
        omic_feature = get_one_omic(omic_info_feature,cellline)
        omicdata[i] = omic_feature
    print(omicdata.shape)
    return omicdata

def get_one_drug(drug_info_feature,drug_id):
    drug_feature = drug_info_feature.loc[drug_info_feature["pubchem_cid"] == drug_id]
    drug_feature = np.array(drug_feature)
    drug_feature = drug_feature.reshape(drug_feature.shape[1])[2:]
    return drug_feature

def get_one_omic(omic_feature, cellline):
    omic = omic_feature.loc[omic_feature["clname"] == cellline]
    omic = np.array(omic)
    omic = omic.reshape(omic.shape[1])[2:]
    return omic

def selection(Label,omicdata,columns,num,select_method):
    data = np.array(omicdata)
    print(data.shape)
    auc = np.array(Label['auc'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    if select_method=='f':
        sel = SelectKBest(f_regression, k=num)
        sel.fit(data, auc)
        data = sel.transform(data)
        fea = sel.get_feature_names_out(input_features=columns)
    elif select_method=='pca':
        data = PCA(n_components=num).fit_transform(data)
        fea = [i for i in range(num)]
    elif select_method=='random':
        cols = np.random.choice([i for i in range(data.shape[1])], size=num, replace=False)
        data = data[:,cols]
        columns = np.array(columns)
        fea = columns[cols]
    print("处理后数据：{}".format(data.shape))
    return data, fea







