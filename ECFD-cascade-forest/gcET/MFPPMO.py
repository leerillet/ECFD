import numpy as np
import pandas as pd
import os
from sklearn.decomposition import PCA,FastICA
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2,f_regression,mutual_info_regression,VarianceThreshold

def load_data():
    # Labelfile = "data/Label_4116.csv"  #4116
    all_labelfile = "data/all_label.csv"  #16000左右的样本
    #(4116, 5)
    #        drug cell line       auc      ec50      ic50
    #0  5702095.0      A375  0.956381  6.461743  6.523677
    MFfile = "data/feature/Drug features/MF.csv"
    #(549, 883)
    #   count  pubchem_cid  >=_4_H  ...  ClC1C(Cl)CCC1  ClC1C(Br)CCC1  BrC1C(Br)CCC1
    #0     28    5702095.0       1  ...              0              0              0
    PPfile = "data/feature/Drug features/PP.csv"
    #(549, 271)
    #   count  pubchem_cid  BalabanJ     BertzCT  ...    logP  nF  sbonds  tbonds
    #0     28    5702095.0  1.670869  595.318727  ...  3.6366   0      48       1
    CNVfile = "data/feature/Multi-omics features/CNV.csv"
    #(31, 23318)
    #      clname    0    A1BG    NAT2  ...   KCNE2   DGCR2  CASP8AP2    SCO2
    #0  A375_SKIN  505  0.0885 -0.2958  ... -0.3611 -0.2978   -0.3405 -0.4367
    expfile = "data/feature/Multi-omics features/exp.csv"
    #(31, 48394)
    #      clname    0  ...  ENSG00000273492.1  ENSG00000273493.1
    #0  A375_SKIN  505  ...                0.0                0.0
    Methfile ="data/feature/Multi-omics features/Meth.csv"   #Meth数据少一个细胞系：HT29
    #(31, 69643)
    #      clname    0  ...  22_51159223_51162060  22_51159223_51162060.1
    #0  A375_SKIN  505  ...               0.74638                  0.7337
    miRNAfile = "data/feature/Multi-omics features/miRNA.csv"
    #(31, 736)
    #      clname    0  hsa-let-7a  ...  kshv-miR-K12-9  mcv-miR-M1-3p  mcv-miR-M1-5p
    #0  A375_SKIN  505     3541.67  ...           35.77          11.92          23.85
    celllinefile = "data/feature/cell_line12328.csv"  #4116 12328 没使用978标签选择  只能对应4116个样本 不能拼接出15459个 即15495个样本不使用该数据
    #(4116,978)
    #        5720.0   55847.0   25803.0  ...   10153.0    2597.0     874.0
    #0    -1.942851 -1.305892  1.383015  ... -1.216087 -0.437080 -2.379971

    # Label = pd.read_csv(Labelfile, index_col=False, sep=',') #不使用第一列作为索引  根据细胞系排序后分隔符变为','
    Label = pd.read_csv(all_labelfile, index_col=False, sep='\t') 
    MF = pd.read_csv(MFfile, index_col=False, sep='\t')
    PP = pd.read_csv(PPfile, index_col=False, sep='\t')
    CNV = pd.read_csv(CNVfile, index_col=False, sep='\t')
    exp = pd.read_csv(expfile, index_col=False, sep='\t')
    Meth = pd.read_csv(Methfile, index_col=False, sep='\t')
    miRNA = pd.read_csv(miRNAfile, index_col=False, sep='\t')
    cellline = pd.read_csv(celllinefile, header=None, index_col=False, sep='\t')
    print(cellline.shape)

    #处理数据集
    # processA549()
    # Label.columns = ["drug", "cell line", "auc", "ec50", "ic50"]  #调整Label_4116
    # Label.to_csv('data/Label_4116.csv', sep='\t',index=False,header=["drug", "cell line", "auc", "ec50", "ic50"])
    # Meth.to_csv('data/feature/Cell line/A549.csv', sep='\t', header=True)
    # miRNA = miRNA.drop([10], axis=0)
    # miRNA.to_csv('data/feature/Multi-omics features/miRNA.csv', sep='\t',index=False, header=True)

    #去掉HT29细胞系
    # Label = Label[~(Label["cell line"] == "HT29")] #(15495, 5)
    # Label.to_csv('data/all_label.csv', sep='\t',index=False, header=True)

    n_sample = Label.shape[0]  # 样本个数

    print("MFshape:",MF.shape)
    MFlabel = MF.iloc[:,0:2]
    MFalldata = np.array(MF.iloc[:,2:])
    MFuniques = np.unique(MFalldata,axis = 1)  #删除重复列
    MFalldata = MFuniques[:, ~np.all(MFuniques[1:] == MFuniques[:-1], axis=0)]
    print(MFuniques.shape)
    print(MFalldata.shape)
    MFalldata = pd.DataFrame(MFalldata)
    columns = [i for i in range(MFalldata.shape[1])]
    MFalldata.columns = columns
    MF = pd.concat([MFlabel, MFalldata], axis=1)
    print("处理后数据：{}".format(MF.shape))
    MFdata = np.zeros((n_sample, (MF.shape[1] - 2)))  # MF最后去掉前前两列
    for i in range(n_sample):
        drug_id =  Label.at[i,"drug"]  #根据药物id来找药物数据
        drug_MF = get_MF(MF,drug_id)
        MFdata[i] = drug_MF
    print(MFdata.shape) #(4116, 881)
    MFdata = selection(Label,MFdata,400)
    # #
    print("PPshape:", PP.shape)
    PPlabel = PP.iloc[:, 0:2]
    PPalldata = np.array(PP.iloc[:, 2:])
    PPuniques = np.unique(PPalldata, axis=1)  # 删除重复列
    PPalldata = PPuniques[:, ~np.all(PPuniques[1:] == PPuniques[:-1], axis=0)]
    print(PPuniques.shape)
    print(PPalldata.shape)
    PPalldata = pd.DataFrame(PPalldata)
    columns = [i for i in range(PPalldata.shape[1])]
    PPalldata.columns = columns
    PP = pd.concat([PPlabel, PPalldata], axis=1)
    print("处理后数据：{}".format(PP.shape))
    PPdata = np.zeros((n_sample, (PP.shape[1] - 2)))  # MF最后去掉前前两列
    for i in range(n_sample):
        drug_id = Label.at[i, "drug"]  # 根据药物id来找药物数据
        drug_PP = get_MF(PP, drug_id)
        PPdata[i] = drug_PP
    print(PPdata.shape)  #
    # PPdata = selection(Label, PPdata,200)

    print("CNVshape:", CNV.shape)
    CNVlabel = CNV.iloc[:, 0:2]
    CNValldata = np.array(CNV.iloc[:, 2:])
    CNVuniques = np.unique(CNValldata, axis=1)  # 删除重复列
    CNValldata = CNVuniques[:, ~np.all(CNVuniques[1:] == CNVuniques[:-1], axis=0)]
    print(CNVuniques.shape)
    print(CNValldata.shape)
    data_median = np.median(np.var(CNValldata, axis=0))
    # data_median = np.var(CNValldata)
    CNValldata = VarianceThreshold(data_median).fit_transform(CNValldata)
    CNValldata = pd.DataFrame(CNValldata)
    columns = [i for i in range(CNValldata.shape[1])]
    CNValldata.columns = columns
    CNV = pd.concat([CNVlabel, CNValldata], axis=1)
    print("处理后数据：{}".format( CNV.shape))
    CNVdata = np.zeros((n_sample, (CNV.shape[1] - 2)))
    for i in range(n_sample):
        drug_cellline = Label.at[i,"cell line"]
        drug_CNV = get_CNV(CNV,drug_cellline)
        CNVdata[i] = drug_CNV
    print(CNVdata.shape) #(4116, 23316)
    CNVdata = selection(Label, CNVdata,100)

    print("expshape:", exp.shape)
    explabel = exp.iloc[:, 0:2]
    expalldata = np.array(exp.iloc[:, 2:])
    expuniques = np.unique(expalldata, axis=1)  # 删除重复列
    expalldata = expuniques[:, ~np.all(expuniques[1:] == expuniques[:-1], axis=0)]
    print(expuniques.shape)
    print(expalldata.shape)
    data_median = np.median(np.var(expalldata, axis=0))
    # data_median = np.var(expalldata)
    expalldata = VarianceThreshold(data_median).fit_transform(expalldata)
    expalldata = pd.DataFrame(expalldata)
    columns = [i for i in range(expalldata.shape[1])]  #194
    expalldata.columns = columns
    exp = pd.concat([explabel, expalldata], axis=1)
    print("处理后数据：{}".format(exp.shape))
    expdata = np.zeros((n_sample, (exp.shape[1] - 2)))
    for i in range(n_sample):
        drug_cellline = Label.at[i,"cell line"]
        drug_exp = get_exp(exp, drug_cellline)
        expdata[i] = drug_exp
    print(expdata.shape) #(4116, 48392)
    expdata = selection(Label, expdata,100)

    print("Methshape:", Meth.shape)
    Methlabel = Meth.iloc[:, 0:2]
    Methalldata = np.array(Meth.iloc[:, 2:])
    Methuniques = np.unique(Methalldata, axis=1)  # 删除重复列
    Methalldata = Methuniques[:, ~np.all(Methuniques[1:] == Methuniques[:-1], axis=0)]
    print(Methuniques.shape)
    print(Methalldata.shape)
    data_median = np.median(np.var(Methalldata, axis=0))
    # data_median = np.var(Methalldata)
    Methalldata = VarianceThreshold(data_median).fit_transform(Methalldata) #452
    Methalldata = pd.DataFrame(Methalldata)
    columns = [i for i in range(Methalldata.shape[1])]  # 194
    Methalldata.columns = columns
    Meth = pd.concat([Methlabel, Methalldata], axis=1)
    print("处理后数据：{}".format(Meth.shape))
    Methdata = np.zeros((n_sample, (Meth.shape[1] - 2)))
    for i in range(n_sample):
        drug_cellline = Label.at[i,"cell line"]
        drug_Meth = get_Meth(Meth, drug_cellline)
        Methdata[i] = drug_Meth
    print(Methdata.shape)  #(4116, 69641)
    Methdata = selection(Label, Methdata,100)

    print("miRNAshape:", miRNA.shape)
    miRNAlabel = miRNA.iloc[:, 0:2]
    miRNAalldata = np.array(miRNA.iloc[:, 2:])
    miRNAuniques = np.unique(miRNAalldata, axis=1)  # 删除重复列
    miRNAalldata = miRNAuniques[:, ~np.all(miRNAuniques[1:] == miRNAuniques[:-1], axis=0)]
    print(miRNAuniques.shape)
    print(miRNAalldata.shape)
    miRNAalldata = pd.DataFrame(miRNAalldata)
    columns = [i for i in range(miRNAalldata.shape[1])]  # 194
    miRNAalldata.columns = columns
    miRNA = pd.concat([miRNAlabel, miRNAalldata], axis=1)
    print("处理后数据：{}".format(miRNA.shape))
    miRNAdata = np.zeros((n_sample, (miRNA.shape[1] - 2)))
    for i in range(n_sample):
        drug_cellline = Label.at[i, "cell line"]
        drug_RNA = get_RNA(miRNA, drug_cellline)
        miRNAdata[i] = drug_RNA
    print(miRNAdata.shape)  #(4116,734)
    miRNAdata = selection(Label, miRNAdata,100)

    # print("celllineshape:", cellline.shape)
    # celllinedata = np.array(cellline)
    # celllineuniques = np.unique(celllinedata, axis=1)  # 删除重复列
    # celllinealldata = celllineuniques[:, ~np.all(celllineuniques[1:] == celllineuniques[:-1], axis=0)]
    # print(celllineuniques.shape)
    # print(celllinealldata.shape)
    # data_median = np.median(np.var(celllinealldata, axis=0))
    # # data_median = np.var(celllinealldata)
    # celllinealldata = VarianceThreshold(data_median).fit_transform(celllinealldata)  # 452
    # print("处理后数据：{}".format(celllinealldata.shape))
    # celllinedata = selection(Label, celllinealldata,400)

    auc = np.array(Label['auc'])
    auc = auc[:, np.newaxis]
    alldata = np.hstack((MFdata,PPdata,CNVdata,expdata, Methdata, miRNAdata,auc))
    # alldata = np.hstack((MFdata, PPdata, CNVdata, expdata, Methdata, miRNAdata, celllinedata, auc))
    print(alldata.shape)
    np.savetxt('reprocessdata/Data15459_f_1031.csv', alldata, fmt='%s', delimiter=',')



def get_MF(MF,drug_id):
    drug_MF = MF.loc[MF["pubchem_cid"] == drug_id]
    drug_MF = np.array(drug_MF)
    drug_MF = drug_MF.reshape(drug_MF.shape[1])[2:]
    return drug_MF

def get_PP(PP,drug_id):
    drug_PP = PP.loc[PP["pubchem_cid"] == drug_id]
    drug_PP = np.array(drug_PP)
    drug_PP = drug_PP.reshape(drug_PP.shape[1])[2:]
    return drug_PP

def get_CNV(CNV, drug_cellline):
    CNV["clname"] = CNV["clname"].map(lambda r:r.split('_')[0])
    drug_CNV = CNV.loc[CNV["clname"] == drug_cellline]
    drug_CNV = np.array(drug_CNV)
    drug_CNV = drug_CNV.reshape(drug_CNV.shape[1])[2:]
    return drug_CNV

def get_exp(exp, drug_cellline):
    exp["clname"] = exp["clname"].map(lambda r: r.split('_')[0])
    drug_exp = exp.loc[exp["clname"] == drug_cellline]
    drug_exp = np.array(drug_exp)
    drug_exp = drug_exp.reshape(drug_exp.shape[1])[2:]
    return drug_exp

def get_Meth(Meth, drug_cellline):
    Meth["clname"] = Meth["clname"].map(lambda r: r.split('_')[0])
    drug_Meth = Meth.loc[Meth["clname"] == drug_cellline]
    drug_Meth = np.array(drug_Meth)
    drug_Meth = drug_Meth.reshape(drug_Meth.shape[1])[2:]
    return drug_Meth

def get_RNA(RNA, drug_cellline):
    RNA["clname"] = RNA["clname"].map(lambda r: r.split('_')[0])
    drug_RNA = RNA.loc[RNA["clname"] == drug_cellline]
    drug_RNA = np.array(drug_RNA)
    drug_RNA = drug_RNA.reshape(drug_RNA.shape[1])[2:]
    return drug_RNA

def processA549():
    A549 = pd.read_csv('data/feature/Cell line/A549.csv', index_col=False, sep='\t')  #处理A549
    A549.iloc[:,0] = A549.iloc[:,-1]  #将最后一列ID调到第一列  #调整列
    A549.rename(columns={'Unnamed: 0':'ID'},inplace=True)  #更改列名
    A549 = A549.drop(['Unnamed: 0','LSM_id','pert_iname','canonical_smiles', 'pubchem_cid', '0','ID.1'], axis=1)  #删除dataframe列
    A549.to_csv('data/feature/Cell line/A549.csv',index=False, sep='\t', header=True)

def selection(Label,data,num):
    data = np.array(data)
    print(data.shape)
    auc = np.array(Label['auc'])
    scaler = MinMaxScaler(feature_range=(0, 1))
    data = scaler.fit_transform(data)
    data = SelectKBest(f_regression, k=num).fit_transform(data, auc)
    # ICA = FastICA(n_components=200, random_state=0,max_iter=1000)
    # data = ICA.fit_transform(data)
    # data = PCA(n_components=200).fit_transform(data, auc)
    # data = PCA(n_components=100).fit_transform(data)
    print("处理后数据：{}".format(data.shape))
    return data



load_data()
