import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from src.utils import get_drugdata, get_omicdata, selection

def load_data(label_file,select_method):
    #   pubchem  clname       auc
    # 0    60164  NCIH28  1.243712
    RDKitfile = "data/drug/RDKitFP.csv"
    # (549, 2050)
    #                                      canonical_smiles  pubchem_cid  ...  2046  2047
    #0      C[C@@]12CCC3C(CCC4=C3CCC(=O)C4)C2CC[C@@]1(O)C#C      5702095  ...     0     0
    PPfile = "data/drug/PP.csv"
    # (549, 271)
    #   count  pubchem_cid  BalabanJ     BertzCT  ...    logP  nF  sbonds  tbonds
    # 0     28    5702095.0  1.670869  595.318727  ...  3.6366   0      48       1
    CNVfile = "data/omics/CNV_61.csv"
    # (61, 23318)
    #  clname         ACH    A1BG    NAT2  ...   KCNE2   DGCR2  CASP8AP2    SCO2
    # 0   A375  ACH-000219  0.0885 -0.2958  ... -0.3611 -0.2978 -0.3405 -0.4367
    expfile = "data/omics/exp_61.csv"
    # (61, 48394)
    #  clname         ACH  ...  ENSG00000273492.1  ENSG00000273493.1
    # 0   A375  ACH-000219  ...               0.00                0.0
    Methfile = "data/omics/Meth_61.csv"  # Meth数据少一个细胞系：HT29
    # (61, 69643)
    #  clname         ACH  ...  22_51159223_51162060  22_51159223_51162060.1
    # 0   A375  ACH-000219  ...               0.74638                 0.73370
    miRNAfile = "data/omics/RNA_61.csv"
    # (61, 736)
    #  clname         ACH  hsa-let-7a  ...  kshv-miR-K12-9  mcv-miR-M1-3p  mcv-miR-M1-5p
    # 0   A375  ACH-000219  3541.67  1022.13  ...  10.22  35.77  11.92  23.85
    Mutfile = "data/omics/Mut_61.csv"
    # (61,34675)
    #  clname         ACH  ...  MTCP1.23:154294279  MTCP1.23:154298967
    # 0   A375  ACH-000219  ...                 0.0                 0.0

    Label = pd.read_csv(label_file, index_col=False,sep='\t')
    MF = pd.read_csv(RDKitfile)
    PP = pd.read_csv(PPfile, index_col=False, sep='\t')
    CNV = pd.read_csv(CNVfile, index_col=False, sep='\t')
    exp = pd.read_csv(expfile, index_col=False, sep='\t')
    Meth = pd.read_csv(Methfile, index_col=False, sep='\t')
    miRNA = pd.read_csv(miRNAfile, index_col=False, sep='\t')
    Mut = pd.read_csv(Mutfile, index_col=False, sep='\t')
    n_sample = len(Label)
    # Label = Label.iloc[:200,:]

    print("MFshape:",MF.shape)
    MFlabel = MF.iloc[:, 0:2]
    MFalldata = np.array(MF.iloc[:, 2:])
    MFuniques = np.unique(MFalldata, axis=1)  # 删除重复列
    MFalldata = MFuniques[:, ~np.all(MFuniques[1:] == MFuniques[:-1], axis=0)]
    print(MFuniques.shape)
    print(MFalldata.shape)
    MFalldata = pd.DataFrame(MFalldata)
    columns = [i for i in range(MFalldata.shape[1])]
    MFalldata.columns = columns
    MF = pd.concat([MFlabel, MFalldata], axis=1)
    MFdata = get_drugdata(Label, MF)
    print(MFdata.shape)

    print("PPshape:", PP.shape)
    PPlabel = PP.iloc[:, 0:2]
    PPdata = PP.iloc[:, 2:]
    duplicated = PPdata.T.duplicated()
    col = [i for i in range(len(duplicated)) if duplicated[i] == False]
    PPdata = PPdata.iloc[:, col]
    print("删除重复列后的数据维度：")
    print(PPdata.shape)
    cols = []
    for i in range(len(PPdata.columns)):
        if len(PPdata.iloc[:, i].unique()) != 1:
            cols.append(i)
    PPalldata = PPdata.iloc[:, cols]
    print("删除列中值唯一后的数据维度:")
    print(PPalldata.shape)
    print("删除列中值唯一后的列名:")
    print(PPalldata.columns)
    PP = pd.concat([PPlabel, PPalldata], axis=1)
    PPdata = get_drugdata(Label, PP)
    print(PPdata.shape)

    print("CNVshape:", CNV.shape)
    CNVlabel = CNV.iloc[:, 0:2]
    CNVdata = CNV.iloc[:, 2:]
    duplicated = CNVdata.T.duplicated()
    col = [i for i in range(len(duplicated)) if duplicated[i] == False]
    CNVdata = CNVdata.iloc[:, col]
    print("删除重复列后的数据维度：")
    print(CNVdata.shape)
    cols = []
    for i in range(len(CNVdata.columns)):
        if len(CNVdata.iloc[:, i].unique()) != 1:
            cols.append(i)
    CNValldata = CNVdata.iloc[:, cols]
    print("删除列中值唯一后的数据维度:")
    print(CNValldata.shape)
    print("删除列中值唯一后的列名:")
    print(CNValldata.columns)
    CNV = pd.concat([CNVlabel, CNValldata], axis=1)
    CNVdata = get_omicdata(Label, CNV)
    CNVdata,CNVcolumns = selection(Label,CNVdata,CNValldata.columns,1000,select_method)

    print("expshape:", exp.shape)
    Genedic = pd.read_csv("data/omics/Genedic.txt", index_col=False, sep='\t')
    gene = list(Genedic["SYMBOL"])
    ensg = list(Genedic["ENSEMBL"])
    new_column = list(exp.columns[:2])
    gene_column = list(exp.columns[:2])
    for name in exp.columns:
        exp_ensg = name.split(".")[0]
        if exp_ensg in ensg:
            index = ensg.index(exp_ensg)
            gene_column.append(gene[index])
            new_column.append(name)
    exp = exp[new_column]
    exp.columns = gene_column

    explabel = exp.iloc[:, 0:2]
    expdata = exp.iloc[:, 2:]
    duplicated = expdata.T.duplicated()
    col = [i for i in range(len(duplicated)) if duplicated[i] == False]
    expdata = expdata.iloc[:, col]
    print("删除重复列后的数据维度：")
    print(expdata.shape)
    cols = []
    for i in range(len(expdata.columns)):
        if len(expdata.iloc[:, i].unique()) != 1:
            cols.append(i)
    expalldata = expdata.iloc[:, cols]
    print("删除列中值唯一后的数据维度:")
    print(expalldata.shape)
    print("删除列中值唯一后的列名:")
    print(expalldata.columns)
    exp = pd.concat([explabel, expalldata], axis=1)
    expdata = get_omicdata(Label, exp)
    expdata,expcolumns = selection(Label,expdata,expalldata.columns,1000,select_method)

    print("Methshape:", Meth.shape)
    new_column = []
    for name in Meth.columns:
        if len(name.split(".")) == 1:
            new_column.append(name.split(".")[0])
    print(len(new_column))
    Meth = Meth[new_column]

    Methlabel = Meth.iloc[:, 0:2]
    Methdata = Meth.iloc[:, 2:]
    duplicated = Methdata.T.duplicated()
    col = [i for i in range(len(duplicated)) if duplicated[i] == False]
    Methdata = Methdata.iloc[:, col]
    print("删除重复列后的数据维度：")
    print(Methdata.shape)
    cols = []
    for i in range(len(Methdata.columns)):
        if len(Methdata.iloc[:, i].unique()) != 1:
            cols.append(i)
    Methalldata = Methdata.iloc[:, cols]
    print("删除列中值唯一后的数据维度:")
    print(Methalldata.shape)
    print("删除列中值唯一后的列名:")
    print(Methalldata.columns)
    Meth = pd.concat([Methlabel, Methalldata], axis=1)
    Methdata = get_omicdata(Label, Meth)
    Methdata,Methcolumns = selection(Label,Methdata,Methalldata.columns,1000,select_method)

    print("miRNAshape:", miRNA.shape)
    miRNAlabel = miRNA.iloc[:, 0:2]
    miRNAdata = miRNA.iloc[:, 2:]
    duplicated = miRNAdata.T.duplicated()
    col = [i for i in range(len(duplicated)) if duplicated[i] == False]
    miRNAdata = miRNAdata.iloc[:, col]
    print("删除重复列后的数据维度：")
    print(miRNAdata.shape)
    cols = []
    for i in range(len(miRNAdata.columns)):
        if len(miRNAdata.iloc[:, i].unique()) != 1:
            cols.append(i)
    miRNAalldata = miRNAdata.iloc[:, cols]
    print("删除列中值唯一后的数据维度:")
    print(miRNAalldata.shape)
    print("删除列中值唯一后的列名:")
    print(miRNAalldata.columns)
    miRNA = pd.concat([miRNAlabel, miRNAalldata], axis=1)
    miRNAdata = get_omicdata(Label, miRNA)
    # miRNAdata,miRNAcolumns = selection(Label,miRNAdata,miRNAalldata.columns,734,select_method)

    print("Mutshape:", Mut.shape)
    Mutlabel = Mut.iloc[:, 0:2]
    Mutdata = Mut.iloc[:, 2:]
    duplicated = Mutdata.T.duplicated()
    col = [i for i in range(len(duplicated)) if duplicated[i] == False]
    Mutdata = Mutdata.iloc[:, col]
    print("删除重复列后的数据维度：")
    print(Mutdata.shape)
    cols = []
    for i in range(len(Mutdata.columns)):
        if len(Mutdata.iloc[:, i].unique()) != 1:
            cols.append(i)
    Mutalldata = Mutdata.iloc[:, cols]
    print("删除列中值唯一后的数据维度:")
    print(Mutalldata.shape)
    print("删除列中值唯一后的列名:")
    print(Mutalldata.columns)
    Mut = pd.concat([Mutlabel, Mutalldata], axis=1)
    Mutdata = get_omicdata(Label, Mut)
    Mutcolumns = Mutalldata.columns

    auc = np.array(Label['auc'])
    auc = auc[:, np.newaxis]
    alldata = np.hstack((MFdata, PPdata, CNVdata, expdata,Methdata, miRNAdata, Mutdata, auc))
    print(alldata.shape)
    scaler = MinMaxScaler(feature_range=(0, 1))
    alldata = scaler.fit_transform(alldata)

    x = alldata[:, :-1]
    y = alldata[:, -1]

    return x, y


