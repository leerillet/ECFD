import numpy as np
import multiprocessing as mp
import pandas as pd
from deepforest import CascadeForestRegressor
from config import *
from sklearn.model_selection import RepeatedKFold,train_test_split,KFold,RepeatedStratifiedKFold
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from scipy.stats import pearsonr, spearmanr
from MultiChannel_gcForest import MCgcF


def par_kfold(x,y,n_spilt=5,n_repeat=3,random_state=0,n_jobs=10,forests_type = 'XGBET',n_estimator=10):
    # metrics
    rmses,r2s,pearsonrs,spearmanrs,y_vals,y_preds={},{},{},{},{},{}
    # train & test spilt, using k-folds cross-validation
    kf=RepeatedKFold(n_splits=n_spilt,n_repeats=n_repeat,random_state=random_state)
    cv=[(t,v) for (t,v) in kf.split(x)]
    # allocate tasks for multiple process
    allocations=allocate_tasks(n_spilt*n_repeat,n_jobs)  #分配n_jobs个
    pool=mp.Pool(n_jobs)
    tasks=[pool.apply_async(kfold_async,args=(x,y,k,task,cv,forests_type,n_estimator)) for k,task in enumerate(allocations)]
    #k 第几个task task为起点和终点
    pool.close()
    pool.join()
    results=[task.get() for task in tasks]
    for result in results:
        rmses.update(result[0])
        r2s.update(result[1])
        pearsonrs.update(result[2])
        spearmanrs.update(result[3])
        y_vals.update(result[4])
        y_preds.update(result[5])
    rmses=sorted(rmses.items(),key=lambda x: x[0])
    rmses=[value[1] for value in rmses]
    r2s=sorted(r2s.items(),key=lambda x: x[0])
    r2s=[value[1] for value in r2s]
    pearsonrs=sorted(pearsonrs.items(),key=lambda x: x[0])
    pearsonrs=[value[1] for value in pearsonrs]
    spearmanrs = sorted(spearmanrs.items(), key=lambda x: x[0])
    spearmanrs = [value[1] for value in spearmanrs]
    y_vals=sorted(y_vals.items(),key=lambda x: x[0])
    y_vals=[value[1] for value in y_vals]
    y_preds=sorted(y_preds.items(),key=lambda x: x[0])
    y_preds=[value[1] for value in y_preds]
    return rmses,r2s,pearsonrs,spearmanrs,y_vals,y_preds

def kfold_async(x,y,k,task,cv,forests_type,n_estimator):
    begin,end=task[0],task[1]
    print("Process_{}: {}-{} folds".format(k,begin,end-1))
    mses,rmses,maes,r2s,pearsonrs,spearmanrs, y_tests,y_preds={},{},{},{},{},{},{},{}
    config = get_config(forests_type, n_estimator)  #获取多通道深度森林参数
    for i in range(begin,end):
        (train_id,test_id)=cv[i]
        x_train,x_test,y_train,y_test=x[train_id],x[test_id],y[train_id],y[test_id]
        model=MCgcF(config,i)
        model.fit(x_train,y_train)
        y_pred=model.predict(x_test)

        mses[i]=mean_squared_error(y_test,y_pred)
        rmses[i]=np.sqrt(mses[i])
        r2s[i]=r2_score(y_test,y_pred)
        pearsonrs[i]=pearsonr(y_test,y_pred)[0]
        spearmanrs[i] = spearmanr(y_test, y_pred)[0]
        y_tests[i]=y_test
        y_preds[i]=y_pred
    return (rmses,r2s,pearsonrs,spearmanrs,y_tests,y_preds)

def allocate_tasks(allfold,n_jobs):
    task_num=allfold//n_jobs  #共n_jobs个task
    fold_left=allfold%n_jobs  #剩余fold数
    job_lens=[task_num for i in range(n_jobs)]  #每个job里有task_num个fold
    for i in range(fold_left):
        job_lens[i]+=1
    allocations=[]
    begin=0
    for i in range(len(job_lens)):
        allocations.append((begin,begin+job_lens[i]))
        begin=begin+job_lens[i]
    return allocations