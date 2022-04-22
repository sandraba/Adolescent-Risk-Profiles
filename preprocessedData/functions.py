import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
from MissForestExtra import MissForestExtra
import sklearn.cluster
import sys
import scipy
import Gap_statistics
from kmodes.kmodes import KModes

def result(df,labels): #show the result of clustering 
    a = ['Young','Old','Female','Male'] #Q1 and Q2 need to be treated specially
    b = []
    for i in df.columns:
        a.append(i)
        b.append(i)
    a.remove('Q1')
    a.remove('Q2')
    b.remove('Q1')
    b.remove('Q2')
    df_result = pd.DataFrame(0,columns = a,index = np.unique(labels)) 
    #create a data frame with rows number equal to the number of clusters and columns number equal to the original data frame
    for i in range(len(df)):
        for j in b:
            if df.loc[i,j] == 2:
                df_result.loc[labels[i],j]+=1
        if df.loc[i,'Q1'] < 15: #treate people younger than 15 as young
            df_result.loc[labels[i],'Young']+=1
        else:
            df_result.loc[labels[i],'Old']+=1
        if df.loc[i,'Q2'] == 2:
            df_result.loc[labels[i],'Female']+=1
        else:
            df_result.loc[labels[i],'Male']+=1
    for i in df_result.columns:
        if df_result[i].max() - df_result[i].min() < 0.2*df_result[i].min():
            df_result.drop(columns = [i], axis=1, inplace=True)
    return df_result
    
def result_gf(df,labels): #show the result of clustering 
    a = ['Young','Old','Female','Male'] #Q1 and Q2 need to be treated specially
    b = []
    for i in df.columns:
        a.append(i)
        b.append(i)
    a.remove('Q1')
    a.remove('Q2')
    b.remove('Q1')
    b.remove('Q2')
    df_result = pd.DataFrame(0,columns = a,index = np.unique(labels)) 
    #create a data frame with rows number equal to the number of clusters and columns number equal to the original data frame
    for i in range(len(df)):
        for j in b:
            if df.loc[i,j] == 2:
                df_result.loc[labels[i],j]+=1
        if df.loc[i,'Q1'] < 15: #treate people younger than 15 as young
            df_result.loc[labels[i],'Young']+=1
        else:
            df_result.loc[labels[i],'Old']+=1
        if df.loc[i,'Q2'] == 2:
            df_result.loc[labels[i],'Female']+=1
        else:
            df_result.loc[labels[i],'Male']+=1

    col_list_dietary_behaviours = ['QN6','QN7','QN8','QN9','QN10']
    col_list_hygiene = ['QN11','QN12','QN13','QN14']
    col_list_injury = ['QN15','QN16','QN17','QN18','QN19','QN20','QN21']
    col_list_mental_health = ['QN22','QN23','QN24','QN25','QN26','QN27']
    col_list_tobacco_use = ['QN28','QN29','QN30','QN31','QN32','QN33']
    col_list_alcohol_use = ['QN34','QN35','QN36','QN37','QN38','QN39']
    col_list_drug_use = ['QN40','QN41','QN42','QN43']
    col_list_sexual_behaviours = ['QN44','QN45','QN46','QN47','QN48']
    col_list_physical_activity = ['QN49','QN50','QN51','QN52']
    col_list_protective_factors = ['QN53','QN54','QN55','QN56','QN57','QN58']
    df_result['dietary_behaviours'] = df_result[col_list_dietary_behaviours].sum(axis=1)
    df_result['hygiene'] = df_result[col_list_hygiene].sum(axis=1)
    df_result['injury'] = df_result[col_list_injury].sum(axis=1)
    df_result['mental_health'] = df_result[col_list_mental_health].sum(axis=1)
    df_result['tobacco_use'] = df_result[col_list_tobacco_use].sum(axis=1)
    df_result['alcohol_use'] = df_result[col_list_alcohol_use].sum(axis=1)
    df_result['drug_use'] = df_result[col_list_drug_use].sum(axis=1)
    df_result['sexual_behaviours'] = df_result[col_list_sexual_behaviours].sum(axis=1)
    df_result['physical_activity'] = df_result[col_list_physical_activity].sum(axis=1)
    df_result['protective_factors'] = df_result[col_list_protective_factors].sum(axis=1)
    a=[]
    for i in range(4,57):
        a.append(i)
    df_result.drop(df_result.columns[a], axis=1, inplace=True)
    
    for i in df_result.columns:
        if df_result[i].max() - df_result[i].min() < 0.2*df_result[i].min():
            df_result.drop(columns = [i], axis=1, inplace=True)
    
    return df_result
        

def elbow_kmode(df,int1,int2,name):
    cost = []
    K = range(int1,int2)
    init_type = ['Cao', 'Huang', 'random']
    for init in init_type:
        for num_clusters in list(K):
            kmode = KModes(n_clusters=num_clusters, init = init, n_init = 5, verbose=1)
            kmode.fit_predict(df)
            cost.append(kmode.cost_)

    fig, axs = plt.subplots(3)
    fig.suptitle('Elbow Method For Optimal k')
    axs[0].plot(K, cost[0:5],'bx-')
    axs[1].plot(K, cost[5:10],'bx-')
    axs[2].plot(K, cost[10:15],'bx-')
    fig.savefig("elbow_"+str(name)+".png")

    
def least_cost_kmode(df,int1):
    cost = []
    init_type = ['Cao', 'Huang', 'random']
    for init in init_type:
        kmode = KModes(n_clusters=int1, init = init, n_init = 5, verbose=1)
        kmode.fit_predict(df)
        cost.append(kmode.cost_)
    
    best_init_type = init_type[cost.index(np.min(cost))]
    kmode = KModes(n_clusters=int1, init = best_init_type, n_init = 5, verbose=1)
    kmode.fit_predict(df)
    return kmode
    
def hierarchical_clustering(df,int1):
    methods = ['ward','complete','average']
    df_hc = {}
    for m in methods:
        AgglomerativeClustering = sklearn.cluster.AgglomerativeClustering(n_clusters = int1,linkage=m).fit(df)
        df_hc[m] = result(df,AgglomerativeClustering.labels_)
        print('Silhouette Score(n='+str(int1)+') of '+m+ ' is ' +str(silhouette_score(df, AgglomerativeClustering.labels_)))
        plt.figure(figsize=(10, 7))  
        plt.title("Dendrograms") 
        dend = scipy.cluster.hierarchy.dendrogram(scipy.cluster.hierarchy.linkage(df, method=m))
        plt.savefig("Dendrograms_ite_"+m+".png")
    return df_hc

def DBSCAN_(df,int1,int2,int3,int4,num):
    df_db = {}
    #int1 is the lower bound of the min number of one cluster
    #int2 is the upper bound of the min number of one cluster
    #int3 is the lower bound of the maximum distance between two samples for one to be considered in one neighborhood
    #int4 is the upper bound of the maximum distance between two samples for one to be considered in one neighborhood
    #num is the pecent of samples that can be seen as noisy
    as in the neighborhood of the other.
    a = np.linspace(int1,int2,num = int2-int1+1)
    for i in a:
        DBSCAN = sklearn.cluster.DBSCAN(eps=int4,algorithm='auto',min_samples=i).fit(df_iterative)
        b = 0
        for j in DBSCAN.labels_:
            if j == -1:
                b+=1 
            if b>num%len(df):
                c = i-1
            break

    a = np.linspace(int3,int4,num = 100)
    d = []
    algorithm = ['auto', 'ball_tree', 'kd_tree', 'brute']
    for al in algorithm:
        for i in a:
            DBSCAN = sklearn.cluster.DBSCAN(eps=i,algorithm=al,min_samples=c).fit(df)
            b = 0
            for j in DBSCAN.labels_:
                if j == -1:
                    b+=1
            if b < 0.1*len(df):
                d.append(i)
            break
 
    for i in range(4):
        DBSCAN = sklearn.cluster.DBSCAN(eps=d[i],algorithm=algorithm[i],min_samples=c).fit(df)    
        df_db[algorithm[i]] = result_gf(df,DBSCAN.labels_)
    return df_db

    
def hierarchical_clustering_gf(df,int1):
#int1 is the number of clusters
    methods = ['ward','complete','single','average']
    df_hc = {}
    for m in methods:
        AgglomerativeClustering = sklearn.cluster.AgglomerativeClustering(n_clusters = int1,linkage=m).fit(df)
        df_hc[m] = result_gf(df,AgglomerativeClustering.labels_)
        print('Silhouette Score(n='+str(int1)+') of '+m+ ' is ' +str(silhouette_score(df, AgglomerativeClustering.labels_)))
        plt.figure(figsize=(10, 7))  
        plt.title("Dendrograms") 
        dend = scipy.cluster.hierarchy.dendrogram(scipy.cluster.hierarchy.linkage(df, method=m))
        plt.savefig("Dendrograms_"+m+".png")
    return df_hc

