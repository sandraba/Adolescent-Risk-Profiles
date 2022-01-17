### Behavioral risk factors in Namibia (GSHS)

### Importing modules
import os
from sas7bdat import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from kmodes.kmodes import KModes
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score

### Importing GSHS Namibia data based on SAS7BDAT data file
### Downloaded from: https://www.cdc.gov/gshs/countries/africa/namibia.htm
cwd = os.getcwd()
full_path_1 = os.path.join(cwd, "nbh2013_public_use.sas7bdat")
sas_namibia = SAS7BDAT(full_path_1)
df_namibia = sas_namibia.to_data_frame()
#print(df_namibia)

### Preprocessing
# typecasting reduced datatset with categorical features for question items 7,8,9,10,11,12,13,14,15,16,29,30,35,38,41,43,52
df_namibia_reduc = df_namibia[["Q1", "Q2", "Q3", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15", "Q16", "Q29", "Q30", "Q38", "Q41", "Q43",
            "Q52"]]

print(df_namibia_reduc)

df_namibia_risk = df_namibia_reduc.rename(
    columns={"Q1": "age",
             "Q2": "sex",
             "Q3": "grade",
             "Q6" : "hunger",
             "Q7": "fruits_30d",
             "Q8": "veg_30d",
             "Q9": "soft_drinks_30d",
             "Q10": "fast_food_30d",
             "Q11": "brush_teeth_30d",
             "Q12": "wash_hands_eat_30d",
             "Q13": "wash_hands_toilet_30d",
             "Q14": "use_soap_30d",
             "Q15": "physically_attacked_12m",
             "Q16": "physical_fight_12m",
             "Q29": "smoke_cigarette_30d",
             "Q30": "use_tobacco_30d",
             "Q38": "times_drunk",
             "Q41": "times_marijuana_lifetime",
             "Q43": "times_amphetamines_lifetime",
             "Q52": "sitting_TV_computer_1d"})

print(df_namibia_risk)
feature_names = df_namibia_risk.columns.tolist()

# predictor variables


# Summary statistics with missing values



# Missing value imputation: random forest imputation
# Because: handles categorical variables automatically (like Bayes_Ridge) and does not need optimization (like knn)

df_namibia_risk.info()
print("Total number of missing values: ", df_namibia_risk.isnull().values.sum())
print("\n")
print("Total number of missing values per columns: \n", df_namibia_risk.isnull().sum())
print("\n")

# impute_cat = IterativeImputer(estimator=RandomForestClassifier(), initial_strategy='most_frequent', max_iter=100,
#                              random_state=0) # max_iter needs to be increased in final runs
# impute_cat.fit_transform(df_namibia_risk)
# df_namibia_impute = pd.DataFrame(impute_cat.transform(df_namibia_risk), columns=feature_names)
# print(df_namibia_impute)

cwd = os.getcwd()
print(cwd)
full_path_1a = os.path.join(cwd, "df_namibia_impute.csv")
print(full_path_1a)
df_namibia_impute = pd.read_csv(full_path_1a)
print(df_namibia_impute)

print("Total number of missing values after imputation: ", df_namibia_impute.isnull().values.sum())
print("\n")
print("Total number of missing values per columns after imputation: \n", df_namibia_impute.isnull().sum())
print("\n")

### Dichotomization

## Without imputation
# df_namibia_risk[["hungry"]] = np.where(df_namibia_risk[["hunger"]]>2,0,1)
df_namibia_risk["veg_fruits_bi"] = np.where((df_namibia_risk["fruits_30d"]+df_namibia_risk["veg_30d"])<5,0,1)
df_namibia_risk["soft_drinks_bi"] = np.where(df_namibia_risk["soft_drinks_30d"]>1,0,1)
df_namibia_risk["fast_food_bi"] = np.where(df_namibia_risk["fast_food_30d"]>1,0,1)
df_namibia_risk["brush_teeth_bi"] = np.where(df_namibia_risk["brush_teeth_30d"]<4,0,1)
df_namibia_risk["wash_hands_eating_bi"] = np.where(df_namibia_risk["wash_hands_eat_30d"]<5,0,1)
df_namibia_risk["wash_hands_toilet_bi"] = np.where(df_namibia_risk["wash_hands_toilet_30d"]<5,0,1)
# df_namibia_risk[["soap_usage"]] = np.where(df_namibia_risk[["use_soap_30d"]]<5,0,1)
df_namibia_risk["sitting_bi"] = np.where(df_namibia_risk["sitting_TV_computer_1d"]<4, 0, 1)
df_namibia_risk["amphetamines_bi"] = np.where(df_namibia_risk["times_amphetamines_lifetime"]<=1, 0, 1)
df_namibia_risk["marijuana_bi"] = np.where(df_namibia_risk["times_marijuana_lifetime"]<=1, 0, 1)
df_namibia_risk["drunk_bi"] = np.where(df_namibia_risk["times_drunk"]<=1, 0, 1)
df_namibia_risk["tobacco_bi"] = np.where(df_namibia_risk["use_tobacco_30d"]<=1, 0, 1)
df_namibia_risk["smoke_bi"] = np.where(df_namibia_risk["smoke_cigarette_30d"]<=1, 0, 1)
df_namibia_risk["fight_bi"] = np.where(df_namibia_risk["physical_fight_12m"]<=1, 0, 1)
df_namibia_risk["attack_bi"] = np.where(df_namibia_risk["physically_attacked_12m"]<=1, 0, 1)

risk_names = df_namibia_risk.iloc[:,-14:].columns.tolist()

## With imputation
df_namibia_impute["Fruits and vegs"] = np.where((df_namibia_risk["fruits_30d"]+df_namibia_risk["veg_30d"])<5,0,1)
df_namibia_impute["Soft drinks"] = np.where(df_namibia_impute["soft_drinks_30d"]>1,0,1)
df_namibia_impute["Fast food"] = np.where(df_namibia_impute["fast_food_30d"]>1,0,1)
df_namibia_impute["Brushing teeth"] = np.where(df_namibia_impute["brush_teeth_30d"]<4,0,1)
df_namibia_impute["Wash hands eating"] = np.where(df_namibia_impute["wash_hands_eat_30d"]<5,0,1)
df_namibia_impute["Wash hands toilet"] = np.where(df_namibia_impute["wash_hands_toilet_30d"]<5,0,1)
# df_namibia_risk[["soap_usage"]] = np.where(df_namibia_risk[["use_soap_30d"]]<5,0,1)
df_namibia_impute["Sitting"] = np.where(df_namibia_impute["sitting_TV_computer_1d"]<4, 0, 1)
df_namibia_impute["Amphetamines"] = np.where(df_namibia_impute["times_amphetamines_lifetime"]<=1, 0, 1)
df_namibia_impute["Marijuana"] = np.where(df_namibia_impute["times_marijuana_lifetime"]<=1, 0, 1)
df_namibia_impute["Drunk"] = np.where(df_namibia_impute["times_drunk"]<=1, 0, 1)
df_namibia_impute["Smokeless tobacco"] = np.where(df_namibia_impute["use_tobacco_30d"]<=1, 0, 1)
df_namibia_impute["Smoking"] = np.where(df_namibia_impute["smoke_cigarette_30d"]<=1, 0, 1)
df_namibia_impute["Fighting"] = np.where(df_namibia_impute["physical_fight_12m"]<=1, 0, 1)
df_namibia_impute["Attacking"] = np.where(df_namibia_impute["physically_attacked_12m"]<=1, 0, 1)

print("Total number of missing values after imputation: ", df_namibia_impute.isnull().values.sum())
print("\n")
print("Total number of missing values per columns after imputation: \n", df_namibia_impute.isnull().sum())
print("\n")

impute_features = df_namibia_impute.iloc[:,-14:].columns.tolist()

cwd = os.getcwd()
print(cwd)
full_path_2 = os.path.join(cwd, "Namibia_impute.csv")
df_namibia_impute.to_csv(full_path_2)

### Graph: overall risk
RF_overall = np.array((df_namibia_impute[impute_features].sum(axis=0)/len(df_namibia_impute))*100)
y = RF_overall.reshape(1,14)[0]
x = impute_features
color_scheme=['red', 'red', 'red', 'blue', 'blue', 'blue', 'yellow', 'brown', 'brown', 'brown', 'brown', 'brown', 'black', 'black']
overall_risk = plt.bar(x,y, color = color_scheme)
plt.axis('tight')
plt.ylabel("% at risk", fontsize=15)
plt.grid(False)
plt.xticks(rotation=90)
plt.ylim([0, 100])
plt.tight_layout()
plt.show()
plt.close()

### Training and testing datasets
X_namibia_train, X_namibia_test = train_test_split(df_namibia_impute, test_size=0.20)

cwd = os.getcwd()
print(cwd)
full_path_2a = os.path.join(cwd, "X_namibia_train.csv")
full_path_2b = os.path.join(cwd, "X_namibia_test.csv")
X_namibia_train.to_csv(full_path_2a)
X_namibia_test.to_csv(full_path_2b)

## K modes
## From https://www.analyticsvidhya.com/blog/2021/06/kmodes-clustering-algorithm-for-categorical-data/

cost = []
k = range(1, 134) ## upper bound of k range 2x sqrt(N) = 134
for num_clusters in list(k):
    kmode_namibia = KModes(n_clusters=num_clusters, init="random", n_init=5, verbose=1)
    kmode_namibia.fit_predict(X_namibia_train.iloc[:, -14:])
    cost.append(kmode_namibia.cost_)
plt.plot(k, cost, 'bx-')
plt.xlabel("Number of clusters", fontsize=15)
plt.ylabel('Cost', fontsize=15)
plt.show()

## Plot narrowing in on elbow
plt.plot(k, cost, 'bx-')
plt.xlabel("Number of clusters")
plt.xlim([0,50])
plt.ylabel('Cost')
plt.show()

kmode_best = KModes(n_clusters=8, init = "random", n_init = 5, verbose=1)
X_namibia_train["clusters_kmode"] = kmode_best.fit_predict(X_namibia_train.iloc[:, -14:])
print(X_namibia_train)

## RF profiles kmode_best

for c in range(0,8):
    df_c = X_namibia_train.iloc[:, -15:-1][X_namibia_train["clusters_kmode"] == c]
    RF_c = np.array((df_c[impute_features].sum(axis=0) / len(df_c)) * 100)
    y = RF_c.reshape(1, 14)[0]
    x = impute_features
    color_scheme = ['red', 'red', 'red', 'blue', 'blue', 'blue', 'yellow', 'brown', 'brown', 'brown', 'brown', 'brown',
                    'black', 'black']
    overall_risk = plt.bar(x, y, color=color_scheme)
    plt.axis('tight')
    plt.title("Cluster {}".format(c), fontsize=25)
    plt.ylabel("% at risk", fontsize=20)
    plt.grid(False)
    plt.xticks(rotation=90, fontsize=15)
    plt.yticks (fontsize=15)
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.show()
    plt.close()

## Bernoulli Mixture Models
cwd = os.getcwd()
print(cwd)
full_path_3 = os.path.join(cwd, "bmm_cluster_assignment.csv")
print(full_path_3)

impute_features.append("clusters_bmm")
df_bmm = pd.read_csv(full_path_3, header=None, names=impute_features)
df_bmm["clusters_bmm"] = df_bmm["clusters_bmm"]-1
impute_features=impute_features[0:14]

for c in range(0,7):
    df_c = df_bmm.iloc[:,:-1][df_bmm["clusters_bmm"] == c]
    RF_c = np.array((df_c[impute_features].sum(axis=0) / len(df_c)) * 100)
    y = RF_c.reshape(1, 14)[0]
    x = impute_features
    color_scheme = ['red', 'red', 'red', 'blue', 'blue', 'blue', 'yellow', 'brown', 'brown', 'brown', 'brown', 'brown',
                    'black', 'black']
    overall_risk = plt.bar(x, y, color=color_scheme)
    plt.axis('tight')
    plt.title("Cluster {}".format(c), fontsize=25)
    plt.ylabel("% at risk", fontsize=20)
    plt.grid(False)
    plt.xticks(rotation=90, fontsize=15)
    plt.yticks (fontsize=15)
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.show()
    plt.close()

# After logistic PCA, using the 5 first principal components
cwd = os.getcwd()
print(cwd)
full_path_10 = os.path.join(cwd, "namibia_pc_train.csv")
print(full_path_10)

cwd = os.getcwd()
print(cwd)
full_path_11 = os.path.join(cwd, "X_namibia_train.csv")
print(full_path_11)

df_pc_train = pd.read_csv(full_path_10, header=None, names=impute_features)
X_namibia_train = pd.read_csv(full_path_11)
X_namibia_train = X_namibia_train.iloc[:,-14:]
print(X_namibia_train)

print(df_pc_train.T.shape)
print(X_namibia_train.shape)
df_namibia_train_pc = X_namibia_train.dot(df_pc_train.T)

cwd = os.getcwd()
print(cwd)
full_path_12 = os.path.join(cwd, "namibia_train_pc.csv")
df_namibia_train_pc.to_csv(full_path_12)


## K means
ssd = []
silhouette = {}
k = range(1,3)
for clusters in k:
    kmeans = KMeans(n_clusters=clusters)
    kmeans.fit(df_namibia_train_pc)
    ssd.append(kmeans.inertia_)
    print(ssd)
    # kmeans.fit_predict(df_namibia_train_pc)
    # score = silhouette_score(df_namibia_train_pc, kmeans.labels_, metric="euclidian")
    # silhouette[clusters] = score
plt.plot(k,ssd, "bx-")
plt.xlabel("Values of k")
plt.ylabel("Sum of squared distances/Inertia")
plt.show()
# print(silhouette)

print(df_namibia_train_pc)
df_namibia_train_pc = df_namibia_train_pc.iloc[:,:-1]
print(df_namibia_train_pc)

kmeans_best = KMeans(n_clusters=7)
kmeans_best.fit(df_namibia_train_pc)
X_namibia_train["clusters_kmeans"] = kmeans_best.predict(df_namibia_train_pc)

for c in range(0,7):
    df_c = X_namibia_train.iloc[:,:-1][X_namibia_train["clusters_kmeans"] == c]
    RF_c = np.array((df_c[impute_features].sum(axis=0) / len(df_c)) * 100)
    y = RF_c.reshape(1, 14)[0]
    x = impute_features
    color_scheme = ['red', 'red', 'red', 'blue', 'blue', 'blue', 'yellow', 'brown', 'brown', 'brown', 'brown', 'brown',
                    'black', 'black']
    overall_risk = plt.bar(x, y, color=color_scheme)
    plt.axis('tight')
    plt.title("Cluster {}".format(c), fontsize=25)
    plt.ylabel("% at risk", fontsize=20)
    plt.grid(False)
    plt.xticks(rotation=90, fontsize=15)
    plt.yticks (fontsize=15)
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.show()
    plt.close()


df_namibia_test_pc["clusters_kmeans"] = kmeans_best.predict(df_namibia_test_pc)
print(df_namibia_train_pc)
score_best = silhouette_score(df_namibia_test_pc.iloc[:,:-1], df_namibia_test_pc["clusters_kmeans"], metric="euclidian")



## GMM
impute_features = impute_features[:-1]
X_namibia_train = X_namibia_train.iloc[:, :-1]
cwd = os.getcwd()
print(cwd)
full_path_15 = os.path.join(cwd, "gmm_cluster_assignment.csv")
print(full_path_15)

df_gmm = pd.read_csv(full_path_15, header=None)
df_gmm = df_gmm.iloc[:,-1:]
print(df_gmm)
X_namibia_train["clusters_gmm"] = df_gmm-1
print(X_namibia_train)

for c in range(0,9):
    df_c = X_namibia_train.iloc[:,:-1][X_namibia_train["clusters_gmm"] == c]
    RF_c = np.array((df_c[impute_features].sum(axis=0) / len(df_c)) * 100)
    y = RF_c.reshape(1, 14)[0]
    x = impute_features
    color_scheme = ['red', 'red', 'red', 'blue', 'blue', 'blue', 'yellow', 'brown', 'brown', 'brown', 'brown', 'brown',
                    'black', 'black']
    overall_risk = plt.bar(x, y, color=color_scheme)
    plt.axis('tight')
    plt.title("Cluster {}".format(c), fontsize=25)
    plt.ylabel("% at risk", fontsize=20)
    plt.grid(False)
    plt.xticks(rotation=90, fontsize=15)
    plt.yticks (fontsize=15)
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.show()
    plt.close()













print(df_namibia_train_pc)
df_namibia_train_pc = df_namibia_train_pc.iloc[:,:-1]
print(df_namibia_train_pc)

kmeans_best = KMeans(n_clusters=7)
kmeans_best.fit(df_namibia_train_pc)
X_namibia_train["clusters_kmeans"] = kmeans_best.predict(df_namibia_train_pc)


for c in range(0,9):
    df_c = df_gmm.iloc[:,:-1][df_gmm["clusters_gmm"] == c]
    RF_c = np.array((df_c[impute_features].sum(axis=0) / len(df_c)) * 100)
    y = RF_c.reshape(1, 14)[0]
    x = impute_features
    color_scheme = ['red', 'red', 'red', 'blue', 'blue', 'blue', 'yellow', 'brown', 'brown', 'brown', 'brown', 'brown',
                    'black', 'black']
    overall_risk = plt.bar(x, y, color=color_scheme)
    plt.axis('tight')
    plt.title("Cluster {}".format(c), fontsize=25)
    plt.ylabel("% at risk", fontsize=20)
    plt.grid(False)
    plt.xticks(rotation=90, fontsize=15)
    plt.yticks (fontsize=15)
    plt.ylim([0, 100])
    plt.tight_layout()
    plt.show()
    plt.close()





## Hierarchical


