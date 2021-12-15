import numpy as np
import pandas as pd
import datetime
from datetime import date
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, normalize
from sklearn import metrics
from sklearn.mixture import GaussianMixture
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import ListedColormap
from sklearn import metrics
import warnings
import sys
import warnings
warnings.filterwarnings('ignore')

data2 = pd.read_csv('archive/marketing_campaign.csv', sep="\t")
data2 = data2.copy()
print("")

pd.set_option("max_columns", None)

# Clearing missing values
data2 = data2.dropna()
data2.isnull().sum()


# ------ PREPROCESSING -------

data2["Age"] = 2021 - data2["Year_Birth"]
data2["Total_Spend"] = data2["MntWines"] + data2["MntFruits"] + data2["MntMeatProducts"] + data2["MntFishProducts"] + data2["MntSweetProducts"] + data2['MntGoldProds']
data2["Dt_Customer"] = pd.to_datetime(data2["Dt_Customer"], dayfirst=True)
last_date = date(2014,7,1)
data2["T"] = pd.to_numeric(data2["Dt_Customer"].dt.date.apply(lambda x: (last_date - x)).dt.days,downcast="integer")
data2["Marital_Status"] = data2["Marital_Status"].replace({'Divorced':'Single', 'Single':'Single', 'Alone':'Single',
                                                        'Widow':'Single', 'Absurd':'Single', 'YOLO':'Single','Married':'Married','Together':'Married'})
data2["Education"] = data2["Education"].replace({'Basic':'Undergraduate', '2n Cycle':'Undergraduate', 'Graduation':'Postgraduate', 'Master':'Postgraduate', 'PhD':'Postgraduate'})
data2["Children"] = data2["Kidhome"] + data2["Teenhome"]
data2["Has_Child"] = np.where(data2.Children >0, 'Has child', 'No child')
data2 = data2.rename(columns={'MntWines': 'Wines', 'MntFruits':'Fruits',
                           'MntMeatProducts':'Meat', 'MntFishProducts':'Fish',
                           'MntSweetProducts':'Sweets', 'MntGoldProds':'Gold'})
drop_list = ['ID' ,'Dt_Customer', 'Z_CostContact', 'Z_Revenue', 'Year_Birth']
data2.drop(drop_list, axis=1 ,inplace=True)




le = LabelEncoder()
cat_cols = [col for col in data2.columns if data2[col].dtypes == "O"]
for i in cat_cols:
    data2[i] = le.fit_transform(data2[[i]])

# standardization process
df = data2.copy()

# leaving out variables like accepted campaigns
col_del = ['AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 'Response', 'Complain']
df = df.drop(col_del, axis=1)

# Scaling
scaler = StandardScaler()
scaler.fit(df)
scaled_df = pd.DataFrame(scaler.transform(df), columns=df.columns)
print("Scaled OK!")

scaled_df.head()

# PCA : is a technique that reduces the dimensionality of datasets, increases their
# interpretability, and also minimizes information loss.
# setting the number of dimensions as 3 and plotting the data2set
pca = PCA(n_components=3)
pca.fit(scaled_df)
pca_df = pd.DataFrame(pca.transform(scaled_df), columns=["columns1", "columns2", "columns3"])

pca_df.describe().T

x = pca_df["columns1"]
y = pca_df["columns2"]
z = pca_df["columns3"]

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c="maroon", marker="o")
plt.show()

# -------- CLUSTERING ---------

""" Performing the clustering process with these steps:

1. Determining the number of clusters to be created with the Elbow Method
2. Clustering with Gaussian Mixture Model
3. Visualization of created clusters """

# 1. Applying elbow method
elbow_method = KElbowVisualizer(KMeans(), k=10)
elbow_method.fit(pca_df)
elbow_method.show()

# 2. Clustering with Gaussian Mixture Model
gmm = GaussianMixture(n_components=4, covariance_type='spherical', max_iter=2000, random_state=42).fit(pca_df)
labels = gmm.predict(pca_df)

pca_df['Clusters'] = labels
data2['Clusters'] = labels

# 3. Plot the clusters
fig = plt.figure(figsize=(12,7))
ax = plt.subplot(111, projection='3d', label="bla")
ax.scatter(x, y, z, s=40, c=pca_df['Clusters'], marker="o", cmap="Paired_r")
plt.show()


# -------- MODEL EVALUATION ---------

cl = ['#FAD3AE', '#855E46', '#FE800F', '#890000']
plt.figure(figsize=(14,8))
sns.countplot(x=data2['Clusters'], palette=cl)

#plt.figure(figsize=(14,8))
#sns.jointplot(x=data2["Total_Spend"], y=data2["Income"], hue=data2["Clusters"], palette=cl);

# Interpreting the clusters according to the Income and Total Spend features:
"""
Group 0: low spend - low income
Group 1: high spend - average income
Group 2: high spend - high income
Group 3: low spend - average income """
plt.figure(figsize=(15,6))
sns.boxenplot(x="Clusters", y="Total_Spend", palette=cl, data=data2);

# the distribution of clusters by products
prod = ['Fish', 'Meat', 'Sweets', 'Wines', 'Gold']
for i in prod:
    plt.figure(figsize=(15,6))
    sns.boxenplot(x="Clusters", y=i, palette=cl ,data=data2);










