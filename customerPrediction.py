import plotly.express as px
import plotly.graph_objects as pg
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

data = pd.read_csv('archive/marketing_campaign.csv', sep="\t")
print("Number of rows:", len(data))
print("Number of columns:", data.shape[1])
data.head()
print("")

# Dataset Summary
data.info()
print("")

# ------- DATA CLEANING --------

# Removing Null Values
data = data.dropna()
print("The shape of the dataset after removing null values is : ", data.shape)
print("")

#copy of data to be used for clustering
data2 = data

# ----------------------------------------------------
# ---------------- DATA VISUALIZATION ----------------
# ----------------------------------------------------

# Parsing Objects as datetime
print("Data type of Dt_Customer column before parsing : ", data["Dt_Customer"].dtypes)
data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])
print("Data type of Dt_Customer column after parsing : ", data["Dt_Customer"].dtypes)
print(data["Dt_Customer"].head())
print("")


# Eploring Income Column
print("Maximum Yearly Income : ",max(data["Income"]))
print("Minimum Yearly Income : ",min(data["Income"]))
print("")

# Visualizing Yearly income by Histogram
fig = px.histogram(data, x="Income", title="CUSTOMER'S YEARLY INCOME")
fig.show()

# Exploring Kidhome column
print("Maximum number of children in customer's house : ",max(data["Kidhome"]))
print("Minimum number of children in customer's house : ",min(data["Kidhome"]))
print("")

# visualizing No. of kids in customer's house by histogram
fig = px.histogram(data, x="Kidhome", title="NUMBER OF KIDS IN CUSTOMER'S HOUSE", nbins=5, color_discrete_sequence=["red"])
fig.show()

# Exploring Teenhome column
print("Maximum number of teens in customer's house : ",max(data["Teenhome"]))
print("Minimum number of teens in customer's house : ",min(data["Teenhome"]))
print("")

# Visualizing no. of teens in customer's house by histogram
fig = px.histogram(data, x="Teenhome", title="NUMBER OF TEENS IN CUSTOMER'S HOUSE", nbins=5, color_discrete_sequence=["darkblue"])
fig.show()

# Exploring Dt_customer column
print("Newest customer enrollment with the company : ",max(data["Dt_Customer"]).date())
print("Oldest customer enrollment with the company : ",min(data["Dt_Customer"]).date())
print("")

# Visualizing monthly Customer Enrolment by histogram
fig = px.histogram(data, x="Dt_Customer", title="MONTHLY CUSTOMER ENROLLMENTS", color_discrete_sequence=["royalblue"])
fig.show()

# Visualizing yearly Customer Enrolment by histogram
fig = px.histogram(data, x="Dt_Customer", title="YEARLY CUSTOMER ENROLLMENTS", nbins=3, color_discrete_sequence=["blueviolet"])
fig.show()

# Exploring Recency Column
print("Maximum number of days since customer's last purchase : ",max(data["Recency"]))
print("Minimum number of days since customer's last purchase : ",min(data["Recency"]))
print("")

# Visualizing Customer's last purchase by histogram
fig = px.histogram(data, x="Recency", title="NUMBER OF DAYS SINCE CUSTOMER'S LAST PURCHASE", 
                   color_discrete_sequence=["dodgerblue"])
fig.show()

# Exploring mntwines column
print("Maximum amount spent on wines by the customers : ",max(data["MntWines"]))
print("Minimum amount spent on wines by the customers : ",min(data["MntWines"]))
print("")

# Visualizing amount spent on wines by histogram
fig = px.histogram(data, x="MntWines", title="AMOUNT SPENT ON WINES", 
                   color_discrete_sequence=["coral"])
fig.show()

# Exploring mntfruits column
print("Maximum amount spent on fruits by the customers : ",max(data["MntFruits"]))
print("Minimum amount spent on fruits by the customers : ",min(data["MntFruits"]))
print("")

# Visualizing amount spent on fruits by histogram
fig = px.histogram(data, x="MntFruits", title="AMOUNT SPENT ON FRUITS", 
                   color_discrete_sequence=["fuchsia"])
fig.show()

# Exploring Mntmeatproducts column
print("Maximum amount spent on meat products by the customers : ",max(data["MntMeatProducts"]))
print("Minimum amount spent on meat products by the customers : ",min(data["MntMeatProducts"]))
print("")

# Visualizing amount spent on meat products by histogram
fig = px.histogram(data, x="MntMeatProducts", title="AMOUNT SPENT ON MEAT PRODUCTS", 
                   color_discrete_sequence=["orange"])
fig.show()

# Exploring Mntfishproducts column
print("Maximum amount spent on fish products by the customers : ",max(data["MntFishProducts"]))
print("Minimum amount spent on fish products by the customers : ",min(data["MntFishProducts"]))
print("")

# Visualizing amount spent on fish products
fig = px.histogram(data, x="MntFishProducts", title="AMOUNT SPENT ON FISH PRODUCTS", 
                   color_discrete_sequence=["mediumorchid"])
fig.show()

# Exploring Mntsweetproducts column
print("Maximum amount spent on sweet products by the customers : ",max(data["MntSweetProducts"]))
print("Minimum amount spent on sweet products by the customers : ",min(data["MntSweetProducts"]))
print("")

# Visualizing Amount spent on sweet products by histogram
fig = px.histogram(data, x="MntSweetProducts", title="AMOUNT SPENT ON SWEET PRODUCTS", 
                   color_discrete_sequence=["mediumslateblue"])
fig.show()

# Exploring Mntgoldproducts column
print("Maximum amount spent on gold products by the customers : ",max(data["MntGoldProds"]))
print("Minimum amount spent on gold products by the customers : ",min(data["MntGoldProds"]))
print("")

# Visualizing Amount spent on gold products by histogram
fig = px.histogram(data, x="MntGoldProds", title="AMOUNT SPENT ON GOLD PRODUCTS", 
                   color_discrete_sequence=["mediumvioletred"])
fig.show()

# -------- REVEALING THE CATEGORICAL FEATURES -----------

# Exploring Education Column
print("Total categories in Education column : ")
edu = data.pivot_table(index = ['Education'], aggfunc = 'size') 
edu = edu.reset_index()
edu.columns= ["Qualifications", "Counts"]
edu.sort_values("Counts", ascending = False, inplace = True)
print(edu)
print("")

# Visualizing Qualifications by donut plot
fig = plt.figure(figsize = (12, 13)) 
circle = plt.Circle( (0,0), 0.5, color = 'white')
plt.pie(edu["Counts"], labels = edu["Qualifications"])
p=plt.gcf()
p.gca().add_artist(circle)
plt.legend(edu["Counts"])
plt.title("QUALIFICATIONS", fontsize=35)
plt.show() 

# Exploring Marital_status column
print("Total categories in Marital_Status column : ")
ms = data.pivot_table(index = ['Marital_Status'], aggfunc = 'size') 
ms = ms.reset_index()
ms.columns= ["Marital_Status", "Counts"]
ms = ms.sort_values("Counts", ascending = False)
print(ms)
print("")

# ----------- FEATURE ENGINEERING ------------

# "Year_Birth" column contains customer's birth year when
# on subtracting it from the current year gives customer's age.
#Customer's Age
data["Age"] = 2021-data["Year_Birth"]
data["Age"].head()
# Exploring Age column
print("Maximum age of the customer : ",max(data["Age"]))
print("Minimum age of the customer : ",min(data["Age"]))
print("")

# Visualizing customer's age by histogram
fig = px.histogram(data, x="Age", title="CUSTOMER'S AGE", color_discrete_sequence=["royalblue"])
fig.show()

# On adding the amount spent on wines, fruits, meat, fish, sweet and gold products,
# we get the customer's total expenses
# Customer's Total Expenses
data["Expenses"] = data["MntWines"]+ data["MntFruits"]+ data["MntMeatProducts"]+ data["MntFishProducts"]+ data["MntSweetProducts"]+ data["MntGoldProds"]
data["Expenses"].head()
# Exploring expense column
print("Customer's Maximum total expenses : ",max(data["Expenses"]))
print("Customer's Minimum total expenses : ",min(data["Expenses"]))
print("")

# Visualizing customer's total expense by histogram
fig = px.histogram(data, x="Expenses", title="CUSTOMER'S TOTAL EXPENSES", color_discrete_sequence=["red"])
fig.show()

# Dividing Customer's Marital_Status into Married and Single Relationship_Status
data["Relationship_Status"]=data["Marital_Status"].replace({"Married":"Married", "Together":"Married",
                                                    "Absurd":"Single", "Widow":"Single", "YOLO":"Single", 
                                                    "Divorced":"Single", "Single":"Single", "Alone":"Single"})
data["Relationship_Status"].head()
# Exploring Relationship ststus column
print("Total categories in Relationship_Status column : ")
rs = data.pivot_table(index = ['Relationship_Status'], aggfunc = 'size') 
rs = rs.reset_index()
rs.columns= ["Relationship_Status", "Counts"]
rs = rs.sort_values("Counts", ascending = False)
print(rs)
print("")

# Visualizing Relationship status by donut plot
fig = plt.figure(figsize = (12, 13)) 
circle = plt.Circle( (0,0), 0.5, color = 'white')
plt.pie(rs["Counts"], labels = rs["Relationship_Status"])
p=plt.gcf()
p.gca().add_artist(circle)
plt.legend(rs["Counts"])
plt.title("Relationship_Status", fontsize=35)
plt.show()

#Calculating total number of purchases by adding website, catalogue and store purchases
data["Total_Purchases"] = data["NumDealsPurchases"]+data["NumWebPurchases"] + data["NumCatalogPurchases"] + data["NumStorePurchases"]
data["Total_Purchases"].head()
#EXPLORING TOTAL_PURCHASES COLUMN
print("Maximum number of purchases made by the customers : ",max(data["Total_Purchases"]))
print("Minimum number of purchases made by the customers : ",min(data["Total_Purchases"]))
print("")

# Visualizing Total purchase made by customers
fig = px.histogram(data, y="Total_Purchases", title="TOTAL PURCHASES MADE BY THE CUSTOMERS", 
                   color_discrete_sequence=["coral"])
fig.show()


# ----------------------------------------------------
# ------------ APPLYING K-MEANS COUSTERING -----------
# ----------------------------------------------------

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









