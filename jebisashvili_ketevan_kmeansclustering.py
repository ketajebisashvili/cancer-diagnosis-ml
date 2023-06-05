import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from minisom import MiniSom
import som 
import numpy as np 
from sklearn.model_selection import train_test_split

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import ParameterGrid
from sklearn.cluster import KMeans

# Load the CSV file into a DataFrame
data = pd.read_csv('Cancer_Data.csv')


X = data.iloc[:, 1:31] 

y = data.iloc[:, 0]

tree = DecisionTreeClassifier(random_state=42)
tree.fit(X, y)
importances = tree.feature_importances_
sorted_importances = sorted(zip(importances, X.columns), reverse=True)
for importance, feature in sorted_importances:
    print(f"{feature}: {importance}")

# Create a mask to select the features with importance > 0.1
mask = importances > 0.01

# Select the features based on the mask
selected_features = X.loc[:, mask]

X = selected_features

print(X)

# checking data shape
cancerdata = pd.concat([selected_features, y], axis=1)
row, col = cancerdata.shape
print(f'There are {row} rows and {col} columns') 
# >> There are 178 rows and 13 columns

print(cancerdata.head(10))

cancerdata_scaled = X.copy()

cancerdata_scaled[cancerdata_scaled.columns] = StandardScaler().fit_transform(cancerdata_scaled)
print(cancerdata_scaled.describe())

pca_2 = PCA(n_components=2)
pca_2_result = pca_2.fit_transform(cancerdata_scaled)
print('Explained variation per principal component: {}'.format(pca_2.explained_variance_ratio_))

# >> Explained variation per principal component:  [0.50565285 0.19554243]

print('Cumulative variance explained by 2 principal components: {:.2%}'.format(np.sum(pca_2.explained_variance_ratio_)))

# >> Cumulative variance explained by 2 principal components: 70.12%
    # Results from pca.components_
dataset_pca = pd.DataFrame(abs(pca_2.components_), columns=cancerdata_scaled.columns, index=['PC_1', 'PC_2'])
print('\n\n', dataset_pca)
print("\n*************** Most important features *************************")
print('As per PC 1:\n', (dataset_pca[dataset_pca > 0.3].iloc[0]).dropna())   
print('\n\nAs per PC 2:\n', (dataset_pca[dataset_pca > 0.3].iloc[1]).dropna())
print("\n******************************************************************")
# candidate values for our number of cluster
parameters = [2, 3, 4, 5, 10, 15, 20, 25, 30, 35, 40]
# instantiating ParameterGrid, pass number of clusters as input
parameter_grid = ParameterGrid({'n_clusters': parameters})
best_score = -1
kmeans_model = KMeans()     # instantiating KMeans model
silhouette_scores = []
# evaluation based on silhouette_score
for p in parameter_grid:
    kmeans_model.set_params(**p)    # set current hyper parameter
    kmeans_model.fit(data)          # fit model on cancer dataset, this will find clusters based on parameter p
    ss = metrics.silhouette_score(data, kmeans_model.labels_)   # calculate silhouette_score
    silhouette_scores += [ss]       # store all the scores
    print('Parameter:', p, 'Score', ss)
    # check p which has the best score
    if ss > best_score:
        best_score = ss
        best_grid = p
# plotting silhouette score
plt.bar(range(len(silhouette_scores)), list(silhouette_scores), align='center', color='#722f59', width=0.5)
plt.xticks(range(len(silhouette_scores)), list(parameters))
plt.title('Silhouette Score', fontweight='bold')
plt.xlabel('Number of Clusters')
plt.show()

# fitting KMeans    
kmeans = KMeans(n_clusters=2) 
kmeans.fit(cancerdata_scaled)
centroids = kmeans.cluster_centers_
centroids_pca = pca_2.transform(centroids)

""" Visualizing the clusters

:param pca_result: PCA applied data
:param label: K Means labels
:param centroids_pca: PCA format K Means centroids
 """
    # ------------------ Using Matplotlib for plotting-----------------------
x = pca_2_result[:, 0]
y = pca_2_result[:, 1]

plt.scatter(x, y, c=kmeans.labels_, alpha=0.5, s=200)  # plot different colors per cluster
plt.title('cancer clusters')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')

plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], marker='X', s=200, linewidths=1.5,
                color='red', edgecolors="black", lw=1.5)

plt.show()