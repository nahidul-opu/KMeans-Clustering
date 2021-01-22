# KMeans-Clustering
Python Implementation of KMeans Clustering algorithm based on Euclidean distance. \
Accepts Pandas DataFrame. \
Api is similar to Scikit-learn. 

## Example
from KMeansClustering import KMeans as KMn \
X, y = make_blobs(n_samples=100, centers=5, n_features=20, random_state=0) \
sample = pd.DataFrame(X) 

kmn = KMn(5) \
kmn.fit(sample) \
print(kmn.predict(X)) \
print("Labels :{}".format(kmn.labels)) \
print("Centers :{}".format(kmn.centers)) \
print("Number of iteration = {}".format(kmn.n_iteration)) 
