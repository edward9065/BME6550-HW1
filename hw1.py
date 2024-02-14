from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib as plt

def get_data(filename):
    with open(filename, 'r') as f:
        first_row = f.readline().strip()

    num_columns = len(first_row.split('\t'))
    
    data = np.genfromtxt(filename, delimiter='\t', usecols=range(2, num_columns))

def cluster(filename, num_clusters):
    data = get_data(filename)

    kmeans_clusters = KMeans(n_clusters=num_clusters).fit_predict(data)
    gm_clusters = GaussianMixture(n_components=num_clusters).fit_predict(data)

    return kmeans_clusters, gm_clusters
    

def get_ground_truth_labels(filename):
     return np.genfromtxt(filename, delimiter='\t', usecols=1)

def get_rand_score(ground_truth_labels, predicted_labels):
    return adjusted_rand_score(ground_truth_labels, predicted_labels)

cho_kmeans, cho_gm = cluster("cho.txt", 5)
cho_gt = get_ground_truth_labels("cho.txt")
cho_kmeans_rand_score = get_rand_score(cho_gt, cho_kmeans)
cho_gm_rand_score = get_rand_score(cho_gt, cho_gm)
print(cho_kmeans_rand_score)
print(cho_gm_rand_score)

iyer_kmeans, iyer_gm = cluster("iyer.txt", 10)
iyer_gt = get_ground_truth_labels("iyer.txt")
iyer_kmeans_rand_score = get_rand_score(iyer_gt, iyer_kmeans)
iyer_gm_rand_score = get_rand_score(iyer_gt, iyer_gm)
print(iyer_kmeans_rand_score)
print(iyer_gm_rand_score)

pca = PCA(n_components=2)
cho_pca = pca.fit_transform(get_data("cho.txt"))
iyer_pca = pca.fit_transform(get_data("iyer.txt"))