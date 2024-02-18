# %%
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

def get_data(filename):
    return np.genfromtxt(filename, delimiter='\t')

def cluster(data, num_clusters):
    kmeans_clusters = KMeans(n_clusters=num_clusters).fit_predict(data[:,2:])
    gm_clusters = GaussianMixture(n_components=num_clusters).fit_predict(data[:,2:])

    return kmeans_clusters, gm_clusters
# %%
cho_data = get_data("cho.txt")
cho_kmeans, cho_gm = cluster(cho_data[:,2:], 5)
cho_gt = cho_data[:,1]
cho_kmeans_rand_score = adjusted_rand_score(cho_gt, cho_kmeans)
cho_gm_rand_score = adjusted_rand_score(cho_gt, cho_gm)
print(f"cho.txt K-means Rand score: {cho_kmeans_rand_score}")
print(f"cho.txt GMM Rand score: {cho_gm_rand_score}")

iyer_data = get_data("iyer.txt")
iyer_data = iyer_data[iyer_data[:,1]!=-1]
iyer_kmeans, iyer_gm = cluster(iyer_data[:,2:], 10)
iyer_gt = iyer_data[:,1]
iyer_kmeans_rand_score = adjusted_rand_score(iyer_gt, iyer_kmeans)
iyer_gm_rand_score = adjusted_rand_score(iyer_gt, iyer_gm)
print(f"iyer.txt K-means Rand score: {iyer_kmeans_rand_score}")
print(f"iyer.txt GMM Rand score: {iyer_gm_rand_score}")

pca = PCA(n_components=2)
cho_pca = pca.fit_transform(cho_data[:,2:])
iyer_pca = pca.fit_transform(iyer_data[:,2:])

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

axs[0,0].scatter(cho_pca[:,0], cho_pca[:,1], c=cho_data[:,1], cmap="viridis")
axs[0,0].set_title("Ground Truth of cho.txt")

axs[0,1].scatter(cho_pca[:,0], cho_pca[:,1], c=cho_kmeans, cmap="viridis")
axs[0,1].set_title("K-Means on cho.txt")
axs[0,1].annotate(f"Rand Index {round(cho_kmeans_rand_score, 4)}", (4,4))

axs[0,2].scatter(cho_pca[:,0], cho_pca[:,1], c=cho_gm, cmap="viridis")
axs[0,2].set_title("Gaussian Mixtures on cho.txt")
axs[0,2].annotate(f"Rand Index {round(cho_gm_rand_score, 4)}", (4,4))



axs[1,0].scatter(iyer_pca[:,0], iyer_pca[:,1], c=iyer_data[:,1], cmap="viridis")
axs[1,0].set_title("Ground Truth of iyer.txt")
# remove the one point that squishes the graph by alot
axs[1,0].set_xlim(-10,40)
axs[1,0].set_ylim(-20,5)

axs[1,1].scatter(iyer_pca[:,0], iyer_pca[:,1], c=iyer_kmeans, cmap="viridis")
axs[1,1].set_title("K-Means on iyer.txt")
axs[1,1].set_xlim(-10,40)
axs[1,1].set_ylim(-20,5)
axs[1,1].annotate(f"Rand Index {round(iyer_kmeans_rand_score, 4)}", (22,4))


axs[1,2].scatter(iyer_pca[:,0], iyer_pca[:,1], c=iyer_gm, cmap="viridis")
axs[1,2].set_title("Gaussian Mixtures on iyer.txt")
axs[1,2].set_xlim(-10,40)
axs[1,2].set_ylim(-20,5)
axs[1,2].annotate(f"Rand Index {round(iyer_gm_rand_score, 4)}", (22,4))


# plt.show()
plt.savefig("plot.png")
# %%