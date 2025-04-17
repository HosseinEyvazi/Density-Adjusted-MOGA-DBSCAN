from sklearn.metrics import euclidean_distances
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from evaluate import *
import pandas as pd


def getMinpts(matrix, eps):
    """
    :param matrix: distance matrix
    :param eps:
    :return: minpts (as int)
    """
    return int(np.round(np.sum(matrix < eps) / len(matrix)))


def paramterAdaptive(data):
    N = len(data)
    distances = np.sort(euclidean_distances(data), axis=1)
    Deps = np.mean(distances[:, 1:N], axis=0)
    db = DBSCAN(eps=Deps[1], min_samples=getMinpts(distances, Deps[1])).fit(data)
    labels = db.labels_
    clusters = len(set(labels)) - (1 if -1 in labels else 0)
    counter = 1
    for i in range(2, len(data) - 2):
        db = DBSCAN(eps=Deps[i], min_samples=getMinpts(distances, Deps[i])).fit(data)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters_ == clusters:
            counter += 1
            if counter < BALANCENUM:
                continue
            else:
                best_clusters = n_clusters_
                left = i
                right = len(data) - 2
                while right - left > 1:
                    mid = int(np.floor((left + right) / 2))
                    db = DBSCAN(eps=Deps[mid], min_samples=getMinpts(distances, Deps[mid])).fit(data)
                    labels = db.labels_
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters < best_clusters:
                        right = mid
                    else:
                        left = mid
                final_index = mid if n_clusters == best_clusters else left
                print(final_index)
                return Deps[final_index], getMinpts(distances, Deps[final_index])
        else:
            clusters = n_clusters_
            counter = 1


def scoreAdaptive(data):
    N = len(data)
    distances = np.sort(euclidean_distances(data), axis=1)
    Deps = np.mean(distances[:, 1:N], axis=0)
    clusters = []
    silhouette_score = []
    for i in range(len(Deps) - 2):
        db = DBSCAN(eps=Deps[i], min_samples=getMinpts(distances, Deps[i])).fit(data)
        labels = db.labels_
        cluster = len(set(labels)) - (1 if -1 in labels else 0)
        if cluster == 1:
            break
        clusters.append(cluster)
        silhouette_score.append(metrics.silhouette_score(data, labels))
    index = np.argmax(silhouette_score)
    print(index)
    x = range(len(clusters))
    plt.scatter(x, clusters)
    plt.xlabel("K")
    plt.ylabel("clusters numbers")
    plt.show()
    plt.scatter(Deps[:len(clusters)], clusters)
    plt.xlabel("eps")
    plt.ylabel("clusters numbers")
    plt.show()
    plt.scatter(x[10:], clusters[10:])
    plt.xlabel("K")
    plt.ylabel("clusters numbers")
    plt.show()
    plt.scatter(x, silhouette_score)
    plt.xlabel("K")
    plt.ylabel("silhouette_score")
    plt.show()
    return Deps[index], getMinpts(distances, Deps[index])


def makeGraph(data, labels_true):
    N = len(data)
    distances = np.sort(euclidean_distances(data), axis=1)
    Deps = np.mean(distances[:, 1:N], axis=0)
    clusters = []
    silhouette_score = []
    for i in range(1, len(Deps)):
        db = DBSCAN(eps=Deps[i], min_samples=getMinpts(distances, Deps[i])).fit(data)
        labels = db.labels_
        cluster = len(set(labels)) - (1 if -1 in labels else 0)
        clusters.append(cluster)
        silhouette_score.append(metrics.v_measure_score(labels_true, labels))
    print(clusters)
    x = range(len(clusters))
    plt.scatter(x, clusters, color="black", linewidth=1.0)
    plt.xlabel("K")
    plt.ylabel("clusters numbers")
    plt.show()
    plt.plot(x, clusters, linewidth=3.0)
    plt.annotate(r'best index', xy=(330, clusters[330]), xycoords='data',
                 xytext=(+10, +30), textcoords='offset points', fontsize=12,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.xlabel("index")
    plt.ylabel("clusters numbers")
    plt.savefig("clusternum.png", dpi=720)
    plt.show()
    plt.plot(x, silhouette_score, linewidth=3.0)
    plt.annotate(r'best index', xy=(330, silhouette_score[330]), xycoords='data',
                 xytext=(30, -50), textcoords='offset points', fontsize=12,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.xlabel("index")
    plt.ylabel("v_measure_score")
    plt.savefig("score.png", dpi=720)
    plt.show()
    plt.scatter(x, silhouette_score)
    plt.xlabel("index")
    plt.ylabel("silhouette_score")
    plt.show()


data_path = "./dataset/Jain.txt"
BALANCENUM = 3

def load_data():
    data = np.loadtxt(data_path, delimiter=',')
    return data[:, 0:2], data[:, 2]

if __name__ == '__main__':
    start_time = time.time()

    data, labels_true = load_data()

    t = []
    num = 1
    for i in range(num):
        t0 = time.time()
        eps, minpts = paramterAdaptive(data)
        t.append(time.time() - t0)
    t = np.array(t)
    tmax = min(t)
    taver = t.sum() / num
    print('eps, minpts',eps, minpts)
    print("Shortest adaptive time:", tmax)
    print("Average adaptive time:", taver)
    y_pred = DBSCAN(eps=eps, min_samples=int(minpts)).fit_predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=y_pred)
    
    end_time = time.time()
    print('end_time - start_time',end_time - start_time)
    plt.show()
    