import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import time

# +-------------------------+
# |      Finding labels     |
# +-------------------------+
def dist_and_labels(args):
    data, centroids = args
    dist = np.power(np.expand_dims(data, 1) - centroids, 2).sum(2)
    labels = np.argmax(dist, 1)
    return labels


# +-------------------------+
# |    Updating centroids   |
# +-------------------------+
def update_centroids(centroids, labels):
    new_centroids = []
    for clust_num in range(n_clusters):
        new_centroid = data[labels == clust_num].mean(0)
        new_centroids.append(new_centroid)
    new_centroids = np.array(new_centroids)
    
    same_centroids_num = 0
    for centroid in new_centroids:
        for j in range(len(centroids)):
            if np.any(centroid == centroids[j]):
                same_centroids_num += 1
    
    return new_centroids, same_centroids_num


# +-------------------------+
# |    Timer - decorator    |
# +-------------------------+
def calc_time(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        func(*args, **kwargs)
        return time.time() - start
    return wrapper


# +-------------------------+
# |   Non-parallel Kmeans   |
# +-------------------------+
@calc_time
def fit_kmeans(data, n_clusters, max_iters):
    num_iters = 0
    centroids = np.random.uniform(data.min(0), data.max(0), (n_clusters, data.shape[1]))

    while num_iters < max_iters:

        labels = dist_and_labels((data, centroids))
    
        new_centroids, same_centroids_num = update_centroids(centroids, labels)
            
        if same_centroids_num == n_clusters:
            print("Number or iters: ", num_iters)
            return
        centroids = new_centroids
        num_iters += 1
        # if num_iters % 50 == 0:
        #     print(num_iters) 
    print("Iterarion reached limit")


# +-------------------------+
# |     parallel Kmeans     |
# +-------------------------+
@calc_time
def fit_kmeans_parallel(data, n_clusters, max_iters, n_processes):
    num_iters = 0
    centroids = np.random.uniform(data.min(0), data.max(0), (n_clusters, data.shape[1]))

    chunks = np.array_split(data, n_processes)

    with mp.Pool(n_processes) as pool:
        while num_iters < max_iters:
            
            labels = pool.map(dist_and_labels, [(chunk, centroids) for chunk in chunks])
            labels = np.concatenate(labels)

            new_centroids, same_centroids_num = update_centroids(centroids, labels)
            
            if same_centroids_num == n_clusters:
                print("Number or iters: ", num_iters)
                return
            centroids = new_centroids
            num_iters += 1
            
    pool.close()
    pool.join()


# +-------------------------+
# |      MAIN FUNCTION      |
# +-------------------------+
if __name__ == "__main__":
    data = np.random.uniform(0, 1, (10000, 10))
    n_clusters = 3
    max_iters = 1000
    
    T = fit_kmeans(data, n_clusters, max_iters)
    print(f"Non parallel time: {T:.3f} s.")

    T_i = []
    for i in range(1, 16):
        t_i = fit_kmeans_parallel(data, n_clusters, max_iters, i)
        print(f"N processes = {i} time: {t_i:.3f} s.")
        T_i.append(t_i)
        
    plt.plot(np.arange(1, len(T_i)+1), T_i, label="parallel", marker='.')
    plt.axhline(y=T, color='r', linestyle='-', label="non parallel")
    plt.legend()
    plt.xlabel("Number of processes")
    plt.ylabel("Time in seconds")
    plt.savefig("result.png")
