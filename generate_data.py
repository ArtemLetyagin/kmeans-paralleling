import numpy as np
import sys
import random

if __name__ == "__main__":
    N_clusters, N_features = [int(arg) for arg in sys.argv[1:]]
    N_samples = 10000 // N_clusters
    X = []
    for _ in range(N_clusters):
        m = random.randint(0, 10)
        loc = random.randint(1, 5)
        X.append(np.random.uniform(m, loc, (N_samples, N_features)))
    X = np.concatenate(X)
    np.random.shuffle(X)
    np.save("data.npy", X)
   