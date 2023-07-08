import numpy as np
import pandas as pd
import seaborn as sns
import random
import matplotlib.pyplot as plt


import numpy as np
import random

# Step 1: Define the number of clusters (K) and max iterations
K = 2
max_iterations = 100

# Step 2: Load the dataset
df = pd.read_csv("kmeans.csv")
data = df.values

# Step 3: Initialize centroids randomly from the data points
centroids = data[random.sample(range(len(data)), K)]

# Step 4: Create an array to store the assigned cluster for each data point
assignments = np.zeros(len(data))

# Step 5: Implement the K-means algorithm
for _ in range(max_iterations):
    # Assign each data point to the nearest centroid
    for i, point in enumerate(data):
        distances = np.linalg.norm(centroids - point, axis=1)
        assignments[i] = np.argmin(distances)

    # Update the centroids
    for k in range(K):
        cluster_points = data[assignments == k]
        centroids[k] = np.mean(cluster_points, axis=0)

# Step 6: Print the results
print("Centroids:")
print(centroids)
print("Cluster Assignments:")
print(assignments)

# Step 7: Plot the results
plt.scatter(data[:, 0], data[:, 1], c=assignments)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x')
plt.xlabel('x')
plt.ylabel('y')
plt.title('K-means Clustering')
plt.show()