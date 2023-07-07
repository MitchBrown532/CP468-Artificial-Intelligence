import numpy as np

def kmeans(data, k, max_iterations=100):
    # Step 1: Initialize centroids
    centroids = data[:k]

    for _ in range(max_iterations):
        # Step 2: Assign each observation to the nearest centroid
        labels = assign_labels(data, centroids)

        # Step 3: Compute new centroids
        new_centroids = compute_centroids(data, labels, k)

        # Step 4: Check convergence
        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return labels

def assign_labels(data, centroids):
    """
    Assigns labels to each data point based on the nearest centroid.

    Args:
        data (numpy.ndarray): Input data points.
        centroids (list): Centroid positions.

    Returns:
        list: Cluster labels for each data point.
    """
    labels = []
    for point in data:
        # Calculate distances between the data point and each centroid
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        # Assign the label of the nearest centroid to the data point
        label = np.argmin(distances)
        labels.append(label)
    return labels

def compute_centroids(data, labels, k):
    """
    Computes new centroids by averaging the data points in each cluster.

    Args:
        data (numpy.ndarray): Input data points.
        labels (list): Cluster labels for each data point.
        k (int): Number of clusters.

    Returns:
        list: Updated centroid positions.
    """
    centroids = []
    for i in range(k):
        # Select the data points belonging to the current cluster
        cluster_points = data[labels == i]
        # Calculate the mean of the data points to obtain the new centroid position
        centroid = np.mean(cluster_points, axis=0)
        centroids.append(centroid)
    return centroids
