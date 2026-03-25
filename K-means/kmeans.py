def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    dim = len(points[0])
    centroids = [[0.0] * dim for _ in range(k)]
    cluster_size = [0] * k

    for point, cluster in zip(points, assignments):
        for d in range(dim):
            centroids[cluster][d] += point[d]
        cluster_size[cluster] += 1

    for i in range(len(centroids)):
        if cluster_size[i] > 0:
            for d in range(dim):
                centroids[i][d] /= cluster_size[i]

    return centroids


def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    dim = len(points[0])
    num_centroids = len(centroids)
    num_points = len(points)
    assignments = [0] * num_points

    for i, point in enumerate(points):
        temp = [0.0] * num_centroids
        for j, centroid in enumerate(centroids):
            for d in range(dim):
                temp[j] += (point[d] - centroid[d]) ** 2
        assignments[i] = temp.index(min(temp))

    return assignments
