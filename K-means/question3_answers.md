# Homework 1 - Question 3 (K-means Clustering)

This file contains the **text answers** for Question 3 and summarizes the key numerical outputs from the existing implementation in `K-means/kmeans_numpy.py`.

## Part 1 (5 pts)

### 1(a) Why K-means is sensitive to initial centroids
K-means minimizes a **non-convex** objective (WCSS), so different initial centroids can lead to different local minima.  
If initialization starts near poor regions, clusters may be assigned suboptimally and the algorithm can converge to a worse solution even though it still converges quickly.

### 1(b) One mitigation strategy
A practical mitigation is **K-means++ initialization**:
1. Choose the first centroid randomly.
2. Choose each next centroid with probability proportional to squared distance from the closest chosen centroid.
3. Run standard K-means updates.

This spreads initial centroids apart and usually reduces bad local minima. Another common mitigation is multiple random restarts (`n_init`) and keeping the best WCSS.

## Part 2 (10 pts)

Part 2 was already implemented in `K-means/kmeans_numpy.py` (not rewritten).

- Dataset loading from `data_cluster.csv`: implemented.
- K-means with iterative assignment/update for `K=3`: implemented.
- Scatter plot with cluster coloring + centroid markers + labels/legend: implemented.

### Output summary (run with seed = 42)
Final centroids:
- Cluster 0: `[4.9753, 6.1344]`
- Cluster 1: `[0.8643, 0.9280]`
- Cluster 2: `[8.1211, 1.9222]`

Generated figures:
- `K-means/kmeans_clusters.png`
- `K-means/kmeans_elbow.png`

## Part 3 (7 pts)

### 3(a) Elbow curve analysis (WCSS for K = 2..6)
Computed WCSS values:

| K | WCSS |
|---|------:|
| 2 | 974.4308 |
| 3 | 283.7533 |
| 4 | 252.0666 |
| 5 | 227.9890 |
| 6 | 185.4181 |

The elbow curve was plotted in `K-means/kmeans_elbow.png`.

### 3(b) Interpretation
The strongest bend appears at **K = 3**: WCSS drops sharply from 2 to 3, then decreases more slowly afterward.  
This is consistent with the expectation of three generated clusters.

## Part 4 (3 pts)

### 4(a) Why WCSS decreases as K increases
When K increases, each point can be assigned to a centroid that is at least as close as before (never farther in the optimal assignment for that K).  
Therefore, the sum of squared within-cluster distances (WCSS) is non-increasing with K.

### 4(b) Effect of different initial centroids
Observed with `K=3` over seeds `0..9`:

| Seed | Iterations to converge | Final WCSS |
|------|-----------------------:|-----------:|
| 0 | 4 | 283.7533 |
| 1 | 4 | 283.7533 |
| 2 | 5 | 943.2848 |
| 3 | 4 | 946.9119 |
| 4 | 4 | 283.7533 |
| 5 | 3 | 283.7533 |
| 6 | 4 | 283.7533 |
| 7 | 4 | 283.7533 |
| 8 | 5 | 283.7533 |
| 9 | 4 | 283.7533 |

Yes, differences were observed. Most seeds reached a good minimum (WCSS around `283.75`), but some seeds converged to much worse local minima (WCSS around `943-947`).  
This empirically confirms initialization sensitivity and supports using K-means++ and/or multiple restarts.
