from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_dataset(csv_path: Path) -> np.ndarray:
    """load csv"""
    return np.loadtxt(csv_path, delimiter=",", skiprows=1)


def assign_clusters(points: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """closest centroid for each point"""
    distances = np.linalg.norm(points[:, np.newaxis, :] - centroids[np.newaxis, :, :], axis=2)
    return np.argmin(distances, axis=1)


def update_centroids(points: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """recompute center of each cluster"""
    new_centroids = centroids.copy()
    for i in range(len(centroids)):
        cluster_points = points[labels == i]
        if len(cluster_points) > 0:
            new_centroids[i] = cluster_points.mean(axis=0)
    return new_centroids


def run_kmeans(points: np.ndarray, k: int = 3, max_iter: int = 100, tol: float = 1e-4, seed: int = 42):
    """basic kmeans loop"""
    rng = np.random.default_rng(seed)
    centroids = points[rng.choice(len(points), size=k, replace=False)]

    for iteration in range(1, max_iter + 1):
        labels = assign_clusters(points, centroids)
        new_centroids = update_centroids(points, labels, centroids)

        shift = np.linalg.norm(new_centroids - centroids)
        print(f"Iteration {iteration:02d} | centroid shift: {shift:.6f}")
        centroids = new_centroids

        if shift < tol:
            print(f"Converged after {iteration} iterations.")
            break

    return labels, centroids


def compute_wcss(points: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
    """within cluster ss"""
    diffs = points - centroids[labels]
    return float(np.sum(diffs ** 2))


def choose_elbow_k(k_values, wcss_values) -> int:
    """pick K from bend in curve"""
    if len(k_values) < 3:
        return int(k_values[0])

    first_diff = np.diff(wcss_values)
    second_diff = np.diff(first_diff)
    elbow_index = int(np.argmax(np.abs(second_diff)) + 1)
    return int(k_values[elbow_index])


def run_elbow_analysis(points: np.ndarray, k_values, max_iter: int = 100, tol: float = 1e-4, seed: int = 42):
    """run kmeans for many K values"""
    wcss_values = []
    for k in k_values:
        labels, centroids = run_kmeans(points, k=k, max_iter=max_iter, tol=tol, seed=seed)
        wcss = compute_wcss(points, labels, centroids)
        wcss_values.append(wcss)
        print(f"K={k} | WCSS={wcss:.4f}")

    return np.array(wcss_values)


def plot_clusters(points: np.ndarray, labels: np.ndarray, centroids: np.ndarray, output_path: Path) -> None:
    """scatter plot with centroids"""
    colors = ["tab:blue", "tab:orange", "tab:green"]

    plt.figure(figsize=(8, 6))
    for i in range(len(centroids)):
        cluster_points = points[labels == i]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            s=35,
            c=colors[i % len(colors)],
            alpha=0.8,
            label=f"Cluster {i}",
        )

    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="red",
        marker="X",
        s=220,
        edgecolors="black",
        linewidths=1,
        label="Final centroids",
    )

    plt.xlabel("Feature1")
    plt.ylabel("Feature2")
    plt.title("K-means Clustering (K=3)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


def plot_elbow_curve(k_values, wcss_values: np.ndarray, output_path: Path) -> None:
    """elbow graph"""
    plt.figure(figsize=(7, 5))
    plt.plot(k_values, wcss_values, marker="o", linestyle="-", color="tab:blue")
    plt.xticks(k_values)
    plt.xlabel("Number of clusters (K)")
    plt.ylabel("WCSS")
    plt.title("Elbow Curve (K-means)")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    data_path = root / "data_cluster.csv"
    output_path_clusters = Path(__file__).resolve().parent / "kmeans_clusters.png"
    output_path_elbow = Path(__file__).resolve().parent / "kmeans_elbow.png"

    # part 2
    points = load_dataset(data_path)
    labels, centroids = run_kmeans(points, k=3)

    print("\nFinal centroids:")
    for i, centroid in enumerate(centroids):
        print(f"Cluster {i}: [{centroid[0]:.4f}, {centroid[1]:.4f}]")

    plot_clusters(points, labels, centroids, output_path_clusters)
    print(f"\nPart 2 plot saved to: {output_path_clusters}")

    # part 3
    print("\nElbow analysis (K = 2..6):")
    k_values = np.array([2, 3, 4, 5, 6])
    wcss_values = run_elbow_analysis(points, k_values)
    plot_elbow_curve(k_values, wcss_values, output_path_elbow)
    print(f"Part 3 elbow plot saved to: {output_path_elbow}")

    best_k = choose_elbow_k(k_values, wcss_values)
    print(f"\nEstimated elbow K: {best_k}")
    if best_k == 3:
        print("Interpretation: The elbow suggests K=3, aligned with the expected three clusters.")
    else:
        print(
            "Interpretation: The elbow suggests a K different from 3; "
            "the curve still decreases but the strongest bend appears there."
        )


if __name__ == "__main__":
    main()
