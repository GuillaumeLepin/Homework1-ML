import numpy as np


np.set_printoptions(precision=2, suppress=True)

# data
B = np.array([
    [2, 2, 4],
    [2, 4, 6],
    [4, 6, 10],
    [4, 8, 12]
])

print("Dataset:")
print(B)
print("\n")


# mean per feature
mean = np.mean(B, axis=0)

# centered version
B_centered = B - mean

print("Centered data:")
print(B_centered)
print("\n")


# covariance
cov_matrix = np.cov(B_centered, rowvar=False)

print("Covariance data:")
print(cov_matrix)
print("\n")


# eig stuff
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:")
print(eigenvalues)
print("\n")


# sort largest first
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# first component
pc1 = eigenvectors[:, 0]

# reduce to 1d
B_reduced = B_centered @ pc1

print("Projected data (1D):")
print(B_reduced)

