import numpy as np


np.set_printoptions(precision=2, suppress=True)

# Dataset
B = np.array([
    [2, 2, 4],
    [2, 4, 6],
    [4, 6, 10],
    [4, 8, 12]
])

print("Dataset:")
print(B)
print("\n")


# Step 1 : Standardize the data

# Mean for each column
mean = np.mean(B, axis=0)

# DataSet centered 
B_centered = B - mean

print("Centered data:")
print(B_centered)
print("\n")


# Step 2 : Covariance

# Covariance data
cov_matrix = np.cov(B_centered, rowvar=False)

print("Covariance data:")
print(cov_matrix)
print("\n")


# Step 3 : EigenVector 


# Eigen decomposition
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("Eigenvalues:")
print(eigenvalues)
print("\n")


# Sort eigenvalues (descending)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

# First principal component
pc1 = eigenvectors[:, 0]

# Projection
B_reduced = B_centered @ pc1

print("Projected data (1D):")
print(B_reduced)


