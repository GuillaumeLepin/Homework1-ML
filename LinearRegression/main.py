import pandas as pd
import numpy as np
from scipy import stats

# Load dataset
df = pd.read_csv(
    "data/Climfrance.csv",
    encoding="latin-1",
    sep=";",
    thousands=","
)

# Save mountains for later prediction
mountains = df[df["station"].isin(["Mont Ventoux", "Pic du Midi"])]

# Remove mountains for training
df = df[~df["station"].isin(["Mont Ventoux", "Pic du Midi"])]

# -------------------------
# PART 1 REGRESSION
# -------------------------

X = df[["lat", "lon", "altitude"]].values
y = df["t_mean"].values

# Add intercept
X = np.hstack((np.ones((X.shape[0], 1)), X))

# OLS coefficients
XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
XtY = X.T @ y
coef = XtX_inv @ XtY

print("Regression coefficients:")
print(coef)

# Predictions
y_pred = X @ coef

# Residuals
residuals = y - y_pred

# MSE
mse = np.mean(residuals ** 2)
print("MSE:", mse)

# p-values
n = X.shape[0]
k = X.shape[1]

sigma2 = np.sum(residuals**2) / (n - k)
var_beta = sigma2 * XtX_inv
se_beta = np.sqrt(np.diag(var_beta))

t_stats = coef / se_beta
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k))

print("\np-values:")
print(p_values)

# -------------------------
# PART 2 REGRESSION
# -------------------------
# Choose two most significant variables (example: lat and altitude)

X2 = df[["lat", "altitude"]].values
y = df["t_mean"].values

# Add intercept
X2 = np.hstack((np.ones((X2.shape[0], 1)), X2))

# Fit new regression
XtX2 = X2.T @ X2
XtX2_inv = np.linalg.inv(XtX2)
XtY2 = X2.T @ y
coef2 = XtX2_inv @ XtY2

print("\nNew regression coefficients (lat + altitude):")
print(coef2)

# Residuals
y_pred2 = X2 @ coef2
residuals2 = y - y_pred2

# MSE
mse2 = np.mean(residuals2 ** 2)
print("MSE (Part 2):", mse2)

# Residual variance
n2 = X2.shape[0]
k2 = X2.shape[1]
sigma2_2 = np.sum(residuals2**2) / (n2 - k2)


X_mountain = mountains[["lat", "altitude"]].values
y_true = mountains["t_mean"].values

X_mountain = np.hstack((np.ones((X_mountain.shape[0], 1)), X_mountain))

pred = X_mountain @ coef2

# 95% confidence intervals
t_val = stats.t.ppf(0.975, df=n2-k2)

intervals = []

for i in range(X_mountain.shape[0]):
    x_i = X_mountain[i]
    se_pred = np.sqrt(sigma2_2 * (x_i @ XtX2_inv @ x_i.T))
    lower = pred[i] - t_val * se_pred
    upper = pred[i] + t_val * se_pred
    intervals.append((lower, upper))

print("\nPredictions with 95% confidence intervals:")

for i, station in enumerate(mountains["station"].values):
    print(f"\n{station}")
    print("True temperature:", y_true[i])
    print("Predicted:", pred[i])
    print("95% CI:", intervals[i])