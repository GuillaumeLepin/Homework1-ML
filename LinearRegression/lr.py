import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# load file
df = pd.read_csv(
    "data/Climfrance.csv",
    encoding="latin-1",
    sep=";",
    thousands=","
)

# keep these two aside
mountains = df[df["station"].isin(["Mont Ventoux", "Pic du Midi"])].copy()

# train without mountain extremes
df = df[~df["station"].isin(["Mont Ventoux", "Pic du Midi"])].copy()

X = df[["lat", "lon", "altitude"]].values
y = df["t_mean"].values

# intercept column
X = np.hstack((np.ones((X.shape[0], 1)), X))

XtX = X.T @ X
XtX_inv = np.linalg.inv(XtX)
XtY = X.T @ y
coef = XtX_inv @ XtY

print("Regression coefficients:")
print(coef)

y_pred = X @ coef

residuals = y - y_pred

mse = np.mean(residuals ** 2)
print("MSE:", mse)

n = X.shape[0]
k = X.shape[1]

sigma2 = np.sum(residuals**2) / (n - k)
var_beta = sigma2 * XtX_inv
se_beta = np.sqrt(np.diag(var_beta))

t_stats = coef / se_beta
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=n-k))

print("\np-values:")
print(p_values)

# picked vars for part 2
X2_raw = df[["lat", "altitude"]].values
y2 = df["t_mean"].values

X2 = np.hstack((np.ones((X2_raw.shape[0], 1)), X2_raw))

XtX2 = X2.T @ X2
XtX2_inv = np.linalg.inv(XtX2)
XtY2 = X2.T @ y2
coef2 = XtX2_inv @ XtY2

print("\nNew regression coefficients (lat + altitude):")
print(coef2)

y_pred2 = X2 @ coef2
residuals2 = y2 - y_pred2

mse2 = np.mean(residuals2 ** 2)
print("MSE (Part 2):", mse2)

n2 = X2.shape[0]
k2 = X2.shape[1]
sigma2_2 = np.sum(residuals2**2) / (n2 - k2)

X_mountain_raw = mountains[["lat", "altitude"]].values
y_true = mountains["t_mean"].values
station_names = mountains["station"].values

X_mountain = np.hstack((np.ones((X_mountain_raw.shape[0], 1)), X_mountain_raw))
pred = X_mountain @ coef2

# 95% CI
t_val = stats.t.ppf(0.975, df=n2-k2)

intervals = []

for i in range(X_mountain.shape[0]):
    x_i = X_mountain[i]
    se_pred = np.sqrt(sigma2_2 * (x_i @ XtX2_inv @ x_i.T))
    lower = pred[i] - t_val * se_pred
    upper = pred[i] + t_val * se_pred
    intervals.append((lower, upper))

print("\nPredictions with 95% confidence intervals:")

for i, station in enumerate(station_names):
    print(f"\n{station}")
    print("True temperature:", y_true[i])
    print("Predicted:", pred[i])
    print("95% CI:", intervals[i])

print("\n--- Evaluation of the second model ---")
print(f"a) Mean square error of the second model: {mse2:.6f}")

print("\nb) Model comment:")
if mse2 < 1:
    print("The model has a relatively low MSE, so it fits the data reasonably well.")
else:
    print("The model has a large MSE, so the fit is not very good.")

print("It captures the main trend of temperature decreasing with latitude and altitude,")
print("but it may be less accurate for extreme mountain stations.")

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

lat = df["lat"].values
alt = df["altitude"].values
temp = df["t_mean"].values

ax.scatter(lat, alt, temp, label="Observed data")

ax.set_xlabel("Latitude")
ax.set_ylabel("Altitude")
ax.set_zlabel("Mean Temperature")
ax.set_title("3D Scatterplot of Mean Temperature vs Latitude and Altitude")

plt.legend()
plt.tight_layout()
plt.show()

# with plane too
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(lat, alt, temp, label="Observed data")

lat_grid = np.linspace(lat.min(), lat.max(), 30)
alt_grid = np.linspace(alt.min(), alt.max(), 30)
LAT, ALT = np.meshgrid(lat_grid, alt_grid)

TEMP_pred = coef2[0] + coef2[1] * LAT + coef2[2] * ALT

ax.plot_surface(LAT, ALT, TEMP_pred, alpha=0.5)

ax.set_xlabel("Latitude")
ax.set_ylabel("Altitude")
ax.set_zlabel("Mean Temperature")
ax.set_title("3D Scatterplot with Regression Plane")

plt.legend()
plt.tight_layout()
plt.show()
