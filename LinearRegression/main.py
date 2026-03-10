import pandas as pd

from regression_model.regression_model import fit_linear_regression, predict, MSE

df = pd.read_csv("data/Climfrance.csv")

# remove extreme stations
df_filtered = df[~df["station name"].isin(["Mont Ventoux", "Pic du Midi"])]

X = df_filtered[["latitude", "longitude", "altitude"]].values
y = df_filtered["mean temperature"].values

beta = fit_linear_regression(X, y)

print("Regression coefficients:")
print(beta)

y_pred = predict(X, beta)

mse = MSE(y, y_pred)

print("MSE:", mse)