# Homework 1 - Linear Regression

## Part 1

### 1.1 Coefficients of the regression model
Reported coefficients:

`[3.72650364e+01, -5.33960332e-01, 3.21010082e-02, -6.41394600e-03]`

Reported p-values:

`[1.23977051e-10, 4.23646362e-01, 3.17285906e-08]`

### 1.2 Are all 3 attributes relevant?
Reported interpretation: latitude and altitude are the most important variables, while longitude is less relevant (high p-value).

## Part 2

### 2.1 Model with two significant attributes
Reported model MSE (Part 2): `0.4815989858138969`

### 2.2 Predictions and 95% confidence intervals
Mont Ventoux:
- True temperature: `3.6`
- Predicted temperature: `6.187573967687731`
- 95% CI: `(4.3337407740354745, 8.041407161339988)`
- Reported comparison: true value is outside the interval.

Pic du Midi:
- True temperature: `-1.2`
- Predicted temperature: `-3.463726915373703`
- 95% CI: `(-8.134782421498766, 1.2073285907513607)`
- Reported comparison: true value is inside the interval.

## Part 3

### 3(a) Mean square error
Reported MSE: `0.4815989858138969`

### 3(b) 3D scatterplot and model quality
Reported note: run `python main.py` in the `LinearRegression` folder to display the plot.

Reported interpretation:
- The model is generally satisfying (based on MSE).
- It does not fit extreme cases well.
- Removing longitude did not improve the model much.
- The model is adequate, but not perfect.
