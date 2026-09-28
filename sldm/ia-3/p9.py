import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import SplineTransformer
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('housing_prices_spline1.csv')

knots = np.array([[1500], [2000]])
spline_transformer = SplineTransformer(degree=1, knots=knots, include_bias=False)
X_spline = spline_transformer.fit_transform(df['SquareFootage'].values.reshape(-1, 1))

model = LinearRegression()
model.fit(X_spline, df['Price'])

price_pred = model.predict(X_spline)

plt.figure(figsize=(10,6))
plt.scatter(df['SquareFootage'], df['Price'], facecolors='none', edgecolors='blue', label='Data')
plt.plot(df['SquareFootage'], price_pred, color='red', label='Spline Regression Fit')
plt.axvline(x=2000, color='grey', linestyle='--', label='Knot at 2000 sq ft')
plt.xlabel('Square Footage')
plt.ylabel('Price ($)')
plt.title('Housing Prices vs. Square Footage with Spline Regression (scikit-learn)')
plt.legend()
plt.grid(True)
plt.savefig('p9_plot.png')
print("Graph saved as p9_plot.png")
