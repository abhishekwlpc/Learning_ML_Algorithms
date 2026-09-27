import numpy as np
from sklearn.linear_model import LinearRegression

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2,4,6,8,10], dtype=float)
x = x.reshape(-1, 1)

model = LinearRegression()
model.fit(x, y)

print("Slope (w):", model.coef_[0])
print("Intercept (b):", model.intercept_)
print("R-squared:", model.score(x, y))