import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

X = X.reshape(-1, 1)

model = LinearRegression()

model.fit(X, y)

predictions = model.predict(X)

print("Weight:", model.coef_[0])
print("Bias:", model.intercept_)
print("Predictions:", predictions)