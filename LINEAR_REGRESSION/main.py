import numpy as np

x = np.array([1,2,3,4,5], dtype=float)
y = np.array([2.3, 3.7, 6.2, 7.8, 10.5], dtype=float)

w =1.0
b =1.0


learning_rate = 0.01
epochs = 5000

w_new = w
b_new = b

for i in range(epochs):

    prediction = (w_new * x) + b_new

    mse_error = np.mean((prediction - y) ** 2)

    error = prediction - y

    w_gradient = 2/len(x) * (x *error ).sum()

    b_gradient = 2/len(x) * (error.sum())

    w_new = w_new - (learning_rate * w_gradient)

    b_new = b_new - (learning_rate * b_gradient)
    print(i,": |W value:" , w_new, "| B value: ", b_new, "| MSE error: ", mse_error)