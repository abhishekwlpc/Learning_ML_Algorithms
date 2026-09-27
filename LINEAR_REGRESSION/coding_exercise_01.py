import numpy as np

x = np.array([1, 2, 3, 4, 5, 6], dtype=float)
y = np.array([32, 41, 49, 57, 68, 73], dtype=float)

w =1.0
b=1.0

learning_rate = 0.01
epochs = 10000

for i in range(epochs):

    predicted_outputs = w * x + b
    errors = predicted_outputs - y

    w_gradient = 2 / len(x) * (x * errors).sum()
    b_gradient = 2 / len(x) * errors.sum()

    w -= (learning_rate * w_gradient)
    b -= (learning_rate * b_gradient)

    mean_square_error = np.mean(errors ** 2)

    if i % 500 == 0:
        print("=================START HERE===================")
        print("Epoch: ", i)
        print("Predicted Outputs: ", predicted_outputs)
        print("Mean Squared Error: ", mean_square_error)
        print("w: ", w)
        print("b: ", b)
        print("====================================")

