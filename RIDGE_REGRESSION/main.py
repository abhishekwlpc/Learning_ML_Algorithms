import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

w = 1.0
b = 1.0

learning_rate = 0.01
lambda_ = 10
epochs = 5000

for i in range(epochs):
    prediction_of_y = w * x + b
    errors_of_predicted_y = prediction_of_y - y

    mean_squared_error = np.mean(errors_of_predicted_y ** 2)

    ridge_loss = mean_squared_error + (lambda_ * w * w)

    w_gradient = ( 2 / len(x) ) * np.sum(errors_of_predicted_y * x) + (2 * lambda_ * w)
    b_gradient = 2 * np.mean(errors_of_predicted_y)


    w = w - (learning_rate * w_gradient)
    b = b - (learning_rate * b_gradient)

    print("=============== ", i, " Loop =============")
    print("MSE: ", mean_squared_error)
    print("Ridge Loss: ", ridge_loss)
    print("W : ", w)
    print("b : ", b)
    print(" ")

print("Final W1:", w)
print("Final B:", b)


