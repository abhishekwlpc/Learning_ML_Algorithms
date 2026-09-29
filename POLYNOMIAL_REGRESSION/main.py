import numpy as np

x_original = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 5, 10, 17, 26], dtype=float)

column_1 = x_original
column_2 = x_original **2

X_poly = np.column_stack((column_1, column_2))

w1 = 1.0
w2 =0.0
b=1.0
learning_rate = 0.1
epochs =100

for k in range(epochs):
    prediction_y = w1 * X_poly[:, 0] + w2 * X_poly[:,1] + b

    errors = prediction_y - y

    gradient_w1 = 2 * np.sum(errors * X_poly[:, 0]) / len(y)
    gradient_w2 = 2*  np.sum(errors * X_poly[:, 1]) / len(y)
    gradient_b = 2* np.mean(errors)

    w1 -= learning_rate * gradient_w1
    w2 -= learning_rate * gradient_w2
    b -= learning_rate * gradient_b

    mse = np.mean(errors ** 2)

    print("=============== ", k, " Loop 1 =============")
    print("MSE: ", mse)
    print("W1 Gradient: ", gradient_w1)
    print("W2 Gradient: ", gradient_w2)
    print("b Gradient: ", gradient_b)
    print(" ")

print("Final W1:", w1)
print("Final W2:", w2)
print("Final B:", b)

new_x = 6

prediction = w1 * new_x + w2 * (new_x ** 2) + b

print("Prediction for x=6:", prediction)