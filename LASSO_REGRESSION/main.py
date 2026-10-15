import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)


w=1
b=1
y_pred = w * x+ b
lambda_ = 0.1

def calculate_lasso_loss(y, y_pred, w, lambda_):
    mse = np.mean((y_pred - y) ** 2)
    penalty = lambda_ * np.abs(w)
    loss = mse + penalty
    return loss

def calculate_lasso_gradients(x, y, y_pred, w, lambda_):

    grad_w_mse = 2/len(x) * (np.sum((y_pred - y) * x))
    sign_w = 1
    if w < 0:
        sign_w = -1
    elif w ==0:
        sign_w = 0
    grad_w_penalty = lambda_ * sign_w
    grad_w = grad_w_mse + grad_w_penalty

    grad_b = 2/len(x) * (np.sum(y_pred - y))

    return grad_w, grad_b

learning_rate = 0.01
epochs = 1000

for i in range(epochs):
    y_pred = w * x + b
    loss = calculate_lasso_loss(y, y_pred, w, lambda_)
    gradient_w, gradient_b = calculate_lasso_gradients(x, y, y_pred, w, lambda_)
    print("Epoch: " ,i,": loss: ", loss, " | Gradient: ", gradient_w," | ", gradient_b)

    w = w - learning_rate * gradient_w
    b = b - learning_rate * gradient_b

y_pred = w * x + b
final_loss = calculate_lasso_loss(y, y_pred, w, lambda_)

print("\nFinal weight:", w)
print("Final bias:", b)
print("Final loss:", final_loss)




